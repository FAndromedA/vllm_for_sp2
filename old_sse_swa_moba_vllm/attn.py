# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Converted to vllm implementation by AI Assistant
from __future__ import annotations
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.utils import logging
import vllm.envs as envs
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.selector import get_attn_backend
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group,
                                          is_v1_kv_transfer_group)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod

from fla.modules import RMSNorm as FLARMSNorm
from fla.modules import RotaryEmbedding as FLARotaryEmbedding
from vllm.platforms import current_platform
from fla.layers.utils import pad_input, unpad_input
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = None
logger = logging.get_logger(__name__)
class VLLMAttention(nn.Module):
    """
    vllm-compatible Attention layer implementation with qk_norm support
    """
    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        head_dim: int = 128,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: int | None = None,
        rope_theta: float | None = 10000.,
        max_position_embeddings: int | None = None,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        cache_config: Optional[CacheConfig] | None = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = head_dim
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx
        self.prefix = prefix
        self.norm_eps = norm_eps
        # vllm-specific parameters
        self.scaling = self.head_dim ** -0.5
        self.window_size = window_size
        self.cache_config = cache_config or CacheConfig()
        self.quant_config = quant_config
        # Tensor parallel configuration
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        
        # Validate tensor parallel configuration
        assert self.num_heads % self.tp_size == 0, \
            f"num_heads ({self.num_heads}) must be divisible by tp_size ({self.tp_size})"
        
        if self.num_kv_heads >= self.tp_size:
            assert self.num_kv_heads % self.tp_size == 0, \
                f"num_kv_heads ({self.num_kv_heads}) must be divisible by tp_size ({self.tp_size})"
        else:
            assert self.tp_size % self.num_kv_heads == 0, \
                f"tp_size ({self.tp_size}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        
        self.tp_heads = self.num_heads // self.tp_size
        self.tp_kv_heads = max(1, self.num_kv_heads // self.tp_size)
        if flash_attn_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")
        # QKV projection layers using vllm's parallel linear layers
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_kv_heads,
            bias=self.qkv_bias,
            quant_config=self.quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        # Output projection layer
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.o_proj",
        )
        # qk_norm support
        if self.qk_norm:
            self.q_norm = FLARMSNorm(self.head_dim, eps=self.norm_eps)
            self.k_norm = FLARMSNorm(self.head_dim, eps=self.norm_eps)
        # 使用fla的RotaryEmbedding替代vllm的get_rope
        self.rotary_emb = FLARotaryEmbedding(
            dim=self.head_dim,
            base=rope_theta if rope_theta is not None else 10000.0
        )
    def load_hf_weights(self, hf_weights: dict, prefix: str = ""):
        """
        Load weights from huggingface implementation.
        
        Args:
            hf_weights: Dictionary of huggingface weights
            prefix: Prefix for weight names (e.g., "model.layers.0.attn.")
        """
        # Map huggingface's separate q_proj, k_proj, v_proj to vllm's qkv_proj
        q_proj_key = f"{prefix}q_proj.weight"
        k_proj_key = f"{prefix}k_proj.weight" 
        v_proj_key = f"{prefix}v_proj.weight"
        
        if q_proj_key in hf_weights and k_proj_key in hf_weights and v_proj_key in hf_weights:
            # Get individual projections
            q_weight = hf_weights[q_proj_key]
            k_weight = hf_weights[k_proj_key]
            v_weight = hf_weights[v_proj_key]
            
            # Combine into qkv_proj weight
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            
            # Load into qkv_proj
            if hasattr(self.qkv_proj, 'weight'):
                self.qkv_proj.weight.data.copy_(qkv_weight)
            
            # Handle biases if present
            if self.qkv_bias:
                q_bias_key = f"{prefix}q_proj.bias"
                k_bias_key = f"{prefix}k_proj.bias"
                v_bias_key = f"{prefix}v_proj.bias"
                
                if q_bias_key in hf_weights and k_bias_key in hf_weights and v_bias_key in hf_weights:
                    q_bias = hf_weights[q_bias_key]
                    k_bias = hf_weights[k_bias_key]
                    v_bias = hf_weights[v_bias_key]
                    
                    qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                    
                    if hasattr(self.qkv_proj, 'bias') and self.qkv_proj.bias is not None:
                        self.qkv_proj.bias.data.copy_(qkv_bias)
        
        # Load o_proj
        o_proj_key = f"{prefix}o_proj.weight"
        if o_proj_key in hf_weights:
            if hasattr(self.o_proj, 'weight'):
                self.o_proj.weight.data.copy_(hf_weights[o_proj_key])
        
        # Load q_norm and k_norm if present
        if self.qk_norm:
            q_norm_key = f"{prefix}q_norm.weight"
            k_norm_key = f"{prefix}k_norm.weight"
            
            if q_norm_key in hf_weights and hasattr(self.q_norm, 'weight'):
                self.q_norm.weight.data.copy_(hf_weights[q_norm_key])
            
            if k_norm_key in hf_weights and hasattr(self.k_norm, 'weight'):
                self.k_norm.weight.data.copy_(hf_weights[k_norm_key])
        
        logger.info(f"Loaded attention weights for {prefix}")
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        past_key_values: dict | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict | None]:
        """
        vllm-compatible forward pass with qk_norm support
        """
        batch_size, q_len, _ = hidden_states.size()
        
        # Get vllm forward context
        forward_context: ForwardContext = get_forward_context()
        
        # Use QKVParallelLinear for projection
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        # Reshape for tensor parallel
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.tp_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.tp_kv_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.tp_kv_heads)
        # Apply qk_norm if enabled
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # 调整fla的RotaryEmbedding调用方式
        seq_len = positions.size(1) if positions.dim() > 1 else positions.size(0)
        q, k = self.rotary_emb(q, k, seqlen_offset=0, max_seqlen=seq_len)
        # Handle KV cache update
        if use_cache and past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            # Flatten for cache storage
            k_flat = rearrange(k, 'b s h d -> b s (h d)')
            v_flat = rearrange(v, 'b s h d -> b s (h d)')
            
            k_cached, v_cached = past_key_values.update(
                attn_state=(k_flat, v_flat),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size),
            )['attn_state']
            
            if cache_has_content:
                k = rearrange(k_cached, 'b s (h d) -> b s h d', h=self.tp_kv_heads)
                v = rearrange(v_cached, 'b s (h d) -> b s h d', h=self.tp_kv_heads)
        # Attention computation with flash attention
        cu_seqlens = None
        max_seqlen = None
        
        # Get cu_seqlens from forward context if available
        if hasattr(forward_context, 'cu_seqlens'):
            cu_seqlens = forward_context.cu_seqlens
            max_seqlen = forward_context.max_seqlen
        if attention_mask is not None:
            if q.shape[1] == 1 and self.window_size is not None:
                attention_mask = attention_mask[:, -self.window_size:]
            q_unpad, (k_unpad, v_unpad), indices_q, cu_seqlens, max_seq_lens = unpad_input(
                q, (k, v), attention_mask, q_len)
            cu_seqlens_q, cu_seqlens_k = cu_seqlens
            max_seqlen_q, max_seqlen_k = max_seq_lens
            
            o = flash_attn_varlen_func(
                q_unpad, k_unpad, v_unpad,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0),
            )
            o = pad_input(o, indices_q, batch_size, q_len)
        elif cu_seqlens is not None:
            # Handle packed sequences
            o = flash_attn_varlen_func(
                q.squeeze(0), k.squeeze(0), v.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0),
            ).unsqueeze(0)
        else:
            o = flash_attn_func(
                q, k, v,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0),
            )
        # Reshape and apply output projection using RowParallelLinear
        o = rearrange(o, 'b s h d -> b s (h d)')
        output, _ = self.o_proj(o)  # RowParallelLinear returns (output, bias)
        return output, None, past_key_values
    
class VLLMAttentionWrapper(nn.Module):
    """
    Wrapper class to provide vllm-compatible interface for the attention layer.
    This handles the unified attention interface expected by vllm.
    """
    def __init__(
        self,
        attention: VLLMAttention,
        layer_name: str = "",
    ):
        super().__init__()
        self.attention = attention
        self.layer_name = layer_name
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query2: Optional[torch.Tensor] = None,
        key2: Optional[torch.Tensor] = None,
        value2: Optional[torch.Tensor] = None,
        output_shape: Optional[torch.Size] = None,
    ) -> torch.Tensor:
        """
        vllm unified attention interface with parallel linear layers support.
        """
        # Get vllm forward context
        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        # Get past key values from context if available
        past_key_values = None
        if hasattr(forward_context, 'past_key_values'):
            past_key_values = forward_context.past_key_values
        # For vllm parallel implementation, the input query, key, value are already
        # projected and partitioned across tensor parallel devices
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        
        # Reshape to (batch, seq_len, heads, head_dim) using tensor parallel heads
        q = rearrange(query, 'b s (h d) -> b s h d', h=self.attention.tp_heads)
        k = rearrange(key, 'b s (h d) -> b s h d', h=self.attention.tp_kv_heads)
        v = rearrange(value, 'b s (h d) -> b s h d', h=self.attention.tp_kv_heads)
        
        # Apply rotary embedding
        seqlen_offset = attn_metadata.seqlen_offset if attn_metadata else 0
        max_seqlen = seq_len + seqlen_offset
        
        q, k = self.attention.rotary_emb(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen)
        
        # Handle attention mask from metadata
        attention_mask = None
        if attn_metadata and hasattr(attn_metadata, 'attention_mask'):
            attention_mask = attn_metadata.attention_mask
        # Perform attention calculation
        if attention_mask is not None:
            if q.shape[1] == 1 and self.attention.window_size is not None:
                attention_mask = attention_mask[:, -self.attention.window_size:]
            q_unpad, (k_unpad, v_unpad), indices_q, cu_seqlens, max_seq_lens = unpad_input(
                q, (k, v), attention_mask, q.shape[1])
            cu_seqlens_q, cu_seqlens_k = cu_seqlens
            max_seqlen_q, max_seqlen_k = max_seq_lens
            
            o = flash_attn_varlen_func(
                q_unpad, k_unpad, v_unpad,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=True,
                window_size=(-1, -1) if self.attention.window_size is None else (self.attention.window_size-1, 0),
            )
            o = pad_input(o, indices_q, batch_size, q.shape[1])
        else:
            o = flash_attn_func(
                q, k, v,
                causal=True,
                window_size=(-1, -1) if self.attention.window_size is None else (self.attention.window_size-1, 0),
            )
        # Reshape and apply output projection using RowParallelLinear
        o = rearrange(o, 'b s h d -> b s (h d)')
        o, _ = self.attention.o_proj(o)  # RowParallelLinear returns (output, bias)
        return o
    
class KVCacheManager:
    """
    Custom KV Cache Manager implementation for vllm compatibility with parallel linear layers.
    This manages the KV cache storage and retrieval in tensor parallel setting.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cuda"),
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = device
        
        # Tensor parallel configuration
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size
        )
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        
        # Calculate per-device KV heads
        if self.num_kv_heads >= self.tp_size:
            assert self.num_kv_heads % self.tp_size == 0, \
                f"num_kv_heads ({self.num_kv_heads}) must be divisible by tp_size ({self.tp_size})"
            self.tp_kv_heads = self.num_kv_heads // self.tp_size
        else:
            assert self.tp_size % self.num_kv_heads == 0, \
                f"tp_size ({self.tp_size}) must be divisible by num_kv_heads ({self.num_kv_heads})"
            self.tp_kv_heads = 1
        
        # Initialize KV cache storage for this tensor parallel device
        self.k_cache = torch.empty(
            (num_layers, max_batch_size, max_seq_len, self.tp_kv_heads, head_dim),
            dtype=dtype,
            device=device
        )
        self.v_cache = torch.empty(
            (num_layers, max_batch_size, max_seq_len, self.tp_kv_heads, head_dim),
            dtype=dtype,
            device=device
        )
        
        # Sequence lengths tracking
        self.seq_lens = torch.zeros(
            (num_layers, max_batch_size),
            dtype=torch.int32,
            device=device
        )
    
    def get_seq_length(self, layer_idx: int) -> int:
        """Get the maximum sequence length for a given layer"""
        return torch.max(self.seq_lens[layer_idx]).item()
    
    def update(
        self,
        attn_state: tuple[torch.Tensor, torch.Tensor],
        layer_idx: int,
        offset: int,
        cache_kwargs: dict = None,
    ) -> dict:
        """Update the KV cache with new states from parallel linear layers"""
        k, v = attn_state
        batch_size = k.shape[0]
        seq_len = k.shape[1]
        
        # Reshape KV to (batch, seq_len, heads, head_dim)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.tp_kv_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.tp_kv_heads)
        
        # Update cache
        if offset + seq_len > self.max_seq_len:
            # Need to expand cache if necessary
            new_max_seq_len = max(self.max_seq_len * 2, offset + seq_len)
            self._expand_cache(new_max_seq_len)
        
        self.k_cache[layer_idx, :batch_size, offset:offset+seq_len] = k
        self.v_cache[layer_idx, :batch_size, offset:offset+seq_len] = v
        
        # Update sequence lengths
        self.seq_lens[layer_idx, :batch_size] = torch.max(
            self.seq_lens[layer_idx, :batch_size],
            torch.full((batch_size,), offset + seq_len, device=self.device, dtype=torch.int32)
        )
        
        # Return updated cache in the format expected by QKVParallelLinear
        return {
            'attn_state': (
                rearrange(self.k_cache[layer_idx, :batch_size], 'b s h d -> b s (h d)'),
                rearrange(self.v_cache[layer_idx, :batch_size], 'b s h d -> b s (h d)')
            )
        }
    
    def _expand_cache(self, new_max_seq_len: int):
        """Expand the KV cache to accommodate longer sequences"""
        logger.info(f"Expanding KV cache from {self.max_seq_len} to {new_max_seq_len}")
        
        new_k_cache = torch.empty(
            (self.num_layers, self.max_batch_size, new_max_seq_len, self.tp_kv_heads, self.head_dim),
            dtype=self.dtype,
            device=self.device
        )
        new_v_cache = torch.empty(
            (self.num_layers, self.max_batch_size, new_max_seq_len, self.tp_kv_heads, self.head_dim),
            dtype=self.dtype,
            device=self.device
        )
        
        # Copy existing data
        new_k_cache[:, :, :self.max_seq_len] = self.k_cache
        new_v_cache[:, :, :self.max_seq_len] = self.v_cache
        
        self.k_cache = new_k_cache
        self.v_cache = new_v_cache
        self.max_seq_len = new_max_seq_len
    
    def reset(self):
        """Reset the KV cache"""
        self.seq_lens.zero_()