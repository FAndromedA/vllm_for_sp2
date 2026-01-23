# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Converted to vllm implementation by AI Assistant

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

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
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op

from fla.layers.utils import pad_input, unpad_input
from fla.ops.utils.index import prepare_lens_from_mask

# 复用之前定义的KVCacheManager
from .attn import KVCacheManager

if TYPE_CHECKING:
    from fla.models.utils import Cache

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = None

try:
    from moba import moba_attn_varlen
except ImportError:
    warnings.warn(
        "MoBA is not installed. Please install it first",
        category=ImportWarning,
    )
    moba_attn_varlen = None

logger = logging.get_logger(__name__)


class VLLMMoBAAttention(nn.Module):
    """
    vllm-compatible MoBAAttention layer implementation.
    This class maintains the original MoBA + FlashAttention logic while adding vllm compatibility.
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
        moba_chunk_size: int = 1024,
        moba_topk: int = 4,
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
        
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.moba_chunk_size = moba_chunk_size
        self.moba_topk = moba_topk
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx
        self.prefix = prefix
        self.norm_eps = norm_eps

        # vllm-specific parameters
        self.scaling = self.head_dim ** -0.5

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
        
        if moba_attn_varlen is None:
            raise ImportError("Please install MoBA first")

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
            self.q_norm = RMSNorm(self.head_dim, eps=self.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=self.norm_eps)

        # vllm's RoPE implementation
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
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
        past_key_values: Union[KVCacheManager, dict, None] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Union[KVCacheManager, dict, None]]:
        """
        vllm-compatible forward pass with MoBA + FlashAttention logic.
        
        Attention selection logic:
        - Training phase: q_len == kv_len, use MoBA attention
        - Inference prefill: q_len == kv_len, use MoBA attention  
        - Inference decode: q_len != kv_len, use FlashAttention
        """
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

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

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # equivalent to cu_seqlens in `flash_attn`
        cu_seqlens = kwargs.get('cu_seqlens')

        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            # Get sequence length from KVCacheManager
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                # to deliminate the offsets of padding tokens
                seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        
        # Apply RoPE using vllm's implementation
        q, k = self.rotary_emb(positions, q, k)

        # Handle KV cache update
        if use_cache and past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            
            # Flatten for cache storage (batch, seq_len, tp_kv_heads * head_dim)
            k_flat = rearrange(k, 'b s h d -> b s (h d)')
            v_flat = rearrange(v, 'b s h d -> b s (h d)')
            
            # Update cache using KVCacheManager
            update_result = past_key_values.update(
                attn_state=(k_flat, v_flat),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size),
            )
            
            if 'attn_state' in update_result:
                k_cached, v_cached = update_result['attn_state']
                
                if cache_has_content:
                    # Reshape back to (batch, seq_len, tp_kv_heads, head_dim)
                    k = rearrange(k_cached, 'b s (h d) -> b s h d', h=self.tp_kv_heads)
                    v = rearrange(v_cached, 'b s (h d) -> b s h d', h=self.tp_kv_heads)
        
        # Repeat KV heads for multi-query attention
        k = torch.repeat_interleave(k, self.num_kv_groups, dim=2)
        v = torch.repeat_interleave(v, self.num_kv_groups, dim=2)

        # Determine attention type based on sequence lengths
        # This differentiates between training/prefill (q_len == kv_len) and decode (q_len != kv_len)
        kv_len = k.size(1)
        
        if q_len == kv_len:
            # ==============================================
            # Training phase or inference prefill phase
            # Use MoBA attention for better performance
            # ==============================================
            logger.debug(f"Using MoBA attention (q_len={q_len}, kv_len={kv_len}) - Training/Prefill phase")
            
            if attention_mask is not None:
                if q.shape[1] == 1 and self.window_size is not None:
                    attention_mask = attention_mask[:, -self.window_size:]
                q_unpad, (k_unpad, v_unpad), indices_q, cu_seqlens, max_seq_lens = unpad_input(
                    q, (k, v), attention_mask, q_len)
                cu_seqlens_q, cu_seqlens_k = cu_seqlens
                max_seqlen_q, max_seqlen_k = max_seq_lens
                
                o = moba_attn_varlen(
                    q=q_unpad,
                    k=k_unpad,
                    v=v_unpad,
                    cu_seqlens=cu_seqlens_k,
                    max_seqlen=max_seqlen_k,
                    moba_chunk_size=self.moba_chunk_size,
                    moba_topk=self.moba_topk,
                )
                o = pad_input(o, indices_q, batch_size, q_len)
            elif cu_seqlens is not None:
                o = moba_attn_varlen(
                    q=q.squeeze(0),
                    k=k.squeeze(0),
                    v=v.squeeze(0),
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    moba_chunk_size=self.moba_chunk_size,
                    moba_topk=self.moba_topk,
                ).unsqueeze(0)
            else:
                cu_seqlens_k = torch.cumsum(
                    torch.tensor([0] + [kv_len] * batch_size, device=q.device),
                    dim=0,
                    dtype=torch.int32,
                )
                q_flat, k_flat, v_flat = [rearrange(src, 'b l h d -> (b l) h d').contiguous() 
                                         for src in [q, k, v]]
                
                o = moba_attn_varlen(
                    q=q_flat,
                    k=k_flat,
                    v=v_flat,
                    cu_seqlens=cu_seqlens_k,
                    max_seqlen=kv_len,
                    moba_chunk_size=self.moba_chunk_size,
                    moba_topk=self.moba_topk,
                )
                o = rearrange(o, "(b l) h d -> b l h d", b=batch_size).contiguous()

        else:
            # ==============================================
            # Inference decode phase
            # Use FlashAttention for faster decoding
            # ==============================================
            logger.debug(f"Using FlashAttention (q_len={q_len}, kv_len={kv_len}) - Decode phase")
            
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

        if not output_attentions:
            attentions = None

        return output, attentions, past_key_values


class VLLMMoBAAttentionWrapper(nn.Module):
    """
    Wrapper class to provide vllm-compatible interface for the MoBA attention layer.
    """

    def __init__(
        self,
        attention: VLLMMoBAAttention,
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
        vllm unified attention interface with MoBA support.
        """
        # Get vllm forward context
        forward_context: ForwardContext = get_forward_context()
        
        # Get positions from context if available
        positions = None
        if hasattr(forward_context, 'positions'):
            positions = forward_context.positions
        else:
            # Generate default positions if not provided
            batch_size, seq_len = query.shape[:2]
            positions = torch.arange(seq_len, device=query.device).unsqueeze(0).repeat(batch_size, 1)

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
        
        # Apply rotary embedding using vllm's implementation
        q, k = self.attention.rotary_emb(positions, q, k)
        
        # Handle attention mask from context
        attention_mask = None
        if hasattr(forward_context, 'attention_mask'):
            attention_mask = forward_context.attention_mask

        # Get cu_seqlens from forward context if available
        cu_seqlens = None
        max_seqlen = None
        if hasattr(forward_context, 'cu_seqlens'):
            cu_seqlens = forward_context.cu_seqlens
            max_seqlen = forward_context.max_seqlen

        # Repeat KV heads for multi-query attention
        k = torch.repeat_interleave(k, self.attention.num_kv_groups, dim=2)
        v = torch.repeat_interleave(v, self.attention.num_kv_groups, dim=2)

        # Determine attention type based on sequence lengths
        kv_len = k.size(1)
        
        if q.shape[1] == kv_len:
            # Training/prefill phase - use MoBA
            if attention_mask is not None:
                if q.shape[1] == 1 and self.attention.window_size is not None:
                    attention_mask = attention_mask[:, -self.attention.window_size:]
                q_unpad, (k_unpad, v_unpad), indices_q, cu_seqlens, max_seq_lens = unpad_input(
                    q, (k, v), attention_mask, q.shape[1])
                cu_seqlens_q, cu_seqlens_k = cu_seqlens
                max_seqlen_q, max_seqlen_k = max_seq_lens
                
                o = moba_attn_varlen(
                    q=q_unpad,
                    k=k_unpad,
                    v=v_unpad,
                    cu_seqlens=cu_seqlens_k,
                    max_seqlen=max_seqlen_k,
                    moba_chunk_size=self.attention.moba_chunk_size,
                    moba_topk=self.attention.moba_topk,
                )
                o = pad_input(o, indices_q, batch_size, q.shape[1])
            elif cu_seqlens is not None:
                o = moba_attn_varlen(
                    q=q.squeeze(0),
                    k=k.squeeze(0),
                    v=v.squeeze(0),
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    moba_chunk_size=self.attention.moba_chunk_size,
                    moba_topk=self.attention.moba_topk,
                ).unsqueeze(0)
            else:
                cu_seqlens_k = torch.cumsum(
                    torch.tensor([0] + [kv_len] * batch_size, device=q.device),
                    dim=0,
                    dtype=torch.int32,
                )
                q_flat, k_flat, v_flat = [rearrange(src, 'b l h d -> (b l) h d').contiguous() 
                                         for src in [q, k, v]]
                
                o = moba_attn_varlen(
                    q=q_flat,
                    k=k_flat,
                    v=v_flat,
                    cu_seqlens=cu_seqlens_k,
                    max_seqlen=kv_len,
                    moba_chunk_size=self.attention.moba_chunk_size,
                    moba_topk=self.attention.moba_topk,
                )
                o = rearrange(o, "(b l) h d -> b l h d", b=batch_size).contiguous()
        else:
            # Decode phase - use FlashAttention
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
            elif cu_seqlens is not None:
                o = flash_attn_varlen_func(
                    q.squeeze(0), k.squeeze(0), v.squeeze(0),
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    causal=True,
                    window_size=(-1, -1) if self.attention.window_size is None else (self.attention.window_size-1, 0),
                ).unsqueeze(0)
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


# Register custom ops for vllm
# def unified_vllm_moba_attention(
#     query: torch.Tensor,
#     key: torch.Tensor,
#     value: torch.Tensor,
#     query2: Optional[torch.Tensor] = None,
#     key2: Optional[torch.Tensor] = None,
#     value2: Optional[torch.Tensor] = None,
#     output_shape: Optional[torch.Size] = None,
# ) -> torch.Tensor:
#     """Unified attention op for vllm with MoBA support."""
#     forward_context = get_forward_context()
#     wrapper: VLLMMoBAAttentionWrapper = forward_context.layer
#     return wrapper(query, key, value, query2, key2, value2, output_shape)


# def unified_vllm_moba_attention_fake(
#     query: torch.Tensor,
#     key: torch.Tensor,
#     value: torch.Tensor,
#     query2: Optional[torch.Tensor] = None,
#     key2: Optional[torch.Tensor] = None,
#     value2: Optional[torch.Tensor] = None,
#     output_shape: Optional[torch.Size] = None,
# ) -> torch.Tensor:
#     """Fake op for vllm graph capture with MoBA support."""
#     batch_size, seqlen, _ = query.shape
#     return torch.empty(
#         batch_size, seqlen, output_shape[-1],
#         device=query.device, dtype=query.dtype)


# # Register the custom ops
# direct_register_custom_op(
#     unified_vllm_moba_attention,
#     "unified_vllm_moba_attention",
#     unified_vllm_moba_attention_fake,
# )


__all__ = [
    "VLLMMoBAAttention",
    "VLLMMoBAAttentionWrapper",
    "KVCacheManager",
    # "unified_vllm_moba_attention",
]