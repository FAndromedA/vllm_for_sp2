# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Converted to vllm implementation by AI Assistant
# Only modified linear layers to vllm's parallel linear layers, kept other logic unchanged

from __future__ import annotations
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.forward_context import ForwardContext, get_forward_context

from transformers.processing_utils import Unpack
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution, RotaryEmbedding
from fla.ops.gla import chunk_gla, fused_recurrent_gla
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from fla.ops.sse import prepare_sample_relpos_global_index_flat, softmax_and_mask, sort_along_l
from fla.layers.utils import pad_input, unpad_input, prepare_lens_from_mask
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.models.utils import Cache

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = None

logger = envs.get_logger(__name__)


class VLLMSSESWAGLA(nn.Module):
    """
    The layer implementation for [SSE: Scaling Linear Attention with Sparse State Expansion](https://arxiv.org/pdf/2507.16577).
    This is a vllm-compatible implementation with parallel linear layers.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1.,
        head_dim: int = 256,
        num_heads: int = 6,
        mode: str = 'chunk',
        use_output_gate: bool = True,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        num_sparse_partition: int = 4,
        num_writer: int = 1,
        num_reader: int = 1,
        sse_implementation: str = "varlen",
        sse_qk_relu: bool = False,
        use_q_softmax: bool = False,
        use_k_softmax: bool = True,
        emulq: bool = True,
        emulk: bool = True,
        gate_logit_normalizer: int = 16,
        gate_low_rank_dim: int = 16,
        qkv_bias: bool = False,
        # === swa configs ===
        swa_num_kv_heads: Optional[int] = None,
        swa_qk_norm: bool = False,
        swa_dropout: float = 0.5,
        window_size: Optional[int] = None,
        rope_theta: Optional[float] = 10000.,
        max_position_embeddings: Optional[int] = None,
        # ===================
        layer_idx: Optional[int] = None,
        norm_eps: float = 1e-5,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.layer_idx = layer_idx

        assert num_reader < num_sparse_partition and num_writer < num_sparse_partition, \
            "num_reader and num_writer must be less than num_sparse_partition."
        assert sse_implementation in ["mask", "varlen"], \
            f"Unknown SSE implementation {sse_implementation}"

        self.num_sparse_partition = num_sparse_partition
        self.num_writer = num_writer
        self.num_reader = num_reader
        self.sse_implementation = {
            "mask": self.sse_linear_attention_mask,
            "varlen": self.sse_linear_attention_varlen,
        }[sse_implementation]

        self.use_output_gate = use_output_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.use_q_softmax = use_q_softmax
        self.use_k_softmax = use_k_softmax
        self.sse_qk_relu = sse_qk_relu
        self.emulq = emulq
        self.emulk = emulk

        # MHA for SSE, GQA for SWA
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.sse_num_kv_heads = num_heads
        self.swa_num_kv_heads = swa_num_kv_heads if swa_num_kv_heads is not None else num_heads
        self.swa_num_kv_groups = num_heads // self.swa_num_kv_heads

        self.sse_head_k_dim = head_dim
        self.sse_head_v_dim = int(self.head_dim * self.expand_v)
        self.sse_key_dim = int(self.sse_num_kv_heads * self.sse_head_k_dim)
        self.sse_value_dim = int(self.sse_num_kv_heads * self.sse_head_v_dim)

        self.swa_q_dim = self.num_heads * self.head_dim
        self.swa_kv_dim = int(self.swa_num_kv_heads * self.head_dim)

        self.swa_qk_norm = swa_qk_norm
        self.qkv_bias = qkv_bias
        self.swa_dropout = swa_dropout
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        # Tensor parallel configuration
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.quant_config = quant_config
        self.prefix = prefix

        # Validate tensor parallel configuration
        assert self.num_heads % self.tp_size == 0, \
            f"num_heads ({self.num_heads}) must be divisible by tp_size ({self.tp_size})"
        
        if self.swa_num_kv_heads >= self.tp_size:
            assert self.swa_num_kv_heads % self.tp_size == 0, \
                f"swa_num_kv_heads ({self.swa_num_kv_heads}) must be divisible by tp_size ({self.tp_size})"
        else:
            assert self.tp_size % self.swa_num_kv_heads == 0, \
                f"tp_size ({self.tp_size}) must be divisible by swa_num_kv_heads ({self.swa_num_kv_heads})"

        self.tp_heads = self.num_heads // self.tp_size
        self.tp_swa_kv_heads = max(1, self.swa_num_kv_heads // self.tp_size)
        self.tp_sse_kv_heads = max(1, self.sse_num_kv_heads // self.tp_size)

        if flash_attn_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")

        # SSE projections using vllm's ColumnParallelLinear
        self.sse_q_proj = ColumnParallelLinear(
            hidden_size, self.sse_key_dim, bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.sse_q_proj"
        )
        self.sse_k_proj = ColumnParallelLinear(
            hidden_size, self.sse_key_dim, bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.sse_k_proj"
        )
        self.sse_v_proj = ColumnParallelLinear(
            hidden_size, self.sse_value_dim, bias=self.qkv_bias,
            quant_config=self.quant_config,
            prefix=f"{prefix}.sse_v_proj"
        )

        # LoRA projections using vllm's ColumnParallelLinear
        self.lora_q_proj_0 = ColumnParallelLinear(
            hidden_size, self.sse_head_v_dim, bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.lora_q_proj.0"
        )
        self.lora_q_proj_1 = ColumnParallelLinear(
            self.sse_head_v_dim, self.sse_key_dim, bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.lora_q_proj.1"
        )

        self.lora_k_proj_0 = ColumnParallelLinear(
            hidden_size, self.sse_head_v_dim, bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.lora_k_proj.0"
        )
        self.lora_k_proj_1 = ColumnParallelLinear(
            self.sse_head_v_dim, self.sse_key_dim, bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.lora_k_proj.1"
        )

        # Gate projections using vllm's ColumnParallelLinear
        self.gate_logit_normalizer = gate_logit_normalizer
        self.gk_proj_0 = ColumnParallelLinear(
            hidden_size, gate_low_rank_dim, bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.gk_proj.0.0"
        )
        self.gk_proj_1 = ColumnParallelLinear(
            gate_low_rank_dim, self.sse_key_dim, bias=True,
            quant_config=self.quant_config,
            prefix=f"{prefix}.gk_proj.0.1"
        )
        self.gk_proj_2 = ColumnParallelLinear(
            hidden_size, gate_low_rank_dim, bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.gk_proj.1.0"
        )
        self.gk_proj_3 = ColumnParallelLinear(
            gate_low_rank_dim, self.sse_key_dim, bias=True,
            quant_config=self.quant_config,
            prefix=f"{prefix}.gk_proj.1.1"
        )

        self.e_proj = ColumnParallelLinear(
            hidden_size, self.num_sparse_partition, bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.e_proj"
        )

        # SWA projections using vllm's QKVParallelLinear
        self.swa_qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.swa_num_kv_heads,
            bias=self.qkv_bias,
            quant_config=self.quant_config,
            prefix=f"{prefix}.swa_qkv_proj",
        )

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d_shared = ShortConvolution(
                hidden_size=self.sse_key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation=None,
            )
            self.k_conv1d_shared = ShortConvolution(
                hidden_size=self.sse_key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation=None,
            )

        if use_output_gate:
            # Output gate projection using vllm's ColumnParallelLinear
            self.sse_g_proj_0 = ColumnParallelLinear(
                hidden_size, self.sse_head_v_dim, bias=False,
                quant_config=self.quant_config,
                prefix=f"{prefix}.sse_g_proj.0"
            )
            self.sse_g_proj_1 = ColumnParallelLinear(
                self.sse_head_v_dim, self.sse_value_dim, bias=False,
                quant_config=self.quant_config,
                prefix=f"{prefix}.sse_g_proj.1"
            )
            self.sse_o_norm = FusedRMSNormGated(self.sse_head_v_dim, eps=norm_eps)
        else:
            self.sse_o_norm = RMSNorm(self.sse_head_v_dim, eps=norm_eps)

        # Output projections using vllm's RowParallelLinear
        self.sse_o_proj = RowParallelLinear(
            self.sse_value_dim, self.hidden_size, bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.sse_o_proj"
        )
        self.swa_o_proj = RowParallelLinear(
            self.swa_q_dim, self.hidden_size, bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.swa_o_proj"
        )

        self.sse_merge_norm = RMSNorm(self.hidden_size, eps=norm_eps)
        self.swa_merge_norm = RMSNorm(self.hidden_size, eps=norm_eps)

        if swa_qk_norm:
            self.swa_q_norm = RMSNorm(self.head_dim, eps=norm_eps)
            self.swa_k_norm = RMSNorm(self.head_dim, eps=norm_eps)

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)

    def load_hf_weights(self, hf_weights: dict, prefix: str = ""):
        """Load weights from huggingface implementation."""
        # Load SSE projections
        self._load_parallel_linear_weights(hf_weights, prefix, "sse_q_proj")
        self._load_parallel_linear_weights(hf_weights, prefix, "sse_k_proj")
        self._load_parallel_linear_weights(hf_weights, prefix, "sse_v_proj")

        # Load LoRA projections
        self._load_parallel_linear_weights(hf_weights, prefix, "lora_q_proj.0")
        self._load_parallel_linear_weights(hf_weights, prefix, "lora_q_proj.1")
        self._load_parallel_linear_weights(hf_weights, prefix, "lora_k_proj.0")
        self._load_parallel_linear_weights(hf_weights, prefix, "lora_k_proj.1")

        # Load gate projections
        self._load_parallel_linear_weights(hf_weights, prefix, "gk_proj.0.0")
        self._load_parallel_linear_weights(hf_weights, prefix, "gk_proj.0.1")
        self._load_parallel_linear_weights(hf_weights, prefix, "gk_proj.1.0")
        self._load_parallel_linear_weights(hf_weights, prefix, "gk_proj.1.1")

        # Load e projection
        self._load_parallel_linear_weights(hf_weights, prefix, "e_proj")

        # Load SWA QKV projections
        if hasattr(self, 'swa_qkv_proj'):
            q_proj_key = f"{prefix}swa_q_proj.weight"
            k_proj_key = f"{prefix}swa_k_proj.weight"
            v_proj_key = f"{prefix}swa_v_proj.weight"
            
            if q_proj_key in hf_weights and k_proj_key in hf_weights and v_proj_key in hf_weights:
                q_weight = hf_weights[q_proj_key]
                k_weight = hf_weights[k_proj_key]
                v_weight = hf_weights[v_proj_key]
                
                qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                if hasattr(self.swa_qkv_proj, 'weight'):
                    self.swa_qkv_proj.weight.data.copy_(qkv_weight)
                
                # Handle biases if present
                if self.qkv_bias:
                    q_bias_key = f"{prefix}swa_q_proj.bias"
                    k_bias_key = f"{prefix}swa_k_proj.bias"
                    v_bias_key = f"{prefix}swa_v_proj.bias"
                    
                    if q_bias_key in hf_weights and k_bias_key in hf_weights and v_bias_key in hf_weights:
                        q_bias = hf_weights[q_bias_key]
                        k_bias = hf_weights[k_bias_key]
                        v_bias = hf_weights[v_bias_key]
                        
                        qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                        if hasattr(self.swa_qkv_proj, 'bias') and self.swa_qkv_proj.bias is not None:
                            self.swa_qkv_proj.bias.data.copy_(qkv_bias)

        # Load output projections
        self._load_parallel_linear_weights(hf_weights, prefix, "sse_o_proj")
        self._load_parallel_linear_weights(hf_weights, prefix, "swa_o_proj")

        # Load normalization layers
        self._load_norm_weights(hf_weights, prefix, "sse_o_norm")
        self._load_norm_weights(hf_weights, prefix, "sse_merge_norm")
        self._load_norm_weights(hf_weights, prefix, "swa_merge_norm")

        if self.swa_qk_norm:
            self._load_norm_weights(hf_weights, prefix, "swa_q_norm")
            self._load_norm_weights(hf_weights, prefix, "swa_k_norm")

        logger.info(f"Loaded SSE-SWA-GLA layer weights for {prefix}")

    def _load_parallel_linear_weights(self, hf_weights: dict, prefix: str, layer_name: str):
        """Helper to load parallel linear layer weights"""
        weight_key = f"{prefix}{layer_name}.weight"
        bias_key = f"{prefix}{layer_name}.bias"
        
        layer = getattr(self, layer_name.replace('.', '_'), None)
        if layer is None:
            return
            
        if weight_key in hf_weights and hasattr(layer, 'weight'):
            layer.weight.data.copy_(hf_weights[weight_key])
        
        if self.qkv_bias and bias_key in hf_weights and hasattr(layer, 'bias') and layer.bias is not None:
            layer.bias.data.copy_(hf_weights[bias_key])

    def _load_norm_weights(self, hf_weights: dict, prefix: str, layer_name: str):
        """Helper to load normalization layer weights"""
        weight_key = f"{prefix}{layer_name}.weight"
        
        layer = getattr(self, layer_name, None)
        if layer is None:
            return
            
        if weight_key in hf_weights and hasattr(layer, 'weight'):
            layer.weight.data.copy_(hf_weights[weight_key])

    def sse_linear_attention_varlen(self, q1, q2, k1, k2, v, gk1, gk2, eta, recurrent_state=None, use_cache=False, cu_seqlens=None):
        """
        SSE linear attention with varlen implementation.
        Directly copied from hf implementation without modification.
        """
        assert self.num_writer == self.num_reader, "varlen only support num_writer == num_reader"
        
        bsz, q_len, nhead, _ = q1.shape
        
        # change to inference mode.
        mode = 'fused_recurrent' if q_len <= 64 else self.mode
        # if self.training:
        #     assert mode == 'chunk', "Only chunk mode is supported in training."
        
        v1 = v
        v2 = v
        
        if cu_seqlens is None:
            cu_seqlens = torch.arange(0, (bsz + 1) * q_len, q_len, dtype=torch.int32, device=q1.device)
        
        q1, k1, gk1, v1 = [rearrange(src, 'b l h d -> 1 (b l) h d').contiguous() for src in [q1, k1, gk1, v]]
        q2, k2, gk2, v2 = [rearrange(src, 'b l h d -> 1 (b l) h d').contiguous() for src in [q2, k2, gk2, v]]
        eta = rearrange(eta, 'b l p -> 1 (b l) p').contiguous()
        
        S = len(cu_seqlens) - 1
        
        if use_cache:
            recurrent_state1 = recurrent_state[:S] if recurrent_state is not None else \
                torch.zeros(S, self.num_heads, self.head_dim, self.head_dim).to(torch.float32).to(v.device)
            recurrent_state2 = recurrent_state[S:] if recurrent_state is not None else \
                torch.zeros(S*self.num_sparse_partition, self.num_heads, self.head_dim, self.head_dim).to(torch.float32).to(v.device)
        
        q2, k2, v2, gk2, _, eta, mask, offsets, state_sizes, global_sorted = sort_along_l(
            q2, k2, v2, gk2, None, eta, cu_seqlens, self.num_writer, self.emulq, self.emulk
        )
        
        aux_loss = torch.zeros(()).to(eta)
        if self.training:
            p = torch.mean(eta.float(), dim=(0, 1))
            f = torch.mean(mask.float(), dim=(0, 1))
            aux_loss = torch.sum(p * f) * self.num_sparse_partition / self.num_writer
        
        q, k, gk, v = [torch.cat(pair, dim=1) for pair in zip((q1, k1, gk1, v1), (q2, k2, gk2, v2))]
        offsets = torch.cat([cu_seqlens.to(offsets), offsets[1:] + cu_seqlens[-1]])
        
        recurrent_state_rec = None
        if use_cache:
            state_id = torch.nonzero(state_sizes.flatten(), as_tuple=True)[0].cpu()
            recurrent_state_rec = torch.cat((recurrent_state1, recurrent_state2[state_id]), dim=0)
        
        if mode == 'fused_recurrent':
            o, recurrent_state_rec = fused_recurrent_gla(
                q=q,
                k=k,
                v=v,
                g=gk,
                initial_state=recurrent_state_rec,
                output_final_state=use_cache,
                cu_seqlens=offsets,
            )
        elif mode == 'chunk':
            o, recurrent_state_rec = chunk_gla(
                q=q,
                k=k,
                v=v,
                g=gk,
                initial_state=recurrent_state_rec,
                output_final_state=use_cache,
                cu_seqlens=offsets,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")
        
        if use_cache and recurrent_state_rec is not None:
            recurrent_state1 = recurrent_state_rec[:S]
            recurrent_state2[state_id] = recurrent_state_rec[S:]
            recurrent_state = torch.cat((recurrent_state1, recurrent_state2), dim=0)
        else:
            recurrent_state = None
        
        o1, o2 = o[:, :cu_seqlens[-1]], o[:, cu_seqlens[-1]:]
        o2_reduce = torch.zeros_like(o1)
        o2_reduce.index_add_(dim=1, index=global_sorted, source=o2)
        o = o1 + o2_reduce
        
        if bsz > 1:
            o = rearrange(o, "1 (b l) h d -> b l h d", b=bsz).contiguous()
        
        return o, recurrent_state, aux_loss

    def sse_linear_attention_mask(self, q1, q2, k1, k2, v, gk1, gk2, eta, recurrent_state=None, use_cache=False, cu_seqlens=None):
        """
        SSE linear attention with mask implementation.
        Directly copied from hf implementation without modification.
        """
        bsz, q_len, nhead, _ = q1.shape
        
        # change to inference mode.
        mode = 'fused_recurrent' if q_len <= 64 else self.mode
        # if self.training:
        #     assert mode == 'chunk', "Only chunk mode is supported in training."
        
        if use_cache:
            recurrent_state = recurrent_state if recurrent_state is not None else \
                torch.zeros(bsz, self.num_heads, self.head_dim, self.head_dim).to(torch.float32).to(v.device)
        
        global_indices, mask = prepare_sample_relpos_global_index_flat(
            q1, k1, eta, self.num_writer, self.num_reader,
            self.num_sparse_partition, self.emulq, self.emulk
        )
        
        aux_loss = torch.zeros(()).to(eta)
        if self.training:
            p = torch.mean(eta.float(), dim=(0, 1))
            f = torch.mean(mask.float(), dim=(0, 1))
            aux_loss = torch.sum(p * f) * self.num_sparse_partition / self.num_writer
        
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gla(
                q=q1,
                k=k1,
                v=v,
                g=gk1,
                initial_state=recurrent_state,
                output_final_state=use_cache,
            )
        elif mode == 'chunk':
            o, recurrent_state = chunk_gla(
                q=q1,
                k=k1,
                v=v,
                g=gk1,
                initial_state=recurrent_state,
                output_final_state=use_cache,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")
        
        if use_cache:
            for i in range(bsz):
                for j in range(q_len):
                    if mask[i, j].any():
                        idx = global_indices[i, j][mask[i, j]]
                        recurrent_state[i] = recurrent_state[i] * (1 - mask[i, j][:, None, None]) + \
                                           (k1[i, j] @ v[i, j].transpose(-1, -2)) * mask[i, j][:, None, None]
        
        return o, recurrent_state, aux_loss

    def swa_softmax_attention(self, q, k, v, attention_mask, cu_seqlens, max_seqlen, batch_size, q_len):
        """
        Sliding window attention implementation using flash attention.
        """
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
        
        return o

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        """
        Forward pass of the SSE-SWA-GLA layer.
        """
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.size()
        hidden_states_ori = hidden_states

        # Handle padding
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        # ==== SSE Linear Attention Starts Here ====
        # SSE projections using vllm's ColumnParallelLinear
        q1, _ = self.sse_q_proj(hidden_states)
        k1, _ = self.sse_k_proj(hidden_states)
        v, _ = self.sse_v_proj(hidden_states)

        # LoRA projections (manually execute two steps)
        lora_q_output, _ = self.lora_q_proj_0(hidden_states)
        lora_q_output, _ = self.lora_q_proj_1(lora_q_output)
        q2 = q1 + lora_q_output

        lora_k_output, _ = self.lora_k_proj_0(hidden_states)
        lora_k_output, _ = self.lora_k_proj_1(lora_k_output)
        k2 = k1 + lora_k_output

        # Gate projections (manually execute two steps for each gate)
        gk1, _ = self.gk_proj_0(hidden_states)
        gk1, _ = self.gk_proj_1(gk1)

        gk2, _ = self.gk_proj_2(hidden_states)
        gk2, _ = self.gk_proj_3(gk2)

        eta, _ = self.e_proj(hidden_states)

        # Handle short convolutions if enabled
        conv_state_q, conv_state_k = None, None
        if self.use_short_conv:
            conv_state = past_key_values.get(f"conv_state_{self.layer_idx}") if past_key_values else None
            conv_state_q, conv_state_k = (conv_state[0], conv_state[1]) if conv_state is not None else (None, None)
            
            q1, conv_state_q = self.q_conv1d_shared(
                q1,
                conv_state_q,
                process_func=lambda x: rearrange(x, '... (h d) -> ... h d', d=self.sse_head_k_dim, h=self.tp_sse_kv_heads),
                unprocess_func=lambda x: rearrange(x, '... h d -> ... (h d)'),
            )
            
            k1, conv_state_k = self.k_conv1d_shared(
                k1,
                conv_state_k,
                process_func=lambda x: rearrange(x, '... (h d) -> ... h d', d=self.sse_head_k_dim, h=self.tp_sse_kv_heads),
                unprocess_func=lambda x: rearrange(x, '... h d -> ... (h d)'),
            )

        # Reshape for multi-head attention
        q1 = rearrange(q1, '... (h d) -> ... h d', d=self.sse_head_k_dim, h=self.tp_sse_kv_heads)
        q2 = rearrange(q2, '... (h d) -> ... h d', d=self.sse_head_k_dim, h=self.tp_sse_kv_heads)
        k1 = rearrange(k1, '... (h d) -> ... h d', d=self.sse_head_k_dim, h=self.tp_sse_kv_heads)
        k2 = rearrange(k2, '... (h d) -> ... h d', d=self.sse_head_k_dim, h=self.tp_sse_kv_heads)
        gk1 = rearrange(gk1, '... (h d) -> ... h d', d=self.sse_head_k_dim, h=self.tp_sse_kv_heads)
        gk2 = rearrange(gk2, '... (h d) -> ... h d', d=self.sse_head_k_dim, h=self.tp_sse_kv_heads)
        v = rearrange(v, '... (h d) -> ... h d', d=self.sse_head_v_dim, h=self.tp_sse_kv_heads)

        # Apply activations
        if self.use_q_softmax:
            q1 = F.softmax(q1.float(), dim=-1).to(v)
            q2 = F.softmax(q2.float(), dim=-1).to(v)
        else:
            q1 = F.relu(q1) if self.sse_qk_relu else F.silu(q1)
            q2 = F.relu(q2) if self.sse_qk_relu else F.silu(q2)

        if self.use_k_softmax:
            k1 = F.softmax(k1.float(), dim=-1).to(v)
            k2 = F.softmax(k2.float(), dim=-1).to(v)
        else:
            k1 = F.relu(k1) if self.sse_qk_relu else F.silu(k1)
            k2 = F.relu(k2) if self.sse_qk_relu else F.silu(k2)

        v = F.silu(v)

        # Apply gate logit normalization
        gk1 = F.logsigmoid(gk1) / self.gate_logit_normalizer
        gk2 = F.logsigmoid(gk2) / self.gate_logit_normalizer

        # Get recurrent state from past key values
        recurrent_state = None
        if use_cache and past_key_values is not None:
            recurrent_state = past_key_values.get(f"recurrent_state_{self.layer_idx}")

        # SSE linear attention
        sse_o, recurrent_state, aux_loss = self.sse_implementation(
            q1,
            q2,
            k1,
            k2,
            v,
            gk1,
            gk2,
            eta,
            recurrent_state=recurrent_state,
            use_cache=use_cache,
            cu_seqlens=cu_seqlens if attention_mask is not None else None,
        )

        # Update past key values
        if use_cache and past_key_values is not None:
            if recurrent_state is not None:
                past_key_values[f"recurrent_state_{self.layer_idx}"] = recurrent_state
            if self.use_short_conv:
                past_key_values[f"conv_state_{self.layer_idx}"] = (conv_state_q, conv_state_k)

        # Apply output gate and normalization
        if self.use_output_gate:
            # Output gate projection (manually execute two steps)
            g, _ = self.sse_g_proj_0(hidden_states_ori)
            g, _ = self.sse_g_proj_1(g)
            g = rearrange(g, '... (h d) -> ... h d', d=self.sse_head_v_dim, h=self.tp_sse_kv_heads)
            sse_o = self.sse_o_norm(sse_o, g)
        else:
            sse_o = self.sse_o_norm(sse_o)

        # Project output using vllm's RowParallelLinear
        sse_o = rearrange(sse_o, 'b t h d -> b t (h d)')
        sse_o, _ = self.sse_o_proj(sse_o)

        # Repad if needed
        if attention_mask is not None:
            sse_o = pad_input(sse_o.squeeze(0), indices, batch_size, q_len)

        # ==== SWA Softmax Attention Starts Here ====
        # Stochastic switching between SSE and SWA during training
        if self.training and torch.rand(()) > 1 - self.swa_dropout:
            o = self.sse_merge_norm(sse_o)
            return o, (None, aux_loss), past_key_values

        # SWA projections using vllm's QKVParallelLinear
        qkv, _ = self.swa_qkv_proj(hidden_states_ori)
        q, k, v = qkv.chunk(chunks=3, dim=-1)

        # Reshape for multi-head attention
        q = rearrange(q, '... (h d) -> ... h d', d=self.head_dim, h=self.tp_heads)
        k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim, h=self.tp_swa_kv_heads)
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim, h=self.tp_swa_kv_heads)

        # Apply QK normalization if enabled
        if self.swa_qk_norm:
            q = self.swa_q_norm(q)
            k = self.swa_k_norm(k)

        # Handle rotary embedding
        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                # to deliminate the offsets of padding tokens
                seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)

        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=kwargs.get('cu_seqlens'))

        # Handle KV cache update
        if use_cache and past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            
            # Flatten for cache storage
            k_flat = rearrange(k, 'b s h d -> b s (h d)')
            v_flat = rearrange(v, 'b s h d -> b s (h d)')
            
            k_cached, v_cached = past_key_values.update(
                attn_state=(k_flat, v_flat),
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size),
            )['attn_state']
            
            if cache_has_content:
                k = rearrange(k_cached, '... (h d) -> ... h d', d=self.head_dim)
                v = rearrange(v_cached, '... (h d) -> ... h d', d=self.head_dim)

        # Perform SWA computation
        swa_o = self.swa_softmax_attention(
            q,
            k,
            v,
            attention_mask,
            cu_seqlens if attention_mask is not None else None,
            max_seqlen,
            batch_size,
            q_len
        )

        # Project SWA output using vllm's RowParallelLinear
        swa_o = rearrange(swa_o, 'b t h d -> b t (h d)')
        swa_o, _ = self.swa_o_proj(swa_o)

        # Merge SSE and SWA outputs
        o = (self.sse_merge_norm(sse_o) + self.swa_merge_norm(swa_o)) / 2

        return o, (None, aux_loss), past_key_values


class VLLMSSESWAGDN(VLLMSSESWAGLA):
    """
    The layer implementation for [SSE: Scaling Linear Attention with Sparse State Expansion](https://arxiv.org/pdf/2507.16577).
    This is a vllm-compatible implementation with GDN (Gated Delta Rule).
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1,
        head_dim: int = 256,
        num_heads: int = 6,
        mode: str = 'chunk',
        use_output_gate: bool = True,
        use_short_conv: bool = False,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        num_sparse_partition: int = 4,
        num_writer: int = 1,
        num_reader: int = 1,
        sse_implementation: str = "varlen",
        sse_qk_relu: bool = False,
        use_q_softmax: bool = False,
        use_k_softmax: bool = False,
        emulq: bool = True,
        emulk: bool = True,
        qkv_bias: bool = False,
        # === swa configs ===
        swa_num_kv_heads: Optional[int] = None,
        swa_qk_norm: bool = False,
        swa_dropout: float = 0.5,
        window_size: Optional[int] = None,
        rope_theta: Optional[float] = 10000.,
        max_position_embeddings: Optional[int] = None,
        # ===================
        layer_idx: Optional[int] = None,
        norm_eps: float = 1e-5,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            expand_v=expand_v,
            head_dim=head_dim,
            num_heads=num_heads,
            mode=mode,
            use_output_gate=use_output_gate,
            use_short_conv=use_short_conv,
            conv_size=conv_size,
            conv_bias=conv_bias,
            num_sparse_partition=num_sparse_partition,
            num_writer=num_writer,
            num_reader=num_reader,
            sse_implementation=sse_implementation,
            sse_qk_relu=sse_qk_relu,
            use_q_softmax=use_q_softmax,
            use_k_softmax=use_k_softmax,
            emulq=emulq,
            emulk=emulk,
            gate_logit_normalizer=16,  # Not used in GDN
            gate_low_rank_dim=16,      # Not used in GDN
            qkv_bias=qkv_bias,
            swa_num_kv_heads=swa_num_kv_heads,
            swa_qk_norm=swa_qk_norm,
            swa_dropout=swa_dropout,
            window_size=window_size,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
            layer_idx=layer_idx,
            norm_eps=norm_eps,
            quant_config=quant_config,
            prefix=prefix,
            **kwargs,
        )
        
        self.allow_neg_eigval = allow_neg_eigval

        # Remove GLA-specific gate projections
        if hasattr(self, 'gk_proj_0'):
            del self.gk_proj_0
            del self.gk_proj_1
            del self.gk_proj_2
            del self.gk_proj_3

        # Add GDN-specific projections using vllm's ColumnParallelLinear
        self.sse_a_proj = ColumnParallelLinear(
            hidden_size, self.sse_num_kv_heads * 2, bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.sse_a_proj"
        )
        self.sse_b_proj = ColumnParallelLinear(
            hidden_size, self.sse_num_kv_heads * 2, bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.sse_b_proj"
        )

        # GDN specific parameters
        A = torch.empty(self.sse_num_kv_heads * 2, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.sse_num_kv_heads * 2) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min),
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

    def sse_linear_attention_varlen(self, q1, q2, k1, k2, v, g1, g2, b1, b2, eta, recurrent_state=None, use_cache=False, cu_seqlens=None):
        """
        SSE linear attention with GDN and varlen implementation.
        Directly copied from hf implementation without modification.
        """
        assert self.num_writer == self.num_reader, "varlen only support num_writer == num_reader"
        
        bsz, q_len, nhead, _ = q1.shape
        
        # change to inference mode.
        mode = 'fused_recurrent' if (q_len // self.num_sparse_partition <= 64) else self.mode
        # if self.training:
        #     assert mode == 'chunk', "Only chunk mode is supported in training."
        
        v1 = v
        v2 = v
        
        if cu_seqlens is None:
            cu_seqlens = torch.arange(0, (bsz + 1) * q_len, q_len, dtype=torch.int32, device=q1.device)
        
        q1, k1, v1 = [rearrange(src, 'b l h d -> 1 (b l) h d').contiguous() for src in [q1, k1, v]]
        q2, k2, v2 = [rearrange(src, 'b l h d -> 1 (b l) h d').contiguous() for src in [q2, k2, v]]
        g1, g2, b1, b2 = [rearrange(src, 'b l h -> 1 (b l) h').contiguous() for src in [g1, g2, b1, b2]]
        eta = rearrange(eta, 'b l p -> 1 (b l) p').contiguous()
        
        S = len(cu_seqlens) - 1
        
        if use_cache:
            recurrent_state1 = recurrent_state[:S] if recurrent_state is not None else \
                torch.zeros(S, self.num_heads, self.head_dim, self.head_dim).to(torch.float32).to(v.device)
            recurrent_state2 = recurrent_state[S:] if recurrent_state is not None else \
                torch.zeros(S*self.num_sparse_partition, self.num_heads, self.head_dim, self.head_dim).to(torch.float32).to(v.device)
        
        q2, k2, v2, g2, b2, eta, mask, offsets, state_sizes, global_sorted = sort_along_l(
            q2, k2, v2, g2, b2, eta, cu_seqlens, self.num_writer, self.emulq, self.emulk
        )
        
        aux_loss = torch.zeros(()).to(eta)
        if self.training:
            p = torch.mean(eta.float(), dim=(0, 1))
            f = torch.mean(mask.float(), dim=(0, 1))
            aux_loss = torch.sum(p * f) * self.num_sparse_partition / self.num_writer
        
        q, k, g, b, v = [torch.cat(pair, dim=1) for pair in zip((q1, k1, g1, b1, v1), (q2, k2, g2, b2, v2))]
        offsets = torch.cat([cu_seqlens.to(offsets), offsets[1:] + cu_seqlens[-1]])
        
        recurrent_state_rec = None
        if use_cache:
            state_id = torch.nonzero(state_sizes.flatten(), as_tuple=True)[0].cpu()
            recurrent_state_rec = torch.cat((recurrent_state1, recurrent_state2[state_id]), dim=0)
        
        if mode == 'fused_recurrent':
            o, recurrent_state_rec = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=b,
                initial_state=recurrent_state_rec,
                output_final_state=use_cache,
                cu_seqlens=offsets,
                use_qk_l2norm_in_kernel=True,
            )
        elif mode == 'chunk':
            o, recurrent_state_rec = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=b,
                initial_state=recurrent_state_rec,
                output_final_state=use_cache,
                cu_seqlens=offsets,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")
        
        if use_cache and recurrent_state_rec is not None:
            recurrent_state1 = recurrent_state_rec[:S]
            recurrent_state2[state_id] = recurrent_state_rec[S:]
            recurrent_state = torch.cat((recurrent_state1, recurrent_state2), dim=0)
        else:
            recurrent_state = None
        
        o1, o2 = o[:, :cu_seqlens[-1]], o[:, cu_seqlens[-1]:]
        o2_reduce = torch.zeros_like(o1)
        o2_reduce.index_add_(dim=1, index=global_sorted, source=o2)
        o = o1 + o2_reduce
        
        if bsz > 1:
            o = rearrange(o, "1 (b l) h d -> b l h d", b=bsz).contiguous()
        
        return o, recurrent_state, aux_loss

    def sse_linear_attention_mask(self, q1, q2, k1, k2, v, g1, g2, b1, b2, eta, recurrent_state=None, use_cache=False, cu_seqlens=None):
        """
        SSE linear attention with GDN and mask implementation.
        Directly copied from hf implementation without modification.
        """
        bsz, q_len, nhead, _ = q1.shape
        
        # change to inference mode.
        mode = 'fused_recurrent' if q_len // self.num_sparse_partition <= 64 else self.mode
        # if self.training:
        #     assert mode == 'chunk', "Only chunk mode is supported in training."
        
        if use_cache:
            recurrent_state = recurrent_state if recurrent_state is not None else \
                torch.zeros(bsz, self.num_heads, self.head_dim, self.head_dim).to(torch.float32).to(v.device)
        
        g2, b2 = [repeat(x, "b l h -> b l n h", n=self.num_sparse_partition) for x in (g2, b2)]
        
        global_indices_r, mask_r = prepare_sample_relpos_global_index_flat(
            q1, k1, eta, self.num_reader, self.num_reader,
            self.num_sparse_partition, self.emulq, self.emulk
        )
        global_indices_w, mask_w = prepare_sample_relpos_global_index_flat(
            q1, k1, eta, self.num_writer, self.num_writer,
            self.num_sparse_partition, self.emulq, self.emulk
        )
        
        mask_r = mask_r[..., None]
        g2, b2 = g2 * mask_r, b2 * mask_r
        
        g2, b2 = [rearrange(x, "b l n h -> b l (n h)") for x in (g2, b2)]
        
        # writer-only auxloss
        aux_loss = torch.zeros(()).to(eta)
        if self.training:
            p = torch.mean(eta.float(), dim=(0, 1))
            f = torch.mean(mask_w.float(), dim=(0, 1))
            aux_loss = torch.sum(p * f) * self.num_sparse_partition / self.num_writer
        
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gated_delta_rule(
                q=q1,
                k=k1,
                v=v,
                g=g1,
                beta=b1,
                initial_state=recurrent_state,
                output_final_state=use_cache,
            )
        elif mode == 'chunk':
            o, recurrent_state = chunk_gated_delta_rule(
                q=q1,
                k=k1,
                v=v,
                g=g1,
                beta=b1,
                initial_state=recurrent_state,
                output_final_state=use_cache,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")
        
        if use_cache:
            for i in range(bsz):
                for j in range(q_len):
                    if mask_w[i, j].any():
                        idx = global_indices_w[i, j][mask_w[i, j]]
                        recurrent_state[i] = recurrent_state[i] * (1 - mask_w[i, j][:, None, None]) + \
                                           (k1[i, j] @ v[i, j].transpose(-1, -2)) * mask_w[i, j][:, None, None]
        
        return o, recurrent_state, aux_loss

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        """
        Forward pass of the SSE-SWA-GDN layer.
        """
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.size()
        hidden_states_ori = hidden_states

        # Handle padding
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        # ==== SSE Linear Attention Starts Here ====
        # SSE projections using vllm's ColumnParallelLinear
        q1, _ = self.sse_q_proj(hidden_states)
        k1, _ = self.sse_k_proj(hidden_states)
        v, _ = self.sse_v_proj(hidden_states)

        # LoRA projections (manually execute two steps)
        lora_q_output, _ = self.lora_q_proj_0(hidden_states)
        lora_q_output, _ = self.lora_q_proj_1(lora_q_output)
        q2 = q1 + lora_q_output

        lora_k_output, _ = self.lora_k_proj_0(hidden_states)
        lora_k_output, _ = self.lora_k_proj_1(lora_k_output)
        k2 = k1 + lora_k_output

        # GDN specific projections using vllm's ColumnParallelLinear
        b, _ = self.sse_b_proj(hidden_states)
        b = b.sigmoid()
        if self.allow_neg_eigval:
            b = b * 2.

        g, _ = self.sse_a_proj(hidden_states)
        g = -self.A_log.float().exp() * F.softplus(g.float() + self.dt_bias)

        b1, b2 = torch.chunk(b, 2, dim=-1)
        g1, g2 = torch.chunk(g, 2, dim=-1)

        eta, _ = self.e_proj(hidden_states)

        # Handle short convolutions if enabled
        conv_state_q, conv_state_k = None, None
        if self.use_short_conv:
            conv_state = past_key_values.get(f"conv_state_{self.layer_idx}") if past_key_values else None
            conv_state_q, conv_state_k = (conv_state[0], conv_state[1]) if conv_state is not None else (None, None)
            
            q1, conv_state_q = self.q_conv1d_shared(
                q1,
                conv_state_q,
                process_func=lambda x: rearrange(x, '... (h d) -> ... h d', d=self.sse_head_k_dim, h=self.tp_sse_kv_heads),
                unprocess_func=lambda x: rearrange(x, '... h d -> ... (h d)'),
            )
            
            k1, conv_state_k = self.k_conv1d_shared(
                k1,
                conv_state_k,
                process_func=lambda x: rearrange(x, '... (h d) -> ... h d', d=self.sse_head_k_dim, h=self.tp_sse_kv_heads),
                unprocess_func=lambda x: rearrange(x, '... h d -> ... (h d)'),
            )

        # Reshape for multi-head attention
        q1 = rearrange(q1, '... (h d) -> ... h d', d=self.sse_head_k_dim, h=self.tp_sse_kv_heads)
        q2 = rearrange(q2, '... (h d) -> ... h d', d=self.sse_head_k_dim, h=self.tp_sse_kv_heads)
        k1 = rearrange(k1, '... (h d) -> ... h d', d=self.sse_head_k_dim, h=self.tp_sse_kv_heads)
        k2 = rearrange(k2, '... (h d) -> ... h d', d=self.sse_head_k_dim, h=self.tp_sse_kv_heads)
        g1 = rearrange(g1, '... (h) -> ... h', h=self.tp_sse_kv_heads)
        g2 = rearrange(g2, '... (h) -> ... h', h=self.tp_sse_kv_heads)
        b1 = rearrange(b1, '... (h) -> ... h', h=self.tp_sse_kv_heads)
        b2 = rearrange(b2, '... (h) -> ... h', h=self.tp_sse_kv_heads)
        v = rearrange(v, '... (h d) -> ... h d', d=self.sse_head_v_dim, h=self.tp_sse_kv_heads)

        # Apply activations
        if self.use_q_softmax:
            q1 = F.softmax(q1.float(), dim=-1).to(v)
            q2 = F.softmax(q2.float(), dim=-1).to(v)
        else:
            q1 = F.relu(q1) if self.sse_qk_relu else F.silu(q1)
            q2 = F.relu(q2) if self.sse_qk_relu else F.silu(q2)

        if self.use_k_softmax:
            k1 = F.softmax(k1.float(), dim=-1).to(v)
            k2 = F.softmax(k2.float(), dim=-1).to(v)
        else:
            k1 = F.relu(k1) if self.sse_qk_relu else F.silu(k1)
            k2 = F.relu(k2) if self.sse_qk_relu else F.silu(k2)

        v = F.silu(v)

        # Get recurrent state from past key values
        recurrent_state = None
        if use_cache and past_key_values is not None:
            recurrent_state = past_key_values.get(f"recurrent_state_{self.layer_idx}")

        # SSE-GDN linear attention
        sse_o, recurrent_state, aux_loss = self.sse_implementation(
            q1,
            q2,
            k1,
            k2,
            v,
            g1,
            g2,
            b1,
            b2,
            eta,
            recurrent_state=recurrent_state,
            use_cache=use_cache,
            cu_seqlens=cu_seqlens if attention_mask is not None else None,
        )

        # Update past key values
        if use_cache and past_key_values is not None:
            if recurrent_state is not None:
                past_key_values[f"recurrent_state_{self.layer_idx}"] = recurrent_state
            if self.use_short_conv:
                past_key_values[f"conv_state_{self.layer_idx}"] = (conv_state_q, conv_state_k)

        # Apply output gate and normalization
        if self.use_output_gate:
            # Output gate projection (manually execute two steps)
            g, _ = self.sse_g_proj_0(hidden_states_ori)
            g, _ = self.sse_g_proj_1(g)
            g = rearrange(g, '... (h d) -> ... h d', d=self.sse_head_v_dim, h=self.tp_sse_kv_heads)
            sse_o = self.sse_o_norm(sse_o, g)
        else:
            sse_o = self.sse_o_norm(sse_o)

        # Project output using vllm's RowParallelLinear
        sse_o = rearrange(sse_o, 'b t h d -> b t (h d)')
        sse_o, _ = self.sse_o_proj(sse_o)

        # Repad if needed
        if attention_mask is not None:
            sse_o = pad_input(sse_o.squeeze(0), indices, batch_size, q_len)

        # ==== SWA Softmax Attention Starts Here ====
        # Stochastic switching between SSE and SWA during training
        if self.training and torch.rand(()) > 1 - self.swa_dropout:
            o = self.sse_merge_norm(sse_o)
            return o, (None, aux_loss), past_key_values

        # SWA projections using vllm's QKVParallelLinear
        qkv, _ = self.swa_qkv_proj(hidden_states_ori)
        q, k, v = qkv.chunk(chunks=3, dim=-1)

        # Reshape for multi-head attention
        q = rearrange(q, '... (h d) -> ... h d', d=self.head_dim, h=self.tp_heads)
        k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim, h=self.tp_swa_kv_heads)
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim, h=self.tp_swa_kv_heads)

        # Apply QK normalization if enabled
        if self.swa_qk_norm:
            q = self.swa_q_norm(q)
            k = self.swa_k_norm(k)

        # Handle rotary embedding
        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                # to deliminate the offsets of padding tokens
                seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)

        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=kwargs.get('cu_seqlens'))

        # Handle KV cache update
        if use_cache and past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            
            # Flatten for cache storage
            k_flat = rearrange(k, 'b s h d -> b s (h d)')
            v_flat = rearrange(v, 'b s h d -> b s (h d)')
            
            k_cached, v_cached = past_key_values.update(
                attn_state=(k_flat, v_flat),
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size),
            )['attn_state']
            
            if cache_has_content:
                k = rearrange(k_cached, '... (h d) -> ... h d', d=self.head_dim)
                v = rearrange(v_cached, '... (h d) -> ... h d', d=self.head_dim)

        # Perform SWA computation
        swa_o = self.swa_softmax_attention(
            q,
            k,
            v,
            attention_mask,
            cu_seqlens if attention_mask is not None else None,
            max_seqlen,
            batch_size,
            q_len
        )

        # Project SWA output using vllm's RowParallelLinear
        swa_o = rearrange(swa_o, 'b t h d -> b t (h d)')
        swa_o, _ = self.swa_o_proj(swa_o)

        # Merge SSE and SWA outputs
        o = (self.sse_merge_norm(sse_o) + self.swa_merge_norm(swa_o)) / 2

        return o, (None, aux_loss), past_key_values


# Register the models with vLLM
__all__ = [
    "VLLMSSESWAGLA",
    "VLLMSSESWAGDN",
]