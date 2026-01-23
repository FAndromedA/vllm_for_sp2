# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Converted to vllm implementation with fixed-sized storage management

from __future__ import annotations
import math
import warnings
from typing import TYPE_CHECKING, Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from transformers.utils import logging
logger = logging.get_logger(__name__)

from vllm.config import VllmConfig
from vllm.attention.backends.abstract import AttentionType
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear, RowParallelLinear, QKVParallelLinear,
)
from vllm.model_executor.utils import set_weight_attrs
# from vllm.model_executor.models.registry import register_model
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size
)
from vllm.forward_context import ForwardContext, get_forward_context

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack
    from vllm.model_executor.models.utils import Cache

from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution, RotaryEmbedding
from fla.ops.gla import chunk_gla, fused_recurrent_gla
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from fla.ops.sse import prepare_sample_relpos_global_index_flat, softmax_and_mask
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input


def sort_along_l(q, k, v, gk, beta, e, cu_seqlens, K, emulq, emulk):
    _, L, H, D = q.shape
    N = e.size(-1)
    S = len(cu_seqlens) - 1
    
    e = F.softmax(e, dim=-1, dtype=torch.float)
    topk_value, topk_expert = torch.topk(e, k=K, dim=2)  # [1, L, K]
    topk_value, e = topk_value.to(q.dtype), e.to(q.dtype)
    
    mask_w = torch.zeros_like(e, dtype=torch.bool).scatter_(
        dim=-1, index=topk_expert, src=torch.ones_like(topk_expert, dtype=torch.bool)
    )
    
    experts_flat = topk_expert.reshape(L * K)  # [L*K]
    values_flat  = topk_value.reshape(L * K)   # [L*K]
    
    sample_idx_flat, relpos_flat, global_idx_flat, lengths = prepare_sample_relpos_global_index_flat(
        cu_seqlens, K
    )  # ([L*K] * 3, S)
    
    assert sample_idx_flat.dtype == torch.long and relpos_flat.dtype == torch.long and global_idx_flat.dtype == torch.long
    
    bits_pos = int(lengths.max().item()).bit_length()
    bits_exp = int((N - 1)).bit_length()
    shift_exp  = bits_pos
    shift_samp = bits_pos + bits_exp
    
    ## sort by (sample_idx <- expert_idx <- relpos_in_sample)
    key = (sample_idx_flat << shift_samp) | (experts_flat << shift_exp) | relpos_flat
    order = torch.argsort(key, stable=False)
    
    experts_sorted = experts_flat.take(order)
    sample_sorted  = sample_idx_flat.take(order)
    global_sorted  = global_idx_flat.take(order)   # gather index
    values_sorted  = values_flat.take(order)       # sorted eta
    
    ## x: [1, L, H, D] -> y: [1, L*K, H, D]
    index4gather = global_sorted[None, :, None, None].expand(1, L * K, H, D)
    
    if beta is None:
        q, k, v, gk = [torch.gather(x, dim=1, index=index4gather) for x in (q, k, v, gk)]  # GLA
    else:
        q, k, v = [torch.gather(x, dim=1, index=index4gather) for x in (q, k, v)]          # GDN
        gk, beta = [torch.gather(x, dim=1, index=index4gather[..., 0]) for x in (gk, beta)] 
    
    if emulq:
        q = q * values_sorted[None, :, None, None]
    if emulk:
        k = k * values_sorted[None, :, None, None]
    
    ## calculate offsets (new cu_seqlens)
    pair_id = sample_sorted * N + experts_sorted  # [L*K]
    counts = torch.bincount(pair_id, minlength=S * N)  # [S*N]
    state_sizes = counts.view(S, N)
    
    offsets = torch.zeros(1 + S * N, dtype=torch.long, device=q.device)
    offsets[1:] = counts.cumsum(dim=0)
    offsets = torch.unique(offsets)
    
    return q, k, v, gk, beta, e, mask_w, offsets, state_sizes, global_sorted


class VLLMSSEGLA(nn.Module):
    """
    vLLM-compatible SSE (Sparse State Expansion) layer with GLA (Gated Linear Attention).
    This is a pure linear attention implementation with fixed-sized storage management.
    """
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        hidden_size: int = 2048,
        expand_v: float = 1.,
        head_dim: int = 256,
        num_heads: int = 6,
        num_v_heads: Optional[int] = None,
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
        layer_idx: Optional[int] = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> VLLMSSEGLA:
        super().__init__()

        self.prefix = prefix
        self.layer_idx = layer_idx
        self.vllm_config = vllm_config
        
        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v

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

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads

        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)

        # Tensor parallel configuration
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        
        # Validate and calculate tensor parallel heads
        assert self.num_heads % self.tp_size == 0, \
            f"num_heads ({self.num_heads}) must be divisible by tp_size ({self.tp_size})"
        
        if self.num_v_heads >= self.tp_size:
            assert self.num_v_heads % self.tp_size == 0, \
                f"num_v_heads ({self.num_v_heads}) must be divisible by tp_size ({self.tp_size})"
        else:
            assert self.tp_size % self.num_v_heads == 0, \
                f"tp_size ({self.tp_size}) must be divisible by num_v_heads ({self.num_v_heads})"
        
        self.tp_heads = self.num_heads // self.tp_size
        self.tp_v_heads = max(1, self.num_v_heads // self.tp_size)

        # Consistency check: Ensure expand_v produces integer values
        if not math.isclose(self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}. "
                f"Resulting value_dim would be {self.num_v_heads * self.head_dim * expand_v}, which is invalid for nn.Linear.",
            )
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}.",
            )

        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={head_dim}. "
                f"Resulting head_v_dim would be {head_dim * expand_v}, which is invalid for FusedRMSNormGated.",
            )
        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."

        # Projection layers using vllm's parallel linear layers
        self.q_proj = ColumnParallelLinear(
            hidden_size, self.key_dim, bias=False, prefix=f"{prefix}.q_proj"
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size, self.key_dim, bias=False, prefix=f"{prefix}.k_proj"
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size, self.value_dim, bias=False, prefix=f"{prefix}.v_proj"
        )
        
        # LoRA projections (split from nn.Sequential to handle tuple outputs)
        self.lora_q_proj_0 = ColumnParallelLinear(
            hidden_size, self.head_v_dim, bias=False, prefix=f"{prefix}.lora_q_proj.0"
        )
        self.lora_q_proj_1 = ColumnParallelLinear(
            self.head_v_dim, self.key_dim, bias=False, prefix=f"{prefix}.lora_q_proj.1"
        )
        
        self.lora_k_proj_0 = ColumnParallelLinear(
            hidden_size, self.head_v_dim, bias=False, prefix=f"{prefix}.lora_k_proj.0"
        )
        self.lora_k_proj_1 = ColumnParallelLinear(
            self.head_v_dim, self.key_dim, bias=False, prefix=f"{prefix}.lora_k_proj.1"
        )

        self.gate_logit_normalizer = gate_logit_normalizer
        # Gate projections (split from nn.Sequential to handle tuple outputs)
        self.gk_proj_0_0 = ColumnParallelLinear(
            hidden_size, gate_low_rank_dim, bias=False, prefix=f"{prefix}.gk_proj.0.0"
        )
        self.gk_proj_0_1 = ColumnParallelLinear(
            gate_low_rank_dim, self.key_dim, bias=True, prefix=f"{prefix}.gk_proj.0.1"
        )
        
        self.gk_proj_1_0 = ColumnParallelLinear(
            hidden_size, gate_low_rank_dim, bias=False, prefix=f"{prefix}.gk_proj.1.0"
        )
        self.gk_proj_1_1 = ColumnParallelLinear(
            gate_low_rank_dim, self.key_dim, bias=True, prefix=f"{prefix}.gk_proj.1.1"
        )

        self.e_proj = ColumnParallelLinear(
            hidden_size, self.num_sparse_partition, bias=False, prefix=f"{prefix}.e_proj"
        )

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d_shared = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation=None,
            )
            self.k_conv1d_shared = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation=None,
            )

        if use_output_gate:
            # Output gate projection (split from nn.Sequential to handle tuple outputs)
            self.g_proj_0 = ColumnParallelLinear(
                hidden_size, self.head_v_dim, bias=False, prefix=f"{prefix}.g_proj.0"
            )
            self.g_proj_1 = ColumnParallelLinear(
                self.head_v_dim, self.value_dim, bias=False, prefix=f"{prefix}.g_proj.1"
            )
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        self.o_proj = RowParallelLinear(
            self.value_dim, hidden_size, bias=False, prefix=f"{prefix}.o_proj"
        )

    def _get_fixed_state_storage(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Get fixed-sized recurrent state storage for linear attention.
        This replaces traditional KVCache with a fixed-size state matrix.
        """
        # Calculate the total number of states needed
        # Base states + sparse partition states
        num_base_states = batch_size
        num_sparse_states = batch_size * self.num_sparse_partition
        total_states = num_base_states + num_sparse_states
        
        # Fixed-sized state matrix: [total_states, heads, head_dim, head_dim]
        state_shape = (total_states, self.tp_heads, self.head_dim, self.head_dim)
        
        # Initialize with zeros if not exists
        return torch.zeros(state_shape, dtype=torch.float32, device=device)

    def _update_fixed_storage(
        self,
        new_state: torch.Tensor,
        state_id: torch.Tensor,
        full_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Update the fixed-sized storage with new state values.
        """
        # state_id indicates which positions to update
        full_state[state_id] = new_state
        return full_state

    def sse_linear_attention_varlen(self, q1, q2, k1, k2, v, gk1, gk2, eta, 
                                   fixed_storage: Optional[torch.Tensor] = None, 
                                   use_cache: bool = False, cu_seqlens=None):
        """
        SSE linear attention with varlen implementation using fixed-sized storage.
        """
        assert self.num_writer == self.num_reader, "varlen only support num_writer == num_reader"
        bsz, q_len, nhead, _ = q1.shape
        # Change to inference mode based on sequence length
        mode = 'fused_recurrent' if q_len <= 64 else self.mode

        v1 = v
        v2 = v
        if cu_seqlens is None:
            cu_seqlens = torch.arange(0, (bsz + 1) * q_len, q_len, dtype=torch.int32, device=q1.device)
            q1, k1, gk1, v1 = [rearrange(src, 'b l h d -> 1 (b l) h d').contiguous() for src in [q1, k1, gk1, v]]
            q2, k2, gk2, v2 = [rearrange(src, 'b l h d -> 1 (b l) h d').contiguous() for src in [q2, k2, gk2, v]]
        S = len(cu_seqlens) - 1

        # Get fixed-sized storage
        if use_cache:
            if fixed_storage is None:
                # Initialize fixed storage if not provided
                fixed_storage = self._get_fixed_state_storage(S, q1.device)
            
            # Split storage into base and sparse parts
            recurrent_state1 = fixed_storage[:S]  # Base states
            recurrent_state2 = fixed_storage[S:]  # Sparse states

        # Sparse state expansion and sorting
        q2, k2, v2, gk2, _, eta, mask, offsets, state_sizes, global_sorted = sort_along_l(
            q2, k2, v2, gk2, None, eta, cu_seqlens, self.num_writer, self.emulq, self.emulk
        )

        # Auxiliary loss calculation
        aux_loss = torch.zeros(()).to(eta)
        if self.training:
            p = torch.mean(eta.float(), dim=(0, 1))
            f = torch.mean(mask.float(), dim=(0, 1))
            aux_loss = torch.sum(p * f) * self.num_sparse_partition / self.num_writer

        # Concatenate queries, keys, gates and values
        q, k, gk, v = [torch.cat(pair, dim=1) for pair in zip((q1, k1, gk1, v1), (q2, k2, gk2, v2))]
        offsets = torch.cat([cu_seqlens.to(offsets), offsets[1:] + cu_seqlens[-1]])
        
        # Prepare recurrent state for update
        recurrent_state_rec = None
        if use_cache:
            # Find which sparse states need to be updated
            state_id = torch.nonzero(state_sizes.flatten(), as_tuple=True)[0].cpu()
            # Combine base and selected sparse states
            recurrent_state_rec = torch.cat((recurrent_state1, recurrent_state2[state_id]), dim=0)

        # Linear attention computation
        if mode == 'fused_recurrent':
            o, recurrent_state_rec = fused_recurrent_gla(
                q=q,
                k=k,
                v=v,
                gk=gk,
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

        # Update fixed storage if needed
        if use_cache and recurrent_state_rec is not None:
            # Split updated states back to base and sparse parts
            recurrent_state1 = recurrent_state_rec[:S]
            # Update only the states that were modified
            state_id = torch.nonzero(state_sizes.flatten(), as_tuple=True)[0].cpu()
            recurrent_state2[state_id] = recurrent_state_rec[S:]
            # Combine back into fixed storage
            fixed_storage = torch.cat((recurrent_state1, recurrent_state2), dim=0)

        # Reconstruct output
        o1, o2 = o[:, :cu_seqlens[-1]], o[:, cu_seqlens[-1]:]
        o2_reduce = torch.zeros_like(o1)
        o2_reduce.index_add_(dim=1, index=global_sorted, source=o2)
        o = o1 + o2_reduce
        if bsz > 1:
            o = rearrange(o, "1 (b l) h d -> b l h d", b=bsz).contiguous()

        return o, fixed_storage, aux_loss
    
    def sse_linear_attention_mask(self, q1, q2, k1, k2, v, gk1, gk2, eta, 
                                 fixed_storage: Optional[torch.Tensor] = None, 
                                 use_cache: bool = False, cu_seqlens=None):
        """
        SSE linear attention with mask implementation using fixed-sized storage.
        """
        bsz, q_len, nhead, _ = q1.shape
        # Change to inference mode based on sequence length
        mode = 'fused_recurrent' if q_len <= 64 else self.mode

        # Sparse state expansion with masking
        q2, k2, v2, gk2, eta, mask_w, mask_r = softmax_and_mask(
            q2, k2, v, gk2, eta, self.num_writer, self.num_reader
        )

        # Auxiliary loss calculation
        aux_loss = torch.zeros(()).to(eta)
        if self.training:
            p = torch.mean(eta.float(), dim=(0, 1))
            f = torch.mean(mask_w.float(), dim=(0, 1))
            aux_loss = torch.sum(p * f) * self.num_sparse_partition / self.num_writer
        
        # Concatenate queries, keys, gates and values
        q, k, gk, v = [torch.cat(pair, dim=-2) for pair in zip((q1, k1, gk1, v), (q2, k2, gk2, v2))]

        # Get fixed-sized storage
        if use_cache and fixed_storage is None:
            fixed_storage = self._get_fixed_state_storage(bsz, q.device)

        # Linear attention computation
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gla(
                q=q,
                k=k,
                v=v,
                gk=gk,
                initial_state=fixed_storage if use_cache else None,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        elif mode == 'chunk':
            o, recurrent_state = chunk_gla(
                q=q,
                k=k,
                v=v,
                g=gk,
                initial_state=fixed_storage if use_cache else None,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        # Update fixed storage
        if use_cache:
            fixed_storage = recurrent_state

        # Reshape output
        o = rearrange(o, "b l (n h) d -> b l n h d", n=self.num_sparse_partition+1)
        o = o.sum(2)

        return o, fixed_storage, aux_loss

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Forward pass with fixed-sized storage management for linear attention.
        """
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape

        # Get fixed storage from past_key_values
        fixed_storage = None
        conv_state = None
        if past_key_values is not None and self.layer_idx in past_key_values:
            layer_state = past_key_values[self.layer_idx]
            fixed_storage = layer_state.get('fixed_storage')
            conv_state = layer_state.get('conv_state')

        # Handle padding and unpadding
        cu_seqlens = kwargs.get('cu_seqlens')
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)
        
        # Project inputs
        q1, _ = self.q_proj(hidden_states)
        k1, _ = self.k_proj(hidden_states)
        # LoRA projections (manually execute two steps to handle tuple outputs)
        lora_q_output, _ = self.lora_q_proj_0(hidden_states)  # Step 1
        lora_q_output, _ = self.lora_q_proj_1(lora_q_output)  # Step 2
        
        lora_k_output, _ = self.lora_k_proj_0(hidden_states)  # Step 1
        lora_k_output, _ = self.lora_k_proj_1(lora_k_output)  # Step 2
        q2 = q1 + lora_q_output
        k2 = k1 + lora_k_output
        v, _ = self.v_proj(hidden_states)

        # Gate projections (manually execute two steps to handle tuple outputs)
        gk1, _ = self.gk_proj_0_0(hidden_states)  # Step 1
        gk1, _ = self.gk_proj_0_1(gk1)            # Step 2
        
        gk2, _ = self.gk_proj_1_0(hidden_states)  # Step 1
        gk2, _ = self.gk_proj_1_1(gk2)            # Step 2
        
        eta, _ = self.e_proj(hidden_states)

        # Handle short convolutions if enabled
        conv_state_q, conv_state_k = None, None
        if self.use_short_conv:
            conv_state_q, conv_state_k = (conv_state[0], conv_state[1]) if conv_state is not None else (None, None)
            
            q1, conv_state_q = self.q_conv1d_shared(
                x=q1,
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k1, conv_state_k = self.k_conv1d_shared(
                x=k1,
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )

        # Reshape for multi-head attention (tensor parallel aware)
        q1, q2, k1, k2, gk1, gk2 = map(
            lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim, h=self.tp_heads),
            (q1, q2, k1, k2, gk1, gk2)
        )
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim, h=self.tp_v_heads)

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

        # Handle grouped value attention (tensor parallel aware)
        if self.num_v_heads > self.num_heads:
            group_factor = self.num_v_heads // self.num_heads
            q1, q2, k1, k2, gk1, gk2 = map(
                lambda x: repeat(x, '... h d -> ... (h g) d', g=group_factor),
                (q1, q2, k1, k2, gk1, gk2)
            )

        # SSE linear attention with fixed storage
        o, fixed_storage, aux_loss = self.sse_implementation(
            q1,
            q2,
            k1,
            k2,
            v,
            gk1,
            gk2,
            eta,
            fixed_storage=fixed_storage,
            use_cache=use_cache,
            cu_seqlens=cu_seqlens,
        )

        # Update past key values with fixed storage
        if use_cache and past_key_values is not None:
            past_key_values[self.layer_idx] = {
                'fixed_storage': fixed_storage,
                'conv_state': (conv_state_q, conv_state_k) if self.use_short_conv else None
            }

        # Apply output gate and normalization
        if self.use_output_gate:
            # Output gate projection (manually execute two steps to handle tuple outputs)
            g, _ = self.g_proj_0(hidden_states)  # Step 1
            g, _ = self.g_proj_1(g)              # Step 2
            g = rearrange(g, '... (h d) -> ... h d', d=self.head_v_dim, h=self.tp_v_heads)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        
        # Project output
        o = rearrange(o, 'b t h d -> b t (h d)')
        o, _ = self.o_proj(o)
        
        # Repad if needed
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, aux_loss, past_key_values
    
    def load_hf_weights(self, hf_weights: dict, prefix: str = ""):
        """
        Load weights from huggingface implementation.
        """
        # Load projection layers
        proj_layers = ['q_proj', 'k_proj', 'v_proj', 'e_proj', 'o_proj']
        for layer_name in proj_layers:
            weight_key = f"{prefix}{layer_name}.weight"
            if weight_key in hf_weights:
                if hasattr(self, layer_name) and hasattr(getattr(self, layer_name), 'weight'):
                    getattr(self, layer_name).weight.data.copy_(hf_weights[weight_key])
        
        # Load LoRA projections (split into separate layers)
        for lora_base in ['lora_q_proj', 'lora_k_proj']:
            for i in range(2):
                layer_name = f"{lora_base}_{i}"
                if hasattr(self, layer_name):
                    weight_key = f"{prefix}{lora_base}.{i}.weight"
                    if weight_key in hf_weights:
                        getattr(self, layer_name).weight.data.copy_(hf_weights[weight_key])
        
        # Load gate projections (split into separate layers)
        for i in range(2):
            for j in range(2):
                layer_name = f"gk_proj_{i}_{j}"
                if hasattr(self, layer_name):
                    weight_key = f"{prefix}gk_proj.{i}.{j}.weight"
                    if weight_key in hf_weights:
                        getattr(self, layer_name).weight.data.copy_(hf_weights[weight_key])
                    # Load bias if present
                    bias_key = f"{prefix}gk_proj.{i}.{j}.bias"
                    if bias_key in hf_weights and hasattr(getattr(self, layer_name), 'bias') and getattr(self, layer_name).bias is not None:
                        getattr(self, layer_name).bias.data.copy_(hf_weights[bias_key])
        
        # Load output gate projection (split into separate layers)
        if self.use_output_gate:
            for i in range(2):
                layer_name = f"g_proj_{i}"
                if hasattr(self, layer_name):
                    weight_key = f"{prefix}g_proj.{i}.weight"
                    if weight_key in hf_weights:
                        getattr(self, layer_name).weight.data.copy_(hf_weights[weight_key])
        
        # Load normalization layers
        if hasattr(self, 'o_norm'):
            norm_key = f"{prefix}o_norm.weight"
            if norm_key in hf_weights:
                self.o_norm.weight.data.copy_(hf_weights[norm_key])
        
        logger.info(f"Loaded SSE-GLA layer weights for {prefix}")


class VLLMSSEGDN(VLLMSSEGLA):
    """
    vLLM-compatible SSE (Sparse State Expansion) layer with GDN (Gated Delta Rule).
    This is a pure linear attention implementation with fixed-sized storage management.
    """
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        hidden_size: int = 2048,
        expand_v: float = 1,
        head_dim: int = 256,
        num_heads: int = 6,
        num_v_heads: Optional[int] = None,
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
        layer_idx: Optional[int] = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> VLLMSSEGDN:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            hidden_size=hidden_size,
            expand_v=expand_v,
            head_dim=head_dim,
            num_heads=num_heads,
            num_v_heads=num_v_heads,
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
            layer_idx=layer_idx,
            norm_eps=norm_eps,
            **kwargs,
        )
        
        self.allow_neg_eigval = allow_neg_eigval
        
        # Remove GLA-specific gate projections
        if hasattr(self, 'gk_proj'):
            del self.gk_proj
        
        # Add GDN-specific projections
        self.a_proj = ColumnParallelLinear(
            hidden_size, self.num_v_heads * 2, bias=False, prefix=f"{prefix}.a_proj"
        )
        self.b_proj = ColumnParallelLinear(
            hidden_size, self.num_v_heads * 2, bias=False, prefix=f"{prefix}.b_proj"
        )

        # GDN specific parameters
        A = torch.empty(self.num_v_heads * 2, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        
        # Initialize dt_bias
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_v_heads * 2) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min),
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

    def sse_linear_attention_varlen(self, q1, q2, k1, k2, v, g1, g2, b1, b2, eta, 
                                   fixed_storage: Optional[torch.Tensor] = None, 
                                   use_cache: bool = False, cu_seqlens=None):
        """
        SSE linear attention with GDN and varlen implementation using fixed-sized storage.
        """
        assert self.num_writer == self.num_reader, "varlen only support num_writer == num_reader"
        bsz, q_len, nhead, _ = q1.shape
        # Change to inference mode based on sequence length
        mode = 'fused_recurrent' if (q_len // self.num_sparse_partition <= 64) else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        v1 = v
        v2 = v
        if cu_seqlens is None:
            cu_seqlens = torch.arange(0, (bsz + 1) * q_len, q_len, dtype=torch.int32, device=q1.device)
            q1, k1, v1 = [rearrange(src, 'b l h d -> 1 (b l) h d').contiguous() for src in [q1, k1, v]]
            q2, k2, v2 = [rearrange(src, 'b l h d -> 1 (b l) h d').contiguous() for src in [q2, k2, v]]
            g1, g2, b1, b2 = [rearrange(src, 'b l h -> 1 (b l) h').contiguous() for src in [g1, g2, b1, b2]]
        S = len(cu_seqlens) - 1

        # Get fixed-sized storage
        if use_cache:
            if fixed_storage is None:
                # Initialize fixed storage if not provided
                fixed_storage = self._get_fixed_state_storage(S, q1.device)
            
            # Split storage into base and sparse parts
            recurrent_state1 = fixed_storage[:S]  # Base states
            recurrent_state2 = fixed_storage[S:]  # Sparse states

        # Sparse state expansion and sorting
        q2, k2, v2, g2, b2, eta, mask, offsets, state_sizes, global_sorted = sort_along_l(
            q2, k2, v2, g2, b2, eta, cu_seqlens, self.num_writer, self.emulq, self.emulk
        )

        # Auxiliary loss calculation
        aux_loss = torch.zeros(()).to(eta)
        if self.training:
            p = torch.mean(eta.float(), dim=(0, 1))
            f = torch.mean(mask.float(), dim=(0, 1))
            aux_loss = torch.sum(p * f) * self.num_sparse_partition / self.num_writer

        # Concatenate queries, keys, gates, betas and values
        q, k, g, b, v = [torch.cat(pair, dim=1) for pair in zip((q1, k1, g1, b1, v1), (q2, k2, g2, b2, v2))]
        offsets = torch.cat([cu_seqlens.to(offsets), offsets[1:] + cu_seqlens[-1]])
        
        # Prepare recurrent state for update
        recurrent_state_rec = None
        if use_cache:
            # Find which sparse states need to be updated
            state_id = torch.nonzero(state_sizes.flatten(), as_tuple=True)[0].cpu()
            # Combine base and selected sparse states
            recurrent_state_rec = torch.cat((recurrent_state1, recurrent_state2[state_id]), dim=0)

        # GDN linear attention computation
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

        # Update fixed storage if needed
        if use_cache and recurrent_state_rec is not None:
            # Split updated states back to base and sparse parts
            recurrent_state1 = recurrent_state_rec[:S]
            # Update only the states that were modified
            state_id = torch.nonzero(state_sizes.flatten(), as_tuple=True)[0].cpu()
            recurrent_state2[state_id] = recurrent_state_rec[S:]
            # Combine back into fixed storage
            fixed_storage = torch.cat((recurrent_state1, recurrent_state2), dim=0)

        # Reconstruct output
        o1, o2 = o[:, :cu_seqlens[-1]], o[:, cu_seqlens[-1]:]
        o2_reduce = torch.zeros_like(o1)
        o2_reduce.index_add_(dim=1, index=global_sorted, source=o2)
        o = o1 + o2_reduce
        if bsz > 1:
            o = rearrange(o, "1 (b l) h d -> b l h d", b=bsz).contiguous()

        return o, fixed_storage, aux_loss
    
    def sse_linear_attention_mask(self, q1, q2, k1, k2, v, g1, g2, b1, b2, eta, 
                                 fixed_storage: Optional[torch.Tensor] = None, 
                                 use_cache: bool = False, cu_seqlens=None):
        """
        SSE linear attention with GDN and mask implementation using fixed-sized storage.
        """
        bsz, q_len, nhead, _ = q1.shape
        # Change to inference mode based on sequence length
        mode = 'fused_recurrent' if q_len // self.num_sparse_partition <= 64 else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        # Sparse state expansion with masking
        q2, k2, v2, _, eta, mask_w, mask_r = softmax_and_mask(
            q2, k2, v, v, eta, self.num_writer, self.num_reader
        )
        g2, b2 = [repeat(x, "b l h -> b l n h", n=self.num_sparse_partition) for x in (g2, b2)]
        mask_r = mask_r[..., None]
        g2, b2 = g2 * mask_r, b2 * mask_r
        g2, b2 = [rearrange(x, "b l n h -> b l (n h)") for x in (g2, b2)]

        # Auxiliary loss calculation
        aux_loss = torch.zeros(()).to(eta)
        if self.training:
            p = torch.mean(eta.float(), dim=(0, 1))
            f = torch.mean(mask_w.float(), dim=(0, 1))
            aux_loss = torch.sum(p * f) * self.num_sparse_partition / self.num_writer
        
        # Concatenate queries, keys, gates, betas and values
        q, k, g, b, v = [torch.cat(pair, dim=2) for pair in zip((q1, k1, g1, b1, v), (q2, k2, g2, b2, v2))]

        # Get fixed-sized storage
        if use_cache and fixed_storage is None:
            fixed_storage = self._get_fixed_state_storage(bsz, q.device)

        # GDN linear attention computation
        if mode == 'chunk':
            o, recurrent_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=b,
                initial_state=fixed_storage if use_cache else None,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
        elif mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=b,
                initial_state=fixed_storage if use_cache else None,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        # Update fixed storage
        if use_cache:
            fixed_storage = recurrent_state

        # Reshape output
        o = rearrange(o, "b l (n h) d -> b l n h d", n=self.num_sparse_partition+1)
        o = o.sum(2)

        return o, fixed_storage, aux_loss

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Forward pass for GDN with fixed-sized storage management.
        """
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape

        # Get fixed storage from past_key_values
        fixed_storage = None
        conv_state = None
        if past_key_values is not None and self.layer_idx in past_key_values:
            layer_state = past_key_values[self.layer_idx]
            fixed_storage = layer_state.get('fixed_storage')
            conv_state = layer_state.get('conv_state')

        # Handle padding and unpadding
        cu_seqlens = kwargs.get('cu_seqlens')
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        # Project inputs
        q1, _ = self.q_proj(hidden_states)
        k1, _ = self.k_proj(hidden_states)
        # LoRA projections (manually execute two steps to handle tuple outputs)
        lora_q_output, _ = self.lora_q_proj_0(hidden_states)  # Step 1
        lora_q_output, _ = self.lora_q_proj_1(lora_q_output)  # Step 2
        
        lora_k_output, _ = self.lora_k_proj_0(hidden_states)  # Step 1
        lora_k_output, _ = self.lora_k_proj_1(lora_k_output)  # Step 2
        q2 = q1 + lora_q_output
        k2 = k1 + lora_k_output
        v, _ = self.v_proj(hidden_states)

        # GDN specific projections
        b = self.b_proj(hidden_states)[0].sigmoid()
        if self.allow_neg_eigval:
            b = b * 2.
        
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states)[0].float() + self.dt_bias)
        b1, b2 = torch.chunk(b, 2, dim=-1)
        g1, g2 = torch.chunk(g, 2, dim=-1)
        
        eta, _ = self.e_proj(hidden_states)

        # Handle short convolutions if enabled
        conv_state_q, conv_state_k = None, None
        if self.use_short_conv:
            conv_state_q, conv_state_k = (conv_state[0], conv_state[1]) if conv_state is not None else (None, None)
            
            q1, conv_state_q = self.q_conv1d_shared(
                x=q1,
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k1, conv_state_k = self.k_conv1d_shared(
                x=k1,
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )

        # Reshape for multi-head attention (tensor parallel aware)
        q1, q2, k1, k2 = map(
            lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim, h=self.tp_heads),
            (q1, q2, k1, k2)
        )
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim, h=self.tp_v_heads)
        g1 = rearrange(g1, '... (h) -> ... h', h=self.tp_heads)
        g2 = rearrange(g2, '... (h) -> ... h', h=self.tp_heads)
        b1 = rearrange(b1, '... (h) -> ... h', h=self.tp_heads)
        b2 = rearrange(b2, '... (h) -> ... h', h=self.tp_heads)

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

        # Handle grouped value attention (tensor parallel aware)
        if self.num_v_heads > self.num_heads:
            group_factor = self.num_v_heads // self.num_heads
            q1, q2, k1, k2 = map(
                lambda x: repeat(x, '... h d -> ... (h g) d', g=group_factor),
                (q1, q2, k1, k2)
            )

        # SSE-GDN linear attention with fixed storage
        o, fixed_storage, aux_loss = self.sse_implementation(
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
            fixed_storage=fixed_storage,
            use_cache=use_cache,
            cu_seqlens=cu_seqlens,
        )

        # Update past key values with fixed storage
        if use_cache and past_key_values is not None:
            past_key_values[self.layer_idx] = {
                'fixed_storage': fixed_storage,
                'conv_state': (conv_state_q, conv_state_k) if self.use_short_conv else None
            }

        # Apply output gate and normalization
        if self.use_output_gate:
            # Output gate projection (manually execute two steps to handle tuple outputs)
            g, _ = self.g_proj_0(hidden_states)  # Step 1
            g, _ = self.g_proj_1(g)              # Step 2
            g = rearrange(g, '... (h d) -> ... h d', d=self.head_v_dim, h=self.tp_v_heads)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        
        # Project output
        o = rearrange(o, 'b t h d -> b t (h d)')
        o, _ = self.o_proj(o)
        
        # Repad if needed
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, aux_loss, past_key_values
    
    def load_hf_weights(self, hf_weights: dict, prefix: str = ""):
        """
        Load weights from huggingface implementation for GDN variant.
        """
        # Load common weights from parent class
        super().load_hf_weights(hf_weights, prefix)
        
        # Load GDN specific projections
        gdn_proj_layers = ['a_proj', 'b_proj']
        for layer_name in gdn_proj_layers:
            if hasattr(self, layer_name):
                weight_key = f"{prefix}{layer_name}.weight"
                if weight_key in hf_weights:
                    getattr(self, layer_name).weight.data.copy_(hf_weights[weight_key])
        
        # Load GDN specific parameters
        param_names = ['A_log', 'dt_bias']
        for param_name in param_names:
            if hasattr(self, param_name):
                param_key = f"{prefix}{param_name}"
                if param_key in hf_weights:
                    getattr(self, param_name).data.copy_(hf_weights[param_key])
        
        logger.info(f"Loaded SSE-GDN layer weights for {prefix}")


# Register the models with vLLM
# @register_model("sse_gla")
# def get_sse_gla(
#     vllm_config: VllmConfig,
#     prefix: str,
#     **kwargs,
# ) -> VLLMSSEGLA:
#     return VLLMSSEGLA(vllm_config, prefix, **kwargs)


# @register_model("sse_gdn")  
# def get_sse_gdn(
#     vllm_config: VllmConfig,
#     prefix: str,
#     **kwargs,
# ) -> VLLMSSEGDN:
#     return VLLMSSEGDN(vllm_config, prefix, **kwargs)


__all__ = [
    "VLLMSSEGLA",
    "VLLMSSEGDN",
    # "get_sse_gla",
    # "get_sse_gdn",
]