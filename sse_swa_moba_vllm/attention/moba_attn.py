# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashAttention."""

# SSE 可以直接用 GDN backend, 不用融合了，这样一来，
# MoBA backend 也可以直接复用 FlashAttn backend, super().forward
# 当且仅当 is_moba = True 的时候才跑自己的 forward

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    MultipleOf,
    is_quantized_kv_cache,
)
from vllm.attention.layer import Attention
from vllm.attention.ops.common import cp_lse_ag_out_rs
from vllm.attention.ops.merge_attn_states import merge_attn_states
from vllm.attention.utils.fa_utils import (
    flash_attn_supports_fp8,
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)

if is_flash_attn_varlen_func_available():
    from vllm.attention.utils.fa_utils import (
        flash_attn_supports_sinks,
        flash_attn_varlen_func,
        get_scheduler_metadata,
        reshape_and_cache_flash,
    )
from vllm.config import VllmConfig, get_current_vllm_config, get_layers_from_vllm_config
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    get_dcp_local_seq_lens,
    get_kv_cache_layout,
    PAD_SLOT_ID,
    compute_causal_conv1d_metadata,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

logger = init_logger(__name__)

from .moba_attn_ops import moba_attn_varlen
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend, 
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
class MobaSseFlashAttentionBackend(FlashAttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "MOBA_SSE_FLASH_ATTN"
    
    @staticmethod
    def get_impl_cls() -> type["SseMobaFlashAttentionImpl"]:
        return SseMobaFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["FlashAttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder

class SseMobaFlashAttentionImpl(FlashAttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
        is_moba: bool = False,
        moba_topk: int | None = None,
        moba_chunk_size: int | None = None,
    ):
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            sinks=sinks,
        )
        self.is_moba = is_moba
        self.moba_topk = moba_topk
        self.moba_chunk_size = moba_chunk_size

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.is_moba:
            return super().forward(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )
        else: # MoBA attention
            # num of decode seqs and tokens
            num_decodes = attn_metadata.num_decodes
            num_decode_tokens = attn_metadata.num_decode_tokens
            num_prefills = attn_metadata.num_prefills
            num_prefill_tokens = attn_metadata.num_prefill_tokens

            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table
            scheduler_metadata = attn_metadata.scheduler_metadata

            if num_prefill_tokens > 0:
                q_p = query[num_decode_tokens: num_decode_tokens + num_prefill_tokens]
                k_p = key[num_decode_tokens: num_decode_tokens + num_prefill_tokens]
                v_p = value[num_decode_tokens: num_decode_tokens + num_prefill_tokens]
                cu_base = cu_seqlens_q[num_decodes]
                cu_seqlens_q_p = cu_seqlens_q[num_decodes: num_decodes + num_prefills + 1] - cu_base
                
                out_prefill = moba_attn_varlen(
                    q=q_p,
                    k=k_p,
                    v=v_p,
                    cu_seqlens=cu_seqlens_q_p,
                    max_seqlen=max_seqlen_q, # because decodes max is 1, so prefill max is same as query max
                    moba_chunk_size=self.moba_chunk_size, # TODO: add to config
                    moba_topk=self.moba_topk,
                )
            else:
                out_prefill = None
            output[num_decode_tokens: num_decode_tokens + num_prefill_tokens] = out_prefill

            if attn_metadata.num_decode_tokens > 0:
                key_cache, value_cache = kv_cache.unbind(0)
                descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)

                flash_attn_varlen_func(
                    q=query[:num_decode_tokens],
                    k=key_cache,
                    v=value_cache,
                    out=output[:num_decode_tokens],
                    cu_seqlens_q=cu_seqlens_q[:num_decodes + 1],
                    max_seqlen_q=max_seqlen_q,
                    seqused_k=seqused_k,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    alibi_slopes=self.alibi_slopes,
                    window_size=self.sliding_window,
                    block_table=block_table,
                    softcap=self.logits_soft_cap,
                    scheduler_metadata=scheduler_metadata,
                    fa_version=self.vllm_flash_attn_version,
                    q_descale=layer._q_scale.expand(descale_shape),
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                    num_splits=attn_metadata.max_num_splits,
                    s_aux=self.sinks,
                )
            return output

