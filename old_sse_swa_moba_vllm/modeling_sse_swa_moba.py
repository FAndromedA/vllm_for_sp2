# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Converted to vllm implementation

from __future__ import annotations
import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import BaseModelOutputWithPast, MoeCausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
# from vllm.model_executor.models.registry import register_model
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer

logger = logging.get_logger(__name__)

# Import vllm-compatible attention modules
from .attn import VLLMAttention, KVCacheManager
from .moba_attn import VLLMMoBAAttention
from .sse_swa import VLLMSSEGLAH, VLLMSSEGDNH
from .sse import VLLMSSEGLA, VLLMSSEGDN
from .configuration_ssw_swa_moba import SSESWAMoBAConfig


if TYPE_CHECKING:
    from transformers.processing_utils import Unpack
    from vllm.model_executor.models.utils import Cache


class VLLMSSESWAMoBABlock(GradientCheckpointingLayer):
    """
    vLLM-compatible SSE-SWA-MoBA block containing a mixture of attention mechanisms.
    """
    def __init__(self, vllm_config: VllmConfig, config: SSESWAMoBAConfig, layer_idx: int):
        super().__init__()

        self.vllm_config = vllm_config
        self.config = config
        self.layer_idx = layer_idx
        self.prefix = f"model.layers.{layer_idx}"

        # Tensor parallel configuration
        self.tp_size = get_tensor_model_parallel_world_size()

        # Normalization layers
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

        # Attention layer selection based on config
        self.attn = self._create_attention_layer()

        # MLP layer
        self.mlp = self._create_mlp_layer()

    def _create_attention_layer(self):
        """Create appropriate attention layer based on configuration"""
        config = self.config
        layer_idx = self.layer_idx

        if config.attn is not None and layer_idx in config.attn['layers']:
            if layer_idx in config.attn['full_layers']:
                # Global Attention Layers (vllm-compatible)
                return VLLMAttention(
                    vllm_config=self.vllm_config,
                    prefix=f"{self.prefix}.attn",
                    hidden_size=config.hidden_size,
                    num_heads=config.attn['num_heads'],
                    num_kv_heads=config.attn['num_kv_heads'],
                    head_dim=config.head_dim,
                    qkv_bias=config.attn['qkv_bias'],
                    qk_norm=config.attn['qk_norm'],
                    window_size=None,  # Full attention
                    rope_theta=config.attn['rope_theta'],
                    max_position_embeddings=config.max_position_embeddings,
                    layer_idx=layer_idx,
                    norm_eps=config.norm_eps,
                )
            else:
                # MoBA Attention Layers (vllm-compatible)
                return VLLMMoBAAttention(
                    vllm_config=self.vllm_config,
                    prefix=f"{self.prefix}.attn",
                    hidden_size=config.hidden_size,
                    num_heads=config.attn['num_heads'],
                    num_kv_heads=config.attn['num_kv_heads'],
                    head_dim=config.head_dim,
                    qkv_bias=config.attn['qkv_bias'],
                    qk_norm=config.attn['qk_norm'],
                    window_size=None,
                    rope_theta=config.attn['rope_theta'],
                    moba_chunk_size=config.attn['moba_chunk_size'],
                    moba_topk=config.attn['moba_topk'],
                    max_position_embeddings=config.max_position_embeddings,
                    layer_idx=layer_idx,
                    norm_eps=config.norm_eps,
                )
        else:
            # Linear & Window Attention Layers (vllm-compatible)
            if config.linear_attn_type == "gla":
                if hasattr(config, 'num_v_heads'):
                    # SSE-GLAH (with SWA)
                    return VLLMSSEGLAH(
                        vllm_config=self.vllm_config,
                        prefix=f"{self.prefix}.attn",
                        hidden_size=config.hidden_size,
                        num_heads=config.num_heads,
                        num_v_heads=config.num_v_heads,
                        head_dim=config.head_dim,
                        mode=config.attn_mode,
                        use_output_gate=config.use_output_gate,
                        use_short_conv=config.use_short_conv,
                        conv_size=config.conv_size,
                        num_sparse_partition=config.num_sparse_partition,
                        num_writer=config.num_writer,
                        num_reader=config.num_reader,
                        sse_implementation=config.sse_implementation,
                        sse_qk_relu=config.sse_qk_relu,
                        qkv_bias=config.attn['qkv_bias'],
                        swa_num_kv_heads=config.attn['num_kv_heads'],
                        swa_qk_norm=config.attn['qk_norm'],
                        swa_dropout=config.swa_dropout,
                        window_size=config.attn['window_size'],
                        rope_theta=config.attn['rope_theta'],
                        max_position_embeddings=config.max_position_embeddings,
                        layer_idx=layer_idx,
                        norm_eps=config.norm_eps,
                    )
                else:
                    # SSE-GLA (pure linear)
                    return VLLMSSEGLA(
                        vllm_config=self.vllm_config,
                        prefix=f"{self.prefix}.attn",
                        hidden_size=config.hidden_size,
                        num_heads=config.num_heads,
                        num_v_heads=config.num_heads,  # Default to same as num_heads
                        head_dim=config.head_dim,
                        mode=config.attn_mode,
                        use_output_gate=config.use_output_gate,
                        use_short_conv=config.use_short_conv,
                        conv_size=config.conv_size,
                        num_sparse_partition=config.num_sparse_partition,
                        num_writer=config.num_writer,
                        num_reader=config.num_reader,
                        sse_implementation=config.sse_implementation,
                        sse_qk_relu=config.sse_qk_relu,
                        use_q_softmax=config.use_q_softmax,
                        use_k_softmax=config.use_k_softmax,
                        emulq=config.emulq,
                        emulk=config.emulk,
                        layer_idx=layer_idx,
                        norm_eps=config.norm_eps,
                    )
            elif config.linear_attn_type == "gdn":
                if hasattr(config, 'num_v_heads'):
                    # SSE-GDNH (with SWA)
                    return VLLMSSEGDNH(
                        vllm_config=self.vllm_config,
                        prefix=f"{self.prefix}.attn",
                        hidden_size=config.hidden_size,
                        num_heads=config.num_heads,
                        num_v_heads=config.num_v_heads,
                        head_dim=config.head_dim,
                        mode=config.attn_mode,
                        use_output_gate=config.use_output_gate,
                        use_short_conv=config.use_short_conv,
                        allow_neg_eigval=config.allow_neg_eigval,
                        conv_size=config.conv_size,
                        num_sparse_partition=config.num_sparse_partition,
                        num_writer=config.num_writer,
                        num_reader=config.num_reader,
                        sse_implementation=config.sse_implementation,
                        sse_qk_relu=config.sse_qk_relu,
                        qkv_bias=config.attn['qkv_bias'],
                        swa_num_kv_heads=config.attn['num_kv_heads'],
                        swa_qk_norm=config.attn['qk_norm'],
                        swa_dropout=config.swa_dropout,
                        window_size=config.attn['window_size'],
                        rope_theta=config.attn['rope_theta'],
                        max_position_embeddings=config.max_position_embeddings,
                        layer_idx=layer_idx,
                        norm_eps=config.norm_eps,
                    )
                else:
                    # SSE-GDN (pure linear)
                    return VLLMSSEGDN(
                        vllm_config=self.vllm_config,
                        prefix=f"{self.prefix}.attn",
                        hidden_size=config.hidden_size,
                        num_heads=config.num_heads,
                        num_v_heads=config.num_heads,  # Default to same as num_heads
                        head_dim=config.head_dim,
                        mode=config.attn_mode,
                        use_output_gate=config.use_output_gate,
                        use_short_conv=config.use_short_conv,
                        allow_neg_eigval=config.allow_neg_eigval,
                        conv_size=config.conv_size,
                        num_sparse_partition=config.num_sparse_partition,
                        num_writer=config.num_writer,
                        num_reader=config.num_reader,
                        sse_implementation=config.sse_implementation,
                        sse_qk_relu=config.sse_qk_relu,
                        use_q_softmax=config.use_q_softmax,
                        use_k_softmax=config.use_k_softmax,
                        emulq=config.emulq,
                        emulk=config.emulk,
                        layer_idx=layer_idx,
                        norm_eps=config.norm_eps,
                    )
            else:
                raise ValueError(f"Unknown linear attention type: {config.linear_attn_type}")

    def _create_mlp_layer(self):
        """Create MLP layer with vllm parallel linear layers"""
        config = self.config
        
        # Use vllm's parallel linear layers for MLP
        intermediate_size = config.intermediate_size or int(config.hidden_size * config.hidden_ratio)
        
        return nn.Sequential(
            ColumnParallelLinear(
                config.hidden_size, intermediate_size, bias=False, 
                prefix=f"{self.prefix}.mlp.gate_proj"
            ),
            ColumnParallelLinear(
                config.hidden_size, intermediate_size, bias=False,
                prefix=f"{self.prefix}.mlp.up_proj"
            ),
            RowParallelLinear(
                intermediate_size, config.hidden_size, bias=False,
                prefix=f"{self.prefix}.mlp.down_proj"
            )
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs: Unpack[dict],
    ) -> Tuple[torch.Tensor, Optional[Dict[int, Dict[str, torch.Tensor]]], Optional[torch.Tensor]]:
        """
        Forward pass with vllm-compatible cache management.
        """
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        
        # Attention forward pass with vllm cache format
        attention_output, past_key_values, aux_loss = self.attn(
            hidden_states=hidden_states,
            positions=positions,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        
        # Residual connection
        hidden_states = residual + attention_output
        
        # MLP forward pass
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        
        # MLP with SwiGLU activation
        gate = self.mlp[0](hidden_states)
        up = self.mlp[1](hidden_states)
        hidden_states = F.silu(gate) * up
        hidden_states = self.mlp[2](hidden_states)
        
        # Final residual connection
        hidden_states = residual + hidden_states
        
        return hidden_states, past_key_values, aux_loss


@dataclass
class VLLMMoeModelOutputWithPastAndAuxLosses(BaseModelOutputWithPast):
    """
    vLLM-compatible output class with aux losses.
    """
    aux_losses: Optional[Tuple[torch.FloatTensor]] = None


class VLLMSSESWAMoBAModel(PreTrainedModel):
    """
    vLLM-compatible SSE-SWA-MoBA model with mixed attention mechanisms.
    """
    config_class = SSESWAMoBAConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['VLLMSSESWAMoBABlock']
    _supports_cache_class = True

    def __init__(self, vllm_config: VllmConfig, config: SSESWAMoBAConfig):
        super().__init__(config)
        self.vllm_config = vllm_config
        self.config = config
        
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Embedding layer using vllm's ParallelEmbedding
        self.embeddings = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, 
            padding_idx=self.padding_idx,
            prefix="model.embeddings"
        )

        # Create layers with vllm-compatible blocks
        self.layers = nn.ModuleList([
            VLLMSSESWAMoBABlock(vllm_config, config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])

        # Final normalization
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs: Unpack[dict],
    ) -> Tuple | VLLMMoeModelOutputWithPastAndAuxLosses:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        output_aux_losses = True
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Validate inputs
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # Get inputs embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds

        # Initialize positions if not provided
        if positions is None and input_ids is not None:
            batch_size, seq_len = input_ids.shape
            positions = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).repeat(batch_size, 1)

        # Initialize past key values if needed
        if use_cache and past_key_values is None:
            past_key_values = {}

        all_hidden_states = () if output_hidden_states else None
        all_aux_losses = () if output_aux_losses else None

        # Process each layer
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Layer forward pass with vllm cache
            hidden_states, past_key_values, aux_loss = layer(
                hidden_states=hidden_states,
                positions=positions,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

            if output_aux_losses and aux_loss is not None:
                all_aux_losses += (aux_loss,)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(i for i in [hidden_states, past_key_values, all_hidden_states, all_aux_losses] if i is not None)
        
        return VLLMMoeModelOutputWithPastAndAuxLosses(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            aux_losses=all_aux_losses,
        )


class VLLMSSESWAMoBAForCausalLM(VLLMSSESWAMoBAModel):
    """
    vLLM-compatible SSE-SWA-MoBA model for causal language modeling.
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, vllm_config: VllmConfig, config: SSESWAMoBAConfig):
        super().__init__(vllm_config, config)
        
        # LM head using vllm's RowParallelLinear
        self.lm_head = RowParallelLinear(
            config.hidden_size, config.vocab_size, bias=False,
            prefix="lm_head"
        )
        
        self.aux_loss_coef = config.aux_loss_coef

        # Tie weights if needed
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embeddings.weight

        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        logits_to_keep: int = 0,
        **kwargs: Unpack[dict],
    ) -> Tuple | MoeCausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Model forward pass
        outputs = super().forward(
            input_ids=input_ids,
            positions=positions,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]

        loss, aux_loss, logits = None, None, None
        
        # Compute logits
        if logits_to_keep > 0:
            logits = self.lm_head(hidden_states[:, -logits_to_keep:])
        else:
            logits = self.lm_head(hidden_states)

        # Compute loss if labels are provided
        if labels is not None:
            # Shift labels for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Add auxiliary losses if present
            if hasattr(outputs, 'aux_losses') and outputs.aux_losses:
                aux_loss = sum(l.to(loss.device) for l in outputs.aux_losses)
                loss += self.aux_loss_coef * aux_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            aux_loss=aux_loss,
        )


# Register the model with vLLM
# @register_model("sse_swa_moba")
# def get_sse_swa_moba(
#     vllm_config: VllmConfig,
#     **kwargs,
# ) -> VLLMSSESWAMoBAForCausalLM:
#     """
#     Register SSE-SWA-MoBA model with vLLM.
#     """
#     # Create config from kwargs or use default
#     config = SSESWAMoBAConfig(**kwargs)
#     return VLLMSSESWAMoBAForCausalLM(vllm_config, config)


__all__ = [
    "VLLMSSESWAMoBABlock",
    "VLLMSSESWAMoBAModel",
    "VLLMSSESWAMoBAForCausalLM",
    "get_sse_swa_moba",
]