# Derived from XLM-RoBERTa implementation posted on HuggingFace; license below:
# coding=utf-8
# Copyright 2019 Facebook AI Research and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch XLM-RoBERTa model."""

import math
from typing import Iterable, List, Optional, Tuple, Union
import torch
from torch import nn
from transformers import XLMRobertaConfig
from transformers.utils import logging
from light_vllm.layers.activation import get_act_fn
from light_vllm.layers.linear import QKVParallelLinear, RowParallelLinear, ColumnParallelLinear
from light_vllm.layers.quantization.base_config import (
    QuantizationConfig)
import torch.nn.functional as F
from vllm_flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
#from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from light_vllm.task.base.loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from light_vllm.models.utils import is_pp_missing_parameter

logger = logging.get_logger(__name__)


class XLMRobertaEmbeddings(nn.Module):
    def __init__(self, config: XLMRobertaConfig):
        super().__init__()
        self.config = config
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        assert self.position_embedding_type == "absolute"

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings0 = None
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def init_token_type_embeddings0(self):
        del self.token_type_embeddings0
        self.register_buffer(
            "token_type_embeddings0",
            torch.zeros(self.config.hidden_size,
                        dtype=self.word_embeddings.weight.dtype,
                        device=self.word_embeddings.weight.device)
        )

    def forward(self, input_ids, position_ids):
        embeddings = self.word_embeddings(input_ids)

        # token_type_embeddings is all zero in FacebookAI/xlm-roberta, so we don't need it.
        # token_type_ids is all zero in BGEM3, so we only need token_type_embeddings[0]
        if self.token_type_embeddings0 is not None:
            token_type_embeddings = self.token_type_embeddings0
            embeddings += token_type_embeddings

        embeddings += self.position_embeddings(position_ids)
        embeddings = self.LayerNorm(embeddings)
        return embeddings


class XLMRobertaSelfAttention(nn.Module):
    def __init__(self,
                 config: XLMRobertaConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_attention_heads

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            num_heads,
            num_kv_heads,
            bias=True,
            quant_config=quant_config,
        )

        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        extended_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        query_layer = self.transpose_for_scores(q)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if extended_attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in XLMRobertaModel forward() function)
            attention_scores = attention_scores + extended_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return context_layer


class XLMRobertaSPDASelfAttention(XLMRobertaSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.require_contiguous_qkv = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        extended_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:

        bsz, tgt_len, _ = hidden_states.size()

        qkv, _ = self.qkv_proj(hidden_states)
        query, key, value = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        query_layer = self.transpose_for_scores(query)

        attention_mask = attention_mask

        # Check `seq_length` of `past_key_value` == `len(current_states)` to support prefix tuning

        key_layer = self.transpose_for_scores(key)
        value_layer = self.transpose_for_scores(value)

        # SDPA with memory-efficient backend is broken in torch==2.1.2 when using non-contiguous inputs and a custom
        # attn_mask, so we need to call `.contiguous()` here. This was fixed in torch==2.2.0.
        # Reference: https://github.com/pytorch/pytorch/issues/112577
        if self.require_contiguous_qkv and query_layer.device.type == "cuda" and attention_mask is not None:
            query_layer = query_layer.contiguous()
            key_layer = key_layer.contiguous()
            value_layer = value_layer.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=extended_attention_mask,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        return attn_output


class XLMRobertaFLashAttentionSelfAttention(XLMRobertaSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.require_contiguous_qkv = False
        self.scaling = self.head_dim ** -0.5

    def unpad_input(self, hidden_states, attention_mask):
        valid_mask = attention_mask.squeeze(1).squeeze(1).eq(0)
        seqlens_in_batch = valid_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(valid_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
        hidden_states = hidden_states[indices]
        return hidden_states, indices, cu_seqlens, max_seqlen_in_batch

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            extended_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        batch_size, seqlen, _ = hidden_states.size()
        qkv, _ = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        query_states = query_states.view(-1, self.num_heads, self.head_dim)
        key_states = key_states.view(-1, self.num_kv_heads, self.head_dim)
        value_states = value_states.view(-1, self.num_kv_heads, self.head_dim)

        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

        query_states = query_states[indices]
        key_states = key_states[indices]
        value_states = value_states[indices]

        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen_in_batch,
            max_seqlen_k=max_seqlen_in_batch,
            softmax_scale=self.scaling,
            causal=False
        )
        hidden_states = torch.empty_like(hidden_states)
        hidden_states = hidden_states.view(batch_size * seqlen, -1)
        hidden_states[indices] = attn_output_unpad.view(-1, self.num_heads*self.head_dim)
        out = hidden_states.view(batch_size, seqlen, -1)
        return out


class XLMRobertaSelfOutput(nn.Module):
    def __init__(self,
                 config: XLMRobertaConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.dense = ColumnParallelLinear(config.hidden_size, config.hidden_size, quant_config=quant_config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class XLMRobertaAttention(nn.Module):
    def __init__(self,
                 config: XLMRobertaConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.self = XLMRobertaFLashAttentionSelfAttention(config)
        self.output = XLMRobertaSelfOutput(config, quant_config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            extended_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            extended_attention_mask
        )
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output


class XLMRobertaIntermediate(nn.Module):
    def __init__(self,
                 config: XLMRobertaConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.dense = RowParallelLinear(config.hidden_size,
                                       config.intermediate_size,
                                       bias=True,
                                       quant_config=quant_config)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_act_fn(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class XLMRobertaOutput(nn.Module):
    def __init__(self,
                 config: XLMRobertaConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.dense = RowParallelLinear(config.intermediate_size, config.hidden_size,
                                       bias=True,
                                       quant_config=quant_config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class XLMRobertaLayer(nn.Module):
    def __init__(self,
                 config: XLMRobertaConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.attention = XLMRobertaAttention(config, quant_config)
        self.intermediate = XLMRobertaIntermediate(config, quant_config)
        self.output = XLMRobertaOutput(config, quant_config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            extended_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            extended_attention_mask
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class XLMRobertaEncoder(nn.Module):
    def __init__(self,
                 config: XLMRobertaConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.layer = nn.ModuleList([XLMRobertaLayer(config, quant_config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            extended_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(
                hidden_states,
                attention_mask,
                extended_attention_mask
            )
        return hidden_states


class XLMRobertaModel(nn.Module):
    def __init__(self,
                 config: XLMRobertaConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.config = config
        self.embeddings = XLMRobertaEmbeddings(config)
        self.encoder = XLMRobertaEncoder(config, quant_config)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:

        input_shape = input_ids.size()
        position_ids = create_position_ids_from_input_ids(input_ids, self.config.pad_token_id, 0)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
        )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape,
                                                                                 dtype=embedding_output.dtype)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            extended_attention_mask=extended_attention_mask,
        )
        return encoder_outputs

    def get_extended_attention_mask(
            self, attention_mask: torch.Tensor, input_shape: Tuple[int], dtype: torch.float = None
    ) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask


class XLMRobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self,
                 config: XLMRobertaConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.dense = ColumnParallelLinear(config.hidden_size, config.hidden_size, quant_config=quant_config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = ColumnParallelLinear(config.hidden_size, config.vocab_size, quant_config=quant_config)
        self.gelu = get_act_fn("gelu")

    def forward(self, features):
        x, _ = self.dense(features)
        x = self.gelu(x)
        x = self.layer_norm(x)
        x, _ = self.decoder(x)
        return x


class XLMRobertaForMaskedLM(nn.Module):
    _ignore_weights_keys = ["roberta.pooler.dense.weight",
                            "roberta.pooler.dense.bias",
                            # token_type_embeddings is all zero
                            "roberta.embeddings.token_type_embeddings.weight"]

    def __init__(self,
                 config: XLMRobertaConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 *args, **kwargs):
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        self.roberta = XLMRobertaModel(config, quant_config)
        self.lm_head = XLMRobertaLMHead(config, quant_config)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:

        sequence_output = self.roberta(
            input_ids,
            attention_mask=attention_mask,
        )
        logits = self.lm_head(sequence_output)

        seq_len = attention_mask.sum(axis=1)

        logits_list = []
        for e, s in zip(logits, seq_len):
            logits_list.append(e[:s])
        return logits_list

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "query", "q"),
            ("qkv_proj", "key", "k"),
            ("qkv_proj", "value", "v")
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if name in self._ignore_weights_keys:
                continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue

                name = name.replace(weight_name, param_name)

                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

        self.tie_weights()

    def tie_weights(self):
        self.lm_head.decoder.weight = self.roberta.embeddings.word_embeddings.weight
        self.lm_head.decoder.bias.zero_()


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx
