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


class XLMRobertaFLashAttentionSelfAttention(nn.Module):
    def __init__(self,
                 config: XLMRobertaConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
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

        self.scaling = self.head_dim ** -0.5

    def forward(
            self,
            hidden_states: torch.Tensor,
            attn_metadata=None,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        query_states = query_states.view(-1, self.num_heads, self.head_dim)
        key_states = key_states.view(-1, self.num_kv_heads, self.head_dim)
        value_states = value_states.view(-1, self.num_kv_heads, self.head_dim)

        seqlens_in_batch, max_seqlen_in_batch, cu_seqlens = attn_metadata

        attn_output = flash_attn_varlen_func(
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
        attn_output = attn_output.view(-1, self.num_heads*self.head_dim)
        return attn_output


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
            attn_metadata=None,
    ) -> torch.Tensor:
        self_outputs = self.self(
            hidden_states,
            attn_metadata
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
        self.intermediate_act_fn = get_act_fn(config.hidden_act)

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
            attn_metadata=None,
    ) -> torch.Tensor:
        attention_output = self.attention(
            hidden_states,
            attn_metadata
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
            attn_metadata=None,
    ) -> torch.Tensor:
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(
                hidden_states,
                attn_metadata
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
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

        input_ids = input_ids.view(-1)[indices]
        position_ids = torch.zeros_like(input_ids)
        for offset, n in zip(cu_seqlens, seqlens_in_batch):
            position_ids[offset:offset+n] = torch.arange(self.config.pad_token_id+1,
                                                          self.config.pad_token_id+1+n,
                                                          dtype=position_ids.dtype,
                                                          device=position_ids.device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
        )

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attn_metadata=(seqlens_in_batch, max_seqlen_in_batch, cu_seqlens)
        )

        return encoder_outputs


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
        batchsize = input_ids.shape[0]
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

        sequence_output = self.roberta(
            input_ids,
            attention_mask=attention_mask,
        )
        logits = self.lm_head(sequence_output)

        logits_list = []
        for i in range(batchsize):
            logits_list.append(logits[cu_seqlens[i]:cu_seqlens[i+1]])
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