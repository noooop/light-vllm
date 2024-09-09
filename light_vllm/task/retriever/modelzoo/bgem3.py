

from typing import Iterable, List, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from light_vllm.task.encode_only.modelzoo.xlm_roberta import XLMRobertaModel, XLMRobertaConfig
from light_vllm.layers.quantization.base_config import (
    QuantizationConfig)

from light_vllm.task.base.loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from light_vllm.models.utils import is_pp_missing_parameter


class BGEM3Model(nn.Module):
    _ignore_weights_keys = ["roberta.pooler.dense.weight", "roberta.pooler.dense.bias"]

    def __init__(self,
                 config: XLMRobertaConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 sentence_pooling_method="cls",
                 normlized=True,
                 *args, **kwargs):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.sentence_pooling_method = sentence_pooling_method
        assert self.sentence_pooling_method == 'cls'
        self.normlized = normlized
        self.roberta = XLMRobertaModel(config, quant_config)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

        sequence_output = self.roberta(
            input_ids,
            attention_mask=attention_mask,
        )

        dense_vecs = sequence_output[cu_seqlens[:-1]]

        if self.normlized:
            dense_vecs = torch.nn.functional.normalize(dense_vecs, dim=-1)

        return dense_vecs

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "query", "q"),
            ("qkv_proj", "key", "k"),
            ("qkv_proj", "value", "v")
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            name = "roberta." + name

            if name in self._ignore_weights_keys:
                continue

            if name == "roberta.embeddings.token_type_embeddings.weight":
                # token_type_ids is all zero, so we only need token_type_embeddings[0]
                self.roberta.embeddings.init_token_type_embeddings0()
                default_weight_loader(self.roberta.embeddings.token_type_embeddings0, loaded_weight[0])
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