

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
        self.normlized = normlized

        self.roberta = XLMRobertaModel(config, quant_config, add_pooling_layer=False)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:

        sequence_output = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        if self.sentence_pooling_method == 'cls':
            dense_vecs = sequence_output[:, 0]
        elif self.sentence_pooling_method == 'mean':
            s = torch.sum(sequence_output * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(axis=1, keepdim=True).float()
            dense_vecs = s / d

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