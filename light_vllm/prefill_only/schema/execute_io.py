from dataclasses import dataclass

import torch

from light_vllm.backends.attention import AttentionMetadata
from light_vllm.core.schema.execute_io import ExecuteInput, ModelInput


@dataclass
class ModelInputForGPU(ModelInput):
    input_ids: torch.Tensor
    positions: torch.Tensor
    attn_metadata: AttentionMetadata

    def to(self, target_device, non_blocking=False):
        for k in self.__dict__:
            self.__dict__[k] = self.__dict__[k].to(device=target_device,
                                                   non_blocking=non_blocking)

    def to_dict(self):
        out = self.__dict__

        if "kv_caches" not in out:
            out["kv_caches"] = None

        return out


class PrefillOnlyExecuteInput(ExecuteInput):
    worker_input = None
    model_input: ModelInputForGPU
