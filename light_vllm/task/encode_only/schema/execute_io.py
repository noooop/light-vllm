

from dataclasses import dataclass
import torch
from light_vllm.task.base.schema.execute_io import ModelInput, ExecuteInput
from light_vllm.task.encode_only.layers.attention import EncodeOnlyAttentionMetadata


@dataclass
class ModelInputForGPU(ModelInput):
    input_ids: torch.Tensor
    positions: torch.Tensor
    attn_metadata: EncodeOnlyAttentionMetadata

    def to(self, device):
        for k in self.__dict__.keys():
            self.__dict__[k] = self.__dict__[k].to(device)

    def to_dict(self):
        return self.__dict__


class EncodeOnlyExecuteInput(ExecuteInput):
    worker_input = None
    model_input: ModelInputForGPU