from dataclasses import dataclass

import torch

from light_vllm.core.schema.execute_io import ExecuteOutput


@dataclass
class RerankerExecuteOutput(ExecuteOutput):
    scores: torch.Tensor
