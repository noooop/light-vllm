
from typing import Optional
from dataclasses import dataclass
import torch
from light_vllm.wde.core.schema.execute_io import WorkerInput, ModelInput, ExecuteInput, ExecuteOutput


@dataclass
class DecodeOnlyWorkerInput(WorkerInput):
    """Local inputs to each worker. May contain device-specific data. These
    fields should be broadcastable to other workers.
    """

    num_seq_groups: Optional[int] = None
    blocks_to_swap_in: Optional[torch.Tensor] = None
    blocks_to_swap_out: Optional[torch.Tensor] = None
    blocks_to_copy: Optional[torch.Tensor] = None
    virtual_engine: int = 0
    num_steps: int = 1


@dataclass
class DecodeOnlyExecuteInput(ExecuteInput):
    worker_input: Optional[DecodeOnlyWorkerInput]
    model_input: Optional[ModelInput]


class DecodeOnlyExecuteOutput(ExecuteOutput):
    pass