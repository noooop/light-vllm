
from dataclasses import dataclass, field
from abc import ABC
from typing import Optional, List, Tuple

import torch
from light_vllm.task.base.schema.sequence import SequenceGroupMetadata


@dataclass
class ExecuteModelInput:
    """The model execution request, containing CPU metadata only. The LLM
    engine should create an instance of this class for each request batch."""
    # The sequence group metadata list.
    seq_group_metadata_list: List[SequenceGroupMetadata]
    # Blocks to swap in. List of CPU -> GPU block number.
    blocks_to_swap_in: List[Tuple[int, int]] = field(default_factory=list)
    # Blocks to swap out. List of GPU -> CPU block number.
    blocks_to_swap_out: List[Tuple[int, int]] = field(default_factory=list)
    # Blocks to copy. Source to dest block.
    blocks_to_copy: List[Tuple[int, int]] = field(default_factory=list)

    def clone(
        self, seq_group_metadata_list: List[SequenceGroupMetadata]
    ) -> "ExecuteModelInput":
        """Clone the request with a new sequence group metadata list."""
        return ExecuteModelInput(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=self.blocks_to_swap_in.copy(),
            blocks_to_swap_out=self.blocks_to_swap_out.copy(),
            blocks_to_copy=self.blocks_to_copy.copy())


@dataclass(frozen=True)
class ModelInput(ABC):
    """Local inputs to each worker's model runner. May contain
    device-specific data. Different worker backends may have different methods
    of converting from the global ExecuteModelRequest produced by the LLM
    engine to the worker-local ModelRunnerInputBase objects.

    Model runners that support multi-GPU execution should define a
    ModelRunnerInputBase subclass, add their required fields, and specify how to
    serialize/deserialize a ModelInput for broadcast between workers.
    """


@dataclass(frozen=True)
class WorkerInput:
    """Local inputs to each worker. May contain device-specific data. These
    fields should be broadcastable to other workers.
    """

    num_seq_groups: Optional[int] = None
    blocks_to_swap_in: Optional[torch.Tensor] = None
    blocks_to_swap_out: Optional[torch.Tensor] = None
    blocks_to_copy: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class ExecuteInput(ABC):
    worker_input: Optional[WorkerInput]
    model_input: ModelInput


class ExecuteOutput(ABC):
    pass
