

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List

import torch

from light_vllm.core.scheduler import SchedulerOutputs
from light_vllm.task.base.schema.sequence import SequenceGroupMetadata
from light_vllm.task.base.schema.execute_io import ExecuteInput, ExecuteModelInput, WorkerInput, ModelInput


class ModelPreProcessor(ABC):
    """
    seq_group_metadata_list, scheduler_outputs = scheduler.schedule()
    (seq_group_metadata_list: List[SequenceGroupMetadata], scheduler_outputs: SchedulerOutputs) -> ModelPreProcessor -> ExecuteInput
    """

    @classmethod
    @abstractmethod
    def from_engine(cls, engine):
        raise NotImplementedError


    @torch.inference_mode()
    def prepare_model_input(self, seq_group_metadata_list: List[SequenceGroupMetadata]) -> ModelInput:
        raise NotImplementedError

    @torch.inference_mode()
    def prepare_worker_input(self, execute_model_input: ExecuteModelInput) -> WorkerInput:
        raise NotImplementedError

    def __call__(self, scheduler_outputs: SchedulerOutputs) -> ExecuteInput:
        execute_model_input = ExecuteModelInput(
            seq_group_metadata_list=scheduler_outputs.seq_group_metadata_list,
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy)

        worker_input: WorkerInput = self.prepare_worker_input(execute_model_input)
        model_input: ModelInput = self.prepare_model_input(execute_model_input.seq_group_metadata_list)

        return ExecuteInput(worker_input=worker_input, model_input=model_input)


