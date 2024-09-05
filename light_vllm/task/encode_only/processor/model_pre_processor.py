

from dataclasses import dataclass
from typing import List, Tuple

import torch


from light_vllm.task.base.schema.execute_io import ExecuteModelInput, WorkerInput, ModelInput, ExecuteInput
from light_vllm.task.base.processor.model_pre_processor import ModelPreProcessor
from light_vllm.task.encode_only.scheduler import SchedulerOutputs


@dataclass(frozen=True)
class ModelInputForGPU(ModelInput):
    batch_data: object


class EncodeOnlyModelPreProcessor(ModelPreProcessor):
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_engine(cls, engine):
        return cls(engine.tokenizer)

    def __call__(self,  scheduler_outputs: SchedulerOutputs) -> ExecuteInput:
        sentences_batch = [r.input.prompt for r in scheduler_outputs.scheduled_requests]

        batch_data = self.tokenizer(
            sentences_batch,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )

        return ExecuteInput(worker_input=None, model_input=ModelInputForGPU(batch_data=batch_data))

