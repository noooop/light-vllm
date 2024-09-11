

from abc import ABC, abstractmethod

from light_vllm.engine.llm_engine import LLMEngine
from light_vllm.wde.core.schema.engine_io import SchedulerOutput
from light_vllm.wde.core.schema.execute_io import ExecuteInput


class ModelInputBuilder(ABC):
    """
    scheduler_output = scheduler.schedule()
    SchedulerOutput  -> ModelInputBuilder -> ExecuteInput
    """

    @abstractmethod
    def __call__(self, scheduler_output: SchedulerOutput) -> ExecuteInput:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_engine(cls, engine: LLMEngine):
        raise NotImplementedError

