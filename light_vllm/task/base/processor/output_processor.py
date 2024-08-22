
from abc import ABC, abstractmethod

from light_vllm.task.base.schema.execute_io import ExecuteOutput


class OutputProcessor(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> ExecuteOutput:
        raise NotImplementedError
