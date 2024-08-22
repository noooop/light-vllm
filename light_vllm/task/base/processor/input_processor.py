
from abc import ABC, abstractmethod
from typing import Optional
from light_vllm.task.base.schema.inputs import Request


class InputProcessor(ABC):
    """
    AnyInput(*args, **kwargs) -> InputProcessor -> Request
    """

    @abstractmethod
    def __call__(
            self,
            request_id: str,
            arrival_time: Optional[float] = None,
            *args, **kwargs
    ) -> Request:
        raise NotImplementedError
