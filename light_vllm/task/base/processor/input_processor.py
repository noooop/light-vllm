
from abc import ABC, abstractmethod
from typing import Optional
from light_vllm.task.base.schema.inputs import Request
from light_vllm.task.base.schema.sequence import SequenceGroup


class InputProcessor(ABC):
    """
    Input(request_id, prompt, params, arrival_time) -> InputProcessor -> Request
    """

    @abstractmethod
    def __call__(self,
                 request_id: str,
                 prompt,
                 params,
                 arrival_time: Optional[float] = None) -> Request:
        raise NotImplementedError


class RequestProcessor(ABC):
    """
    Request -> RequestProcessor -> SequenceGroup
    """
    @abstractmethod
    def __call__(self, request: Request) -> SequenceGroup:
        raise NotImplementedError
