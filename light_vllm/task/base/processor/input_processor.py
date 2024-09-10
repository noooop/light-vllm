
from abc import ABC, abstractmethod
from typing import Optional, Union
from light_vllm.task.base.schema.engine_io import Params, Inputs, Request, SchedulableRequest


class InputProcessor(ABC):
    """
    Input(request_id, inputs, params, arrival_time) -> InputProcessor -> Request
    """

    @abstractmethod
    def __call__(self,
                 request_id: str,
                 inputs: Optional[Union[str, Inputs]] = None,
                 params: Optional[Params] = None,
                 arrival_time: Optional[float] = None) -> Request:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_engine(cls, engine):
        raise NotImplementedError


class RequestProcessor(ABC):
    """
    Request -> RequestProcessor -> SchedulableRequest
    """
    @abstractmethod
    def __call__(self, request: Request) -> SchedulableRequest:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_engine(cls, engine):
        raise NotImplementedError
