

from abc import ABC, abstractmethod
from typing import Deque, Union, Iterable, List
from collections import deque
from light_vllm.engine.llm_engine import LLMEngine
from light_vllm.wde.core.config import SchedulerConfig
from light_vllm.wde.core.schema.engine_io import Request, SchedulableRequest, SchedulerOutput, RequestOutput
from light_vllm.wde.core.processor.input_processor import RequestProcessor

from light_vllm.logger import init_logger
logger = init_logger(__name__)


class Scheduler(ABC):
    def __init__(
            self,
            scheduler_config: SchedulerConfig,
            request_processor: RequestProcessor,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.request_processor = request_processor

        self.waiting: Deque[Union[Request, SchedulableRequest]] = deque()

        self.requests = set()
        self.aborted_requests = set()

    @classmethod
    def from_engine(cls, engine: LLMEngine) -> "Scheduler":
        raise NotImplementedError

    def add_request(self, request: Union[Request, SchedulableRequest]) -> None:
        if request.request_id in self.requests or  request.request_id in self.aborted_requests:
            logger.warning("[%s] request_id conflict")
            return

        self.waiting.append(request)
        self.requests.add(request.request_id)

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id,)
        request_ids = set(request_id)

        self.requests -= request_ids
        self.aborted_requests += request_ids

    def remove_abort_request(self, request_outputs: List[RequestOutput]) -> List[RequestOutput]:
        out = []
        for request in request_outputs:
            if request.request_id in self.aborted_requests:
                self.aborted_requests.remove(request.request_id)
            else:
                out.append(request)
        return out

    def has_unfinished_requests(self) -> bool:
        return len(self.requests) != 0

    def get_num_unfinished_requests(self) -> int:
        return len(self.requests)

    @abstractmethod
    def schedule(self) -> SchedulerOutput:
        raise NotImplementedError

    @abstractmethod
    def free_finished_request(self):
        raise NotImplementedError