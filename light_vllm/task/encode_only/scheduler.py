import enum
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple, Union

from light_vllm.task.encode_only.config import EncodeOnlySchedulerConfig
from light_vllm.logger import init_logger
from light_vllm.task.encode_only.schema.inputs import EncodeOnlyInput, EncodeOnlyRequest
from light_vllm.task.encode_only.processor.input_processor import EncodeOnlyModelRequestProcessor
logger = init_logger(__name__)


@dataclass
class SchedulingBudget:
    token_budget: int
    max_num_requests: int
    _curr_requests: Set[str] = field(default_factory=set)
    _num_batched_tokens: int = 0

    def can_schedule(self, *, num_new_tokens: int, num_new_request: int = 1):
        assert num_new_tokens != 0
        assert num_new_request != 0
        return (self.num_batched_tokens + num_new_tokens <= self.token_budget
                and self.num_curr_request + num_new_request <= self.max_num_requests)

    def remaining_token_budget(self):
        return self.token_budget - self.num_batched_tokens

    def add_num_batched_tokens(self, req_id: str, num_batched_tokens: int):
        if req_id in self._curr_requests:
            return

        self._curr_requests.add(req_id)
        self._num_batched_tokens += num_batched_tokens

    def subtract_num_batched_tokens(self, req_id: str):
        if req_id in self._curr_requests:
            self._curr_requests.remove(req_id)

    @property
    def num_batched_tokens(self):
        return self._num_batched_tokens

    @property
    def num_curr_request(self):
        return len(self._curr_requests)


@dataclass
class SchedulerOutputs:
    scheduled_requests: Iterable[EncodeOnlyInput]

    def is_empty(self) -> bool:
        return not self.scheduled_requests


class EncodeOnlyScheduler:

    def __init__(
            self,
            scheduler_config: EncodeOnlySchedulerConfig,
            request_processor: EncodeOnlyModelRequestProcessor,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.request_processor = request_processor

        # Sequence groups in the WAITING state.
        # Contain new prefill or preempted requests.
        self.waiting: Deque[EncodeOnlyRequest] = deque()
        # Sequence groups in the RUNNING state.
        # Contain decode requests.
        self.running: Dict[str: EncodeOnlyRequest] = dict()

        # Time at previous scheduling step
        self.prev_time = 0.0
        # Did we schedule a prompt at previous step?
        self.prev_prompt = False
        # Latency of the last prompt step
        self.last_prompt_latency = 0.0

    @classmethod
    def from_engine(cls, engine):
        return cls(engine.engine_config.scheduler_config,
                   engine.request_processor)

    @property
    def num_decoding_tokens_per_seq(self) -> int:
        """The number of new tokens."""
        return 1

    def add_request(self, request: EncodeOnlyRequest) -> None:
        # Add request to the waiting queue.
        self.waiting.append(request)

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id,)
        request_ids = set(request_id)

        # abort from waiting
        aborted_requests: List[EncodeOnlyRequest] = []
        for request in self.waiting:
            if not request_ids:
                break

            if request.request_id in request_ids:
                request_ids.remove(request.request_id)
                aborted_requests.append(request)
        for request in aborted_requests:
            self.waiting.remove(request)

        # abort from running
        for request_id in request_ids:
            self.running.pop(request_id)

    def has_unfinished_requests(self) -> bool:
        return len(self.waiting) != 0

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def schedule(self):
        now = time.time()

        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_requests=self.scheduler_config.max_num_requests,
        )

        waiting_queue = self.waiting

        scheduler_outputs = []
        while waiting_queue:
            request = waiting_queue[0]
            request = self.request_processor(request)

            num_new_tokens = request.num_new_tokens

            if not budget.can_schedule(num_new_tokens=num_new_tokens):
                break

            budget.add_num_batched_tokens(request.request_id, num_new_tokens)

            waiting_queue.popleft()
            scheduler_outputs.append(request)
            self.running[request.request_id] = request

        return SchedulerOutputs(scheduled_requests=scheduler_outputs)

    def free_finished_request(self):
        return