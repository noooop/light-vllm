import enum
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple, Union

from light_vllm.task.encode_only.config import EncodeOnlySchedulerConfig
from light_vllm.logger import init_logger
from light_vllm.task.encode_only.schema.engine_io import EncodeOnlyRequest
from light_vllm.task.encode_only.processor.input_processor import EncodeOnlyModelRequestProcessor

from light_vllm.task.base.schema.engine_io import SchedulableRequest, SchedulerOutputs
from light_vllm.task.base.scheduler import Scheduler


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
class EncodeOnlySchedulerOutputs(SchedulerOutputs):
    scheduled_requests: Iterable[EncodeOnlyRequest]

    def is_empty(self) -> bool:
        return not self.scheduled_requests


class EncodeOnlyScheduler(Scheduler):

    def __init__(
            self,
            scheduler_config: EncodeOnlySchedulerConfig,
            request_processor: EncodeOnlyModelRequestProcessor,
    ) -> None:
        super().__init__(scheduler_config, request_processor)
        self.scheduler_config = scheduler_config
        self.request_processor = request_processor

    @classmethod
    def from_engine(cls, engine):
        return cls(engine.engine_config.scheduler_config,
                   engine.request_processor)

    def schedule(self):
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_requests=self.scheduler_config.max_num_requests,
        )

        waiting_queue = self.waiting

        scheduler_outputs = []
        while waiting_queue:
            request = waiting_queue[0]
            if request.request_id in self.aborted_requests:
                self.aborted_requests.remove(request.request_id)
                waiting_queue.popleft()
                continue

            if not isinstance(request, SchedulableRequest):
                request = self.request_processor(request)

            num_new_tokens = request.num_new_tokens

            if not budget.can_schedule(num_new_tokens=num_new_tokens):
                break

            budget.add_num_batched_tokens(request.request_id, num_new_tokens)

            waiting_queue.popleft()
            scheduler_outputs.append(request)
            self.requests.remove(request.request_id)

        return EncodeOnlySchedulerOutputs(scheduled_requests=scheduler_outputs)

    def free_finished_request(self):
        return
