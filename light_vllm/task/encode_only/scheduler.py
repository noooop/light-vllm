import enum
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple, Union

from light_vllm.config import CacheConfig, SchedulerConfig
from light_vllm.logger import init_logger
from light_vllm.task.encode_only.schema.inputs import EncodeOnlyInput, EncodeOnlyRequest
from light_vllm.task.encode_only.processor.input_processor import EncodeOnlyModelRequestProcessor
logger = init_logger(__name__)


@dataclass
class SchedulingBudget:
    """The available slots for scheduling.

    TODO(sang): Right now, the budget is request_id-aware meaning it can ignore
    budget update from the same request_id. It is because in normal scheduling
    path, we update RUNNING num_seqs ahead of time, meaning it could be
    updated more than once when scheduling RUNNING requests. Since this won't
    happen if we only have chunked prefill scheduling, we can remove this
    feature from the API when chunked prefill is enabled by default.
    """
    token_budget: int
    max_num_seqs: int
    _request_ids_num_batched_tokens: Set[str] = field(default_factory=set)
    _request_ids_num_curr_seqs: Set[str] = field(default_factory=set)
    _num_batched_tokens: int = 0
    _num_curr_seqs: int = 0

    def can_schedule(self, *, num_new_tokens: int, num_new_seqs: int):
        assert num_new_tokens != 0
        assert num_new_seqs != 0
        return (self.num_batched_tokens + num_new_tokens <= self.token_budget
                and self.num_curr_seqs + num_new_seqs <= self.max_num_seqs)

    def remaining_token_budget(self):
        return self.token_budget - self.num_batched_tokens

    def add_num_batched_tokens(self, req_id: str, num_batched_tokens: int):
        if req_id in self._request_ids_num_batched_tokens:
            return

        self._request_ids_num_batched_tokens.add(req_id)
        self._num_batched_tokens += num_batched_tokens

    def subtract_num_batched_tokens(self, req_id: str,
                                    num_batched_tokens: int):
        if req_id in self._request_ids_num_batched_tokens:
            self._request_ids_num_batched_tokens.remove(req_id)
            self._num_batched_tokens -= num_batched_tokens

    def add_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._request_ids_num_curr_seqs:
            return

        self._request_ids_num_curr_seqs.add(req_id)
        self._num_curr_seqs += num_curr_seqs

    def subtract_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._request_ids_num_curr_seqs:
            self._request_ids_num_curr_seqs.remove(req_id)
            self._num_curr_seqs -= num_curr_seqs

    @property
    def num_batched_tokens(self):
        return self._num_batched_tokens

    @property
    def num_curr_seqs(self):
        return self._num_curr_seqs




@dataclass
class SchedulerOutputs:
    scheduled_requests: Iterable[EncodeOnlyInput]

    def is_empty(self) -> bool:
        return not self.scheduled_requests


class EncodeOnlyScheduler:

    def __init__(
            self,
            scheduler_config: SchedulerConfig,
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
        return cls(engine.scheduler_config,
                   engine.request_processor)

    @property
    def num_decoding_tokens_per_seq(self) -> int:
        """The number of new tokens."""
        return 1

    def add_request(self, request: EncodeOnlyRequest) -> None:
        # Add request to the waiting queue.
        self.waiting.append(request)

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        pass

    def has_unfinished_seqs(self) -> bool:
        return len(self.waiting) != 0

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running)

    def schedule(self):
        now = time.time()

        waiting_queue = self.waiting

        i = 0
        scheduler_outputs = []
        while waiting_queue:
            request = waiting_queue[0]

            i+=1

            if i == 32:
                break

            request = self.request_processor(request)

            waiting_queue.popleft()
            scheduler_outputs.append(request)
            self.running[request.request_id] = request

        return SchedulerOutputs(scheduled_requests=scheduler_outputs)

    def free_finished_seq_groups(self):
        pass