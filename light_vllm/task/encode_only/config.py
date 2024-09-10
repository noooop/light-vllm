from dataclasses import dataclass, fields
from typing import Optional

from light_vllm.logger import init_logger
from light_vllm.task.base.config import EngineConfig, ModelConfig, SchedulerConfig

logger = init_logger(__name__)

_GB = 1 << 30


class EncodeOnlyModelConfig(ModelConfig):
    pass


class EncodeOnlySchedulerConfig(SchedulerConfig):
    """Scheduler configuration.

    Args:
        max_num_batched_tokens: Maximum number of tokens to be processed in
            a single iteration.
        max_num_seqs: Maximum number of sequences to be processed in a single
            iteration.
        max_model_len: Maximum length of a sequence.
            If None, will be derived from the model.
    """

    def __init__(self,
                 max_model_len: int,
                 max_num_batched_tokens: Optional[int] = None,
                 max_num_requests: Optional[int] = None,
                 max_num_seqs: Optional[int] = None,
                 ) -> None:
        self.max_model_len = max_model_len
        self.max_num_requests: int = 0
        self.max_num_batched_tokens: int = 0

        self.set_args(max_num_batched_tokens,
                      max_num_requests,
                      max_num_seqs)

    def set_args(self,
                 max_num_batched_tokens: Optional[int] = None,
                 max_num_requests: Optional[int] = None,
                 max_num_seqs: Optional[int] = None, ):
        if max_num_seqs is not None:
            self.max_num_requests = max_num_seqs
        else:
            self.max_num_requests = max_num_requests

        if max_num_batched_tokens is not None:
            self.max_num_batched_tokens = max_num_batched_tokens
        else:
            self.max_num_batched_tokens = self.max_model_len * self.max_num_requests

        self._verify_args()

    def _verify_args(self) -> None:
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
                "be greater than or equal to max_model_len "
                f"({self.max_model_len}).")


@dataclass(frozen=True)
class EncodeOnlyEngineConfig(EngineConfig):
    """Dataclass which contains all engine-related configuration. This
    simplifies passing around the distinct configurations in the codebase.
    """

    model_config: EncodeOnlyModelConfig
    scheduler_config: EncodeOnlySchedulerConfig

    def to_dict(self):
        """Return the configs as a dictionary, for use in **kwargs.
        """
        return dict(
            (field.name, getattr(self, field.name)) for field in fields(self))
