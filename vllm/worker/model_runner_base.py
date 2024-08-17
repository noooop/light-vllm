import dataclasses
from abc import ABC, abstractmethod
from typing import (Dict, Generic, List, Optional, TypeVar)

import torch

from vllm.platforms import current_platform
from vllm.sequence import SamplerOutput, SequenceGroupMetadata

T = TypeVar('T', bound="ModelRunnerInputBase")


@dataclasses.dataclass(frozen=True)
class ModelRunnerInputBase(ABC):
    """Local inputs to each worker's model runner. May contain
    device-specific data. Different worker backends may have different methods
    of converting from the global ExecuteModelRequest produced by the LLM
    engine to the worker-local ModelRunnerInputBase objects.

    Model runners that support multi-GPU execution should define a
    ModelRunnerInputBase subclass, add their required fields, and specify how to
    serialize/deserialize a ModelInput for broadcast between workers.
    """


class ModelRunnerInputBuilderBase(ABC, Generic[T]):
    """A builder to create ModelRunnerInputBase objects.
  """

    @abstractmethod
    def add_seq_group(self, seq_group_metadata):
        """TBA"""
        raise NotImplementedError

    @abstractmethod
    def build(self, *args, **kwargs) -> T:
        """Build metadata with on-device tensors."""
        raise NotImplementedError


class ModelRunnerBase(ABC, Generic[T]):
    """
    Model runner interface that abstracts a particular hardware and/or type of
    model. Model execution may communicate data with model runners in other
    processes, but it should not include control plane metadata communication.

    Each ModelRunnerBase subclass should define a corresponding
    ModelRunnerInputBase subclass.
    """

    # Map of request_id -> generator used for seeded random sampling
    generators: Dict[str, torch.Generator] = {}

    @abstractmethod
    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        finished_requests_ids: Optional[List[str]] = None,
    ) -> T:
        """
        Prepare the inputs to ModelRunnerBase.execute_model from an execution
        request. This method may move data to the worker's local device. It is
        not allowed to communicate with other workers or devices.
        """
        raise NotImplementedError

    @current_platform.inference_mode()
    def execute_model(
        self,
        model_input: T,
        kv_caches: Optional[List[torch.Tensor]],
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        """
        Execute the model on the given input.
        """
        raise NotImplementedError

    def get_generators(self, finished_request_ids: Optional[List[str]] = None):
        """
        Return dict of per-request generators used for random sampling.
        """

        # Clean up generators from completed requests
        if finished_request_ids:
            for request_id in finished_request_ids:
                self.generators.pop(request_id, None)

        return self.generators
