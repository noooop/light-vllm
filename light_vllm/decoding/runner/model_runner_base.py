from abc import ABC
from typing import Generic, List, Optional, TypeVar

import torch

from light_vllm.core.schema.execute_io import ExecuteOutput
from light_vllm.platforms import current_platform

T = TypeVar('T', bound="ModelRunnerInputBase")


class ModelRunnerBase(ABC, Generic[T]):
    """
    Model runner interface that abstracts a particular hardware and/or type of
    model. Model execution may communicate data with model runners in other
    processes, but it should not include control plane metadata communication.

    Each ModelRunnerBase subclass should define a corresponding
    ModelRunnerInputBase subclass.
    """

    @current_platform.inference_mode()
    def execute_model(
        self,
        model_input: T,
        kv_caches: Optional[List[torch.Tensor]],
    ) -> Optional[List[ExecuteOutput]]:
        """
        Execute the model on the given input.
        """
        raise NotImplementedError