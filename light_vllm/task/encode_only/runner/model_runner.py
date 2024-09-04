import dataclasses

import warnings
import weakref
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type,
                    TypeVar, Union)
import torch
import torch.distributed
import torch.nn as nn
import light_vllm.envs as envs
from light_vllm.layers.attention import AttentionMetadata, get_attn_backend
from light_vllm.layers.sampling_metadata import SamplingMetadata
from light_vllm.layers.sampling_params import SamplingParams
from light_vllm.config import (CacheConfig, DeviceConfig, LoadConfig,
                               ModelConfig, SchedulerConfig)

from light_vllm.task.base.runner.model_runner_base import ModelRunnerBase
from light_vllm.task.base.runner.cuda_graph_util import CUDAGraph

from light_vllm.inputs import INPUT_REGISTRY
from light_vllm.logger import init_logger

from light_vllm.models.loader import get_model
from light_vllm.models.loader.tensorizer import TensorizerConfig
from light_vllm.models.utils import set_cpu_offload_max_bytes

from light_vllm.task.base.schema.sequence import SequenceGroupMetadata
from light_vllm.task.base.schema.execute_io import ModelInput, ExecuteOutput
from light_vllm.utils import (CudaMemoryProfiler, flatten_2d_lists,
                              is_hip,
                              is_pin_memory_available)
from light_vllm.task.chat.schema.execute_io import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)

TModelInputForGPU = TypeVar('TModelInputForGPU', bound="ModelInputForGPU")


class GPUModelRunnerBase(ModelRunnerBase[TModelInputForGPU]):
    """
    Helper class for shared methods between GPU model runners.
    """
    _model_input_cls: Type[TModelInputForGPU]

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        return_hidden_states: bool = False,
    ):
        self.model_config = model_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.load_config = load_config
        self.is_driver_worker = is_driver_worker
        self.return_hidden_states = return_hidden_states

        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        self.sliding_window = model_config.get_sliding_window()

        num_attn_heads = self.model_config.get_num_attention_heads()
        self.attn_backend = get_attn_backend(
            num_attn_heads,
            self.model_config.get_head_size(),
            self.model_config.get_num_kv_heads(),
            self.model_config.get_sliding_window(),
            self.model_config.dtype,
            kv_cache_dtype,
            0,
        ) if num_attn_heads else None

        self.cuda_graph = None

        # Lazy initialization
        self.model: nn.Module  # Set after load_model

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        with CudaMemoryProfiler() as m:
            self.model = get_model(model_config=self.model_config,
                                   device_config=self.device_config,
                                   load_config=self.load_config,
                                   scheduler_config=self.scheduler_config,
                                   cache_config=self.cache_config)

        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GB",
                    self.model_memory_usage / float(2**30))

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        from light_vllm.models.loader.loader import ShardedStateLoader
        ShardedStateLoader.save_model(
            self.model,
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(
        self,
        tensorizer_config: TensorizerConfig,
    ) -> None:
        from light_vllm.models.loader.loader import TensorizerLoader
        TensorizerLoader.save_model(
            self.model,
            tensorizer_config=tensorizer_config,
        )

    @torch.inference_mode()
    def profile_run(self) -> None:
        pass

    def capture_model(self, kv_caches: List[torch.Tensor]) -> None:
        pass

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()


class ModelRunner(GPUModelRunnerBase[ModelInputForGPUWithSamplingMetadata]):
    """
    GPU model runner with sampling step.
    """
    _model_input_cls: Type[ModelInputForGPUWithSamplingMetadata] = (
        ModelInputForGPUWithSamplingMetadata)

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForGPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
    ):
        batch_data = model_input.batch_data.to("cuda")
        return self.model(**batch_data)[0]

