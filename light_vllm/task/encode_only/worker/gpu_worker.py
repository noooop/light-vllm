"""A GPU worker class."""
import gc
import os
from typing import List, Optional, Set, Tuple, Type

import torch
import torch.distributed

from light_vllm.config import (CacheConfig, DeviceConfig, LoadConfig,
                               ModelConfig, SchedulerConfig)
from light_vllm.layers.utils import set_random_seed
from light_vllm.models.loader.tensorizer import TensorizerConfig
from light_vllm.platforms import current_platform
from light_vllm.task.encode_only.runner.model_runner import GPUModelRunnerBase, ModelRunner
from light_vllm.task.base.worker.worker_base import LocalOrDistributedWorkerBase, WorkerInput


class Worker(LocalOrDistributedWorkerBase):
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        is_driver_worker: bool = False,
        model_runner_cls: Optional[Type[GPUModelRunnerBase]] = None,
    ) -> None:
        self.model_config = model_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.load_config = load_config

        self.is_driver_worker = is_driver_worker
        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from light_vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        ModelRunnerClass: Type[GPUModelRunnerBase] = ModelRunner
        if model_runner_cls is not None:
            ModelRunnerClass = model_runner_cls

        self.model_runner: GPUModelRunnerBase = ModelRunnerClass(
            model_config,
            scheduler_config,
            device_config,
            cache_config,
            load_config=load_config,
            is_driver_worker=is_driver_worker,
        )

    def init_device(self) -> None:
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:0")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")

        # Set random seed.
        set_random_seed(self.model_config.seed)

    @torch.inference_mode
    def load_model(self):
        self.model_runner.load_model()

    @torch.inference_mode
    def __call__(
        self,
        execute_input
    ):
        """Executes at least one model step on the given sequences, unless no
        sequences are provided."""

        self.execute_worker(execute_input.worker_input)

        output = self.model_runner.execute_model(
            execute_input.model_input,
            self.kv_cache if self.kv_cache is not None else None)

        return output

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        self.model_runner.save_sharded_state(
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(
        self,
        tensorizer_config: TensorizerConfig,
    ) -> None:
        self.model_runner.save_tensorized_model(
            tensorizer_config=tensorizer_config, )

    @torch.inference_mode()
    def execute_worker(self, worker_input: WorkerInput) -> None:
        pass
    @property
    def max_model_len(self) -> int:
        return self.model_config.max_model_len

    @property
    def vocab_size(self) -> int:
        return self.model_runner.vocab_size

    def get_cache_block_size_bytes(self) -> int:
        return 0

    def determine_num_available_blocks(self):
        pass

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        pass
    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        return None


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:
        compute_capability = current_platform.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}. "
                "You can use float16 instead by explicitly setting the"
                "`dtype` flag in CLI, for example: --dtype=half.")


def raise_if_cache_size_invalid(num_gpu_blocks, block_size,
                                max_model_len) -> None:
    if num_gpu_blocks <= 0:
        raise ValueError("No available memory for the cache blocks. "
                         "Try increasing `gpu_memory_utilization` when "
                         "initializing the engine.")
    max_seq_len = block_size * num_gpu_blocks
    if max_model_len > max_seq_len:
        raise ValueError(
            f"The model's max seq len ({max_model_len}) "
            "is larger than the maximum number of tokens that can be "
            f"stored in KV cache ({max_seq_len}). Try increasing "
            "`gpu_memory_utilization` or decreasing `max_model_len` when "
            "initializing the engine.")
