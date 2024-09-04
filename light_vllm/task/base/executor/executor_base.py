from abc import ABC, abstractmethod
from typing import List, Optional, Set, Tuple

from light_vllm.config import (CacheConfig, DeviceConfig, LoadConfig,
                               ModelConfig, SchedulerConfig)
from light_vllm.task.base.schema.execute_io import ExecuteInput, ExecuteOutput


class ExecutorBase(ABC):
    """Base class for all executors.

    An executor is responsible for executing the model on a specific device type.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        load_config: LoadConfig,
        workflow
    ) -> None:
        self.model_config = model_config
        self.cache_config = cache_config
        self.load_config = load_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.workflow = workflow
        self._init_executor()

    @classmethod
    def from_engine(cls, engine):
        return cls(
            model_config=engine.engine_config.model_config,
            cache_config=engine.engine_config.cache_config,
            scheduler_config=engine.engine_config.scheduler_config,
            device_config=engine.engine_config.device_config,
            load_config=engine.engine_config.load_config,
            workflow=engine.workflow
        )

    @abstractmethod
    def _init_executor(self) -> None:
        pass

    @abstractmethod
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available blocks for the GPU KV cache and
        swappable CPU KV cache.

        Normally, this should simply delegate to the underlying Worker. Some
        ExecutorBase may require modification of the result, e.g. to ensure the
        selected cache sizes are compatible with all workers.

        Returns a Tuple[num_gpu_blocks, num_cpu_blocks], where num_gpu_blocks
        are blocks that are "active" on the device and can be appended to.
        num_cpu_blocks refers to "swapped" blocks in CPU memory and cannot be
        appended to.
        """
        raise NotImplementedError

    @abstractmethod
    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache with the given size in blocks.
        """
        raise NotImplementedError

    @abstractmethod
    def execute_model(self, execute_input: ExecuteInput
    ) -> Optional[List[ExecuteOutput]]:
        """Executes at least one model step on the given sequences."""
        raise NotImplementedError

    def shutdown(self) -> None:
        """Shutdown the executor."""
        return

    def __del__(self):
        self.shutdown()
