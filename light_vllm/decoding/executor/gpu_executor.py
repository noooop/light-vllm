from typing import Optional, Tuple

import torch

from light_vllm.core.config import EngineConfig
from light_vllm.core.llm_engine import LLMEngine
from light_vllm.core.schema.execute_io import ExecuteInput, ExecuteOutput
from light_vllm.core.worker import WorkerWrapperBase
from light_vllm.core.workflow import Workflow
from light_vllm.decoding.backends.attention import DecodeOnlyAttentionBackend
from light_vllm.logger import init_logger

logger = init_logger(__name__)


def create_worker(module, **kwargs):
    module_name, class_name = module.split(":")
    wrapper = WorkerWrapperBase(
        worker_module_name=module_name,
        worker_class_name=class_name,
    )
    wrapper.init_worker(**kwargs)
    return wrapper.worker


class GPUExecutor:
    support_scheduling = ["sync_scheduling"]

    def __init__(
        self,
        engine_config: EngineConfig,
        workflow: Workflow,
        attn_backend: DecodeOnlyAttentionBackend,
    ) -> None:
        self.engine_config = engine_config
        self.workflow = workflow
        self.attn_backend = attn_backend
        self._init_executor()

        self.h2d_stream = torch.cuda.Stream()
        self.d2h_stream = torch.cuda.Stream()

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls(engine_config=engine.engine_config,
                   workflow=engine.workflow,
                   attn_backend=engine.attn_backend)

    def _init_executor(self) -> None:
        """Initialize the worker and load the model.
        """

        worker_kwargs = dict(
            engine_config=self.engine_config,
            attn_backend=self.attn_backend,
        )
        worker_kwargs.update(module=self.workflow.Worker)

        self.worker = create_worker(**worker_kwargs)
        self.worker.init_device()
        self.worker.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks by invoking the
        underlying worker.
        """
        return self.worker.determine_num_available_blocks()

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks) -> None:
        """Initialize the KV cache by invoking the underlying worker.
        """
        # NOTE: This is logged in the executor because there can be >1 worker
        # with other executors. We could log in the engine level, but work
        # remains to abstract away the device for non-GPU configurations.
        logger.info("# GPU blocks: %d, # CPU blocks: %d", num_gpu_blocks,
                    num_cpu_blocks)

        self.worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    def execute_model(self,
                      execute_input: ExecuteInput) -> Optional[ExecuteOutput]:

        with torch.cuda.stream(self.h2d_stream):
            self.worker.non_blocking_h2d(execute_input)

        self.h2d_stream.synchronize()

        execute_output = self.worker(execute_input)

        with torch.cuda.stream(self.d2h_stream):
            self.worker.non_blocking_d2h(execute_output)

        self.d2h_stream.synchronize()

        return execute_output

    def initialize_kv_caches(self, engine: LLMEngine) -> None:
        """Initialize the KV cache in the worker(s).

        The workers will determine the number of blocks in both the GPU cache
        and the swap CPU cache.
        """
        self.worker.model_runner.prepare_model_input = engine.model_inputs_builder.prepare_model_input

        num_gpu_blocks, num_cpu_blocks = (
            self.determine_num_available_blocks())

        if self.engine_config.cache_config.num_gpu_blocks_override is not None:
            num_gpu_blocks_override = self.engine_config.cache_config.num_gpu_blocks_override
            logger.info(
                "Overriding num_gpu_blocks=%d with "
                "num_gpu_blocks_override=%d", num_gpu_blocks,
                num_gpu_blocks_override)
            num_gpu_blocks = num_gpu_blocks_override

        self.engine_config.cache_config.num_gpu_blocks = num_gpu_blocks
        self.engine_config.cache_config.num_cpu_blocks = num_cpu_blocks

        self.initialize_cache(num_gpu_blocks, num_cpu_blocks)

        del self.worker.model_runner.prepare_model_input

    def shutdown_execute_loop(self):
        pass
