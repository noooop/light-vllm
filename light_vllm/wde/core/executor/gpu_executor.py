from typing import List, Optional, Tuple

from light_vllm.logger import init_logger
from light_vllm.wde.core.executor.executor_base import ExecutorBase
from light_vllm.wde.core.schema.execute_io import ExecuteInput, ExecuteOutput
from light_vllm.wde.core.worker2.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


def create_worker(module, **kwargs):
    module_name, class_name = module.split(":")
    wrapper = WorkerWrapperBase(
        worker_module_name=module_name,
        worker_class_name=class_name,
    )
    wrapper.init_worker(**kwargs)
    return wrapper.worker


class GPUExecutor(ExecutorBase):

    def _init_executor(self) -> None:
        """Initialize the worker and load the model.
        """

        worker_kwargs = dict(
            engine_config=self.engine_config,
            is_driver_worker=True,
        )
        worker_kwargs.update(module=self.workflow.Worker)

        self.driver_worker = create_worker(**worker_kwargs)
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks by invoking the
        underlying worker.
        """
        return self.driver_worker.determine_num_available_blocks()

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks) -> None:
        """Initialize the KV cache by invoking the underlying worker.
        """
        # NOTE: This is logged in the executor because there can be >1 worker
        # with other executors. We could log in the engine level, but work
        # remains to abstract away the device for non-GPU configurations.
        logger.info("# GPU blocks: %d, # CPU blocks: %d", num_gpu_blocks,
                    num_cpu_blocks)

        self.driver_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    def execute_model(
            self,
            execute_input: ExecuteInput) -> Optional[List[ExecuteOutput]]:
        output = self.driver_worker(execute_input)
        return output
