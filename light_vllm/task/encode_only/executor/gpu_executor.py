from typing import Any, Dict, List, Optional, Set, Tuple, Union

from light_vllm.task.base.executor.executor_base import ExecutorBase
from light_vllm.logger import init_logger

from light_vllm.task.base.schema.execute_io import ExecuteInput, ExecuteOutput
from light_vllm.task.base.worker.worker_base import WorkerWrapperBase

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
        pass

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks) -> None:
       pass

    def execute_model(self, execute_input: ExecuteInput
    ) -> Optional[List[ExecuteOutput]]:
        output = self.driver_worker(execute_input)
        return output