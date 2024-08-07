from typing import Any, Dict, List, Optional, Set, Tuple, Union

from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.logger import init_logger
from vllm.sequence import ExecuteModelRequest, PoolerOutput, SamplerOutput
from vllm.utils import make_async
from vllm.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


def create_worker(worker_module_name, worker_class_name, **kwargs):
    wrapper = WorkerWrapperBase(
        worker_module_name=worker_module_name,
        worker_class_name=worker_class_name,
    )
    wrapper.init_worker(**kwargs)
    return wrapper.worker


class GPUExecutor(ExecutorBase):

    uses_ray: bool = False

    def _init_executor(self) -> None:
        """Initialize the worker and load the model.
        """
        self.driver_worker = self._create_worker()
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def _get_worker_kwargs(
            self) -> Dict[str, Any]:
        """Return worker init args for a given rank."""
        return dict(
            model_config=self.model_config,
            scheduler_config=self.scheduler_config,
            device_config=self.device_config,
            cache_config=self.cache_config,
            load_config=self.load_config,
            is_driver_worker=True,
        )

    def _get_create_worker_kwargs(self) -> Dict:
        worker_kwargs = self._get_worker_kwargs()

        worker_kwargs.update(worker_module_name="vllm.worker.worker",
                             worker_class_name="Worker")

        return worker_kwargs

    def _create_worker(self):
        return create_worker(**self._get_create_worker_kwargs())

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
        self, execute_model_req: ExecuteModelRequest
    ) -> Optional[List[Union[SamplerOutput, PoolerOutput]]]:
        output = self.driver_worker.execute_model(execute_model_req)
        return output

    def check_health(self) -> None:
        # GPUExecutor will always be healthy as long as
        # it's running.
        return


class GPUExecutorAsync(GPUExecutor, ExecutorAsyncBase):

    async def execute_model_async(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> List[Union[SamplerOutput, PoolerOutput]]:
        output = await make_async(self.driver_worker.execute_model
                                  )(execute_model_req=execute_model_req, )
        return output
