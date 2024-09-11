from typing import Optional

from light_vllm.wde.core.config import EngineConfig
from light_vllm.wde.core.workflow import Workflow
from light_vllm.logger import init_logger
from queue import Queue
from threading import Thread
from light_vllm.wde.core.llm_engine import LLMEngine
from light_vllm.wde.core.schema.execute_io import ExecuteInput, ExecuteOutput
from light_vllm.wde.core.worker.worker_base import WorkerWrapperBase
from light_vllm.wde.encode_only.layers.attention.backends.abstract import EncodeOnlyAttentionBackend

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
    def __init__(
        self,
        engine_config: EngineConfig,
        workflow: Workflow,
        attn_backend: EncodeOnlyAttentionBackend,
    ) -> None:
        self.engine_config = engine_config
        self.workflow = workflow
        self.attn_backend = attn_backend
        self._init_executor()

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls(
            engine_config=engine.engine_config,
            workflow=engine.workflow,
            attn_backend=engine.attn_backend
        )

    def _init_executor(self) -> None:
        """Initialize the worker and load the model.
        """

        worker_kwargs = dict(
            engine_config=self.engine_config,
            attn_backend=self.attn_backend,
        )
        worker_kwargs.update(module=self.workflow.Worker)

        self.driver_worker = create_worker(**worker_kwargs)
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def execute_model(self, execute_input: ExecuteInput
    ) -> Optional[ExecuteOutput]:
        output = self.driver_worker(execute_input)
        return output


class GPUAsyncExecutor(GPUExecutor):
    def __init__(
        self,
        engine_config: EngineConfig,
        workflow: Workflow,
        attn_backend: EncodeOnlyAttentionBackend,
        executor_in: Queue,
        executor_out: Queue
    ) -> None:
        super().__init__(engine_config, workflow, attn_backend)
        self.executor_in = executor_in
        self.executor_out = executor_out

        self.executor_thread = None

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls(
            engine_config=engine.engine_config,
            workflow=engine.workflow,
            attn_backend=engine.attn_backend,
            executor_in=engine.executor_in,
            executor_out=engine.executor_out
        )

    def execute_loop(self):
        while True:
            o = self.executor_in.get()
            if o is None:
                break

            scheduler_output, executor_input = o
            executor_output = self.execute_model(executor_input)
            self.executor_out.put((scheduler_output, executor_output))

    def start_execute_loop(self):
        if self.executor_thread is None or not self.executor_thread.is_alive():
            self.executor_thread = Thread(target=self.execute_loop)
            self.executor_thread.start()

    def shutdown_execute_loop(self):
        if self.executor_thread.is_alive():
            self.executor_in.put(None)
            self.executor_thread.join()