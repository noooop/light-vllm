from typing import Optional

from light_vllm.task.base.config import EngineConfig
from light_vllm.task.base.workflow import Workflow
from light_vllm.logger import init_logger

from light_vllm.engine.llm_engine import LLMEngine
from light_vllm.task.base.schema.execute_io import ExecuteInput, ExecuteOutput
from light_vllm.task.base.worker.worker_base import WorkerWrapperBase
from light_vllm.task.encode_only.layers.attention.backends.abstract import EncodeOnlyAttentionBackend

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