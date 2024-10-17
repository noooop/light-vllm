import atexit
from queue import Queue
from threading import Thread
from typing import List, Optional

from light_vllm.backends.attention import AttentionBackend
from light_vllm.core.config import EngineConfig
from light_vllm.core.llm_engine import LLMEngine
from light_vllm.core.worker import create_worker
from light_vllm.core.workflow import Workflow
from light_vllm.logger import init_logger
from light_vllm.prefill_only.executor.gpu_executor import Executor

logger = init_logger(__name__)


class GPUDataParallelismExecutor:
    support_scheduling = ["async_scheduling"]

    def __init__(self, engine_config: EngineConfig, workflow: Workflow,
                 attn_backend: AttentionBackend, executor_in: Queue,
                 executor_out: Queue) -> None:
        self.engine_config = engine_config
        self.workflow = workflow
        self.attn_backend = attn_backend
        self.output_to_cpu = False

        self.executor_in = executor_in
        self.executor_out = executor_out

        self.threads: Optional[List[Thread]] = None

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls(engine_config=engine.engine_config,
                   workflow=engine.workflow,
                   attn_backend=engine.attn_backend,
                   executor_in=engine.executor_in,
                   executor_out=engine.executor_out)

    def thread_target(self, rank: int):
        # Is there a better way to avoid loading the model multiple times?
        # Load to cpu first?
        worker_kwargs = dict(engine_config=self.engine_config,
                             attn_backend=self.attn_backend,
                             envs={'CUDA_VISIBLE_DEVICES': str(rank)})
        worker_kwargs.update(module=self.workflow.Worker)
        worker = create_worker(**worker_kwargs)
        worker.init_device()
        worker.load_model()
        executor = Executor(worker)

        if self.engine_config.scheduler_config.scheduling == "double_buffer":
            execute_loop = executor.double_buffer_execute_loop
        elif self.engine_config.scheduler_config.scheduling == "simple_async":
            execute_loop = executor.simple_async_execute_loop
        else:
            execute_loop = executor.async_execute_loop

        execute_loop(self.executor_in, self.executor_out)

    def ensure_start_execute_loop(self):
        if self.threads is None:
            self.threads = []
            for rank in range(
                    self.engine_config.parallel_config.data_parallel_size):
                thread = Thread(target=self.thread_target,
                                args=(rank, ),
                                daemon=True)
                thread.start()
                self.threads.append(thread)
            atexit.register(self.shutdown_execute_loop)

    def shutdown_execute_loop(self):
        if self.threads is not None:
            for thread in self.threads:
                self.executor_in.put(None)
            for thread in self.threads:
                thread.join()
            self.threads = None
            atexit.unregister(self.shutdown_execute_loop)
