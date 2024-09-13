import queue
from typing import Optional
import torch
from queue import Queue
from threading import Thread
from light_vllm.wde.core.config import EngineConfig
from light_vllm.wde.core.workflow import Workflow
from light_vllm.logger import init_logger
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
    support_scheduling = ["sync_scheduling"]

    def __init__(
        self,
        engine_config: EngineConfig,
        workflow: Workflow,
        attn_backend: EncodeOnlyAttentionBackend,
    ) -> None:
        self.engine_config = engine_config
        self.workflow = workflow
        self.attn_backend = attn_backend
        self.output_to_cpu = False
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

    def execute_model(self, executor_input: ExecuteInput
    ) -> Optional[ExecuteOutput]:
        executor_input.model_input.to(self.driver_worker.device)
        output = self.driver_worker(executor_input)
        if self.output_to_cpu:
            output.to("cpu")
        return output

    def shutdown_execute_loop(self):
        pass


class GPUAsyncExecutor(GPUExecutor):
    support_scheduling = ["async_scheduling"]

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

        self.executor_thread: Optional[Thread] = None

        if self.engine_config.scheduler_config.scheduling == "double_buffer":
            self.execute_loop = self.double_buffer_execute_loop
        else:
            self.execute_loop = self.simple_execute_loop

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls(
            engine_config=engine.engine_config,
            workflow=engine.workflow,
            attn_backend=engine.attn_backend,
            executor_in=engine.executor_in,
            executor_out=engine.executor_out
        )

    def simple_execute_loop(self):
        while True:
            o = self.executor_in.get()
            if o is None:
                break

            scheduler_output, executor_input = o
            executor_output = self.execute_model(executor_input)
            if self.output_to_cpu:
                executor_output.to("cpu")
            self.executor_out.put((scheduler_output, executor_output))

    def double_buffer_execute_loop(self):
        # Looks cool
        # But offers little performance improvement and takes up twice the GPU memory
        from dataclasses import dataclass
        from light_vllm.wde.core.schema.engine_io import SchedulerOutput

        @dataclass
        class Task:
            scheduler_output: SchedulerOutput
            executor_input: ExecuteInput
            executor_output: Optional[ExecuteOutput]

            @classmethod
            def get(cls, block):
                o = self.executor_in.get(block)
                if o is None:
                    return None

                scheduler_output, executor_input = o

                task = cls(scheduler_output=scheduler_output,
                           executor_input=executor_input,
                           executor_output=None)
                return task

        current_task: Optional[Task] = None
        next_task: Optional[Task] = None
        compute_stream = torch.cuda.Stream()
        io_stream = torch.cuda.Stream()

        go_on = True
        while go_on:
            if current_task is None:
                current_task = Task.get(block=True)
                if current_task is None:
                    break

                with torch.cuda.stream(compute_stream):
                    current_task.executor_input.model_input.to(self.driver_worker.device, non_blocking=True)
                    current_task.executor_output = self.execute_model(current_task.executor_input)
                    end_compute = torch.cuda.Event()
            else:
                with torch.cuda.stream(compute_stream):
                    end_compute = torch.cuda.Event()

            try:
                next_task = Task.get(block=False)
                if next_task is None:
                    go_on = False
                else:
                    with torch.cuda.stream(io_stream):
                        next_task.executor_input.model_input.to(self.driver_worker.device, non_blocking=True)

                    compute_stream.wait_stream(io_stream)

                    with torch.cuda.stream(compute_stream):
                        next_task.executor_output = self.execute_model(next_task.executor_input)
            except queue.Empty:
                pass
                #logger.info("Executor_in Queue Empty. "
                #            "If this occurs frequently, "
                #            "setting max_num_on_the_fly higher help.")

            end_compute.wait()
            if self.output_to_cpu:
                with torch.cuda.stream(io_stream):
                    current_task.executor_output.to("cpu", non_blocking=True)
                    io_stream.synchronize()
            self.executor_out.put((current_task.scheduler_output, current_task.executor_output))

            current_task = next_task
            next_task = None

    def ensure_start_execute_loop(self):
        if self.executor_thread is None or not self.executor_thread.is_alive():
            self.executor_thread = Thread(target=self.execute_loop, daemon=True)
            self.executor_thread.start()

    def shutdown_execute_loop(self):
        if self.executor_thread.is_alive():
            self.executor_in.put(None)
            self.executor_thread.join()
