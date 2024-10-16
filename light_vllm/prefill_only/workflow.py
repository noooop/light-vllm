from light_vllm.core.workflow import Workflow


class PrefillOnlyWorkflow(Workflow):

    InputProcessor: str = ("light_vllm.core.processor."
                           "input_processor:TextInputProcessor")
    RequestProcessor: str = ("light_vllm.core.processor."
                             "input_processor:TextRequestProcessor")
    ModelInputBuilder: str = (
        "light_vllm.prefill_only.processor."
        "model_input_builder:PrefillOnlyModelInputBuilder")
    Worker: str = "light_vllm.prefill_only.worker.gpu_worker:Worker"
    Executor: str = "light_vllm.prefill_only.executor.gpu_executor"
    Scheduler: str = ("light_vllm.prefill_only.scheduler:"
                      "PrefillOnlyScheduler")
    AttnBackend: str = ("light_vllm.prefill_only.backends."
                        "attention.selector:AttnBackend")

    @classmethod
    def from_engine(cls, engine):
        workflow = cls()

        if engine.engine_config.parallel_config is None:
            if engine.engine_config.scheduler_config.scheduling in ["sync"]:
                workflow.Executor += ":GPUExecutor"
            elif engine.engine_config.scheduler_config.scheduling in [
                    "async", "double_buffer"
            ]:
                if (engine.engine_config.scheduler_config.async_thread ==
                        "gevent"):
                    workflow.Executor += ":GPUGeventAsyncExecutor"
                else:
                    workflow.Executor += ":GPUThreadAsyncExecutor"
        else:
            assert engine.engine_config.parallel_config.data_parallel_size > 0
            assert engine.engine_config.scheduler_config.scheduling in [
                "async", "double_buffer"
            ]
            assert (
                engine.engine_config.scheduler_config.async_thread == "thread")

            engine.engine_config.scheduler_config.max_num_on_the_fly *= (
                engine.engine_config.parallel_config.data_parallel_size)

            workflow.Executor = (
                "light_vllm.prefill_only.executor.gpu_data_parallelism_executor:"
                "GPUDataParallelismExecutor")

        return workflow
