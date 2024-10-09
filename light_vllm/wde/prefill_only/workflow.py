from light_vllm.wde.core.workflow import Workflow


class PrefillOnlyWorkflow(Workflow):

    InputProcessor: str = ("light_vllm.wde.prefill_only.processor."
                           "input_processor:PrefillOnlyModelInputProcessor")
    RequestProcessor: str = (
        "light_vllm.wde.prefill_only.processor."
        "input_processor:PrefillOnlyModelRequestProcessor")
    OutputProcessor: str = ("light_vllm.wde.prefill_only.processor."
                            "output_processor:PrefillOnlyModelOutputProcessor")
    ModelInputBuilder: str = (
        "light_vllm.wde.prefill_only.processor."
        "model_input_builder:PrefillOnlyModelInputBuilder")
    Worker: str = "light_vllm.wde.prefill_only.worker.gpu_worker:Worker"
    Executor: str = "light_vllm.wde.prefill_only.executor.gpu_executor"
    Scheduler: str = "light_vllm.wde.prefill_only.scheduler:PrefillOnlyScheduler"
    AttnBackend: str = ("light_vllm.wde.prefill_only.layers."
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
                workflow.Executor += ":GPUAsyncExecutor"
        else:
            assert engine.engine_config.parallel_config.data_parallel_size > 0
            assert engine.engine_config.scheduler_config.scheduling in [
                "async", "double_buffer"
            ]

            engine.engine_config.scheduler_config.max_num_on_the_fly *= (
                engine.engine_config.parallel_config.data_parallel_size)

            workflow.Executor = (
                "light_vllm.wde.prefill_only.executor.gpu_data_parallelism_executor:"
                "GPUDataParallelismExecutor")

        return workflow
