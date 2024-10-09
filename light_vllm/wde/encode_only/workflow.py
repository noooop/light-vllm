from light_vllm.wde.core.workflow import Workflow


class EncodeOnlyWorkflow(Workflow):
    EngineArgs: str = ("light_vllm.wde.encode_only.arg_utils:"
                       "EncodeOnlyEngineArgs")
    InputProcessor: str = ("light_vllm.wde.encode_only.processor."
                           "input_processor:EncodeOnlyModelInputProcessor")
    RequestProcessor: str = ("light_vllm.wde.encode_only.processor."
                             "input_processor:EncodeOnlyModelRequestProcessor")
    OutputProcessor: str = ("light_vllm.wde.encode_only.processor."
                            "output_processor:EncodeOnlyModelOutputProcessor")
    ModelInputBuilder: str = (
        "light_vllm.wde.encode_only.processor."
        "model_input_builder:EncodeOnlyModelInputBuilder")
    Worker: str = "light_vllm.wde.encode_only.worker.gpu_worker:Worker"
    Executor: str = "light_vllm.wde.encode_only.executor.gpu_executor"
    Scheduler: str = "light_vllm.wde.encode_only.scheduler:EncodeOnlyScheduler"
    GetAttnBackend: str = ("light_vllm.wde.encode_only.layers."
                           "attention.selector:GetAttnBackend")

    @classmethod
    def from_engine(cls, engine):

        workflow = cls()
        if engine.engine_config.scheduler_config.scheduling in ["sync"]:
            workflow.Executor += ":GPUExecutor"
        elif engine.engine_config.scheduler_config.scheduling in [
                "async", "double_buffer"
        ]:
            workflow.Executor += ":GPUAsyncExecutor"

        return workflow
