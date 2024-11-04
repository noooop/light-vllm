from light_vllm.core.workflow import Workflow


class DecodeDecodingOnlyWorkflow(Workflow):
    EngineArgs: str = "light_vllm.decoding.arg_utils:ChatEngineArgs"
    InputProcessor: str = ("light_vllm.core.processor."
                           "input_processor:TextInputProcessor")
    RequestProcessor: str = ("light_vllm.decoding.processor.input_processor:"
                             "ChatModelRequestProcessor")
    OutputProcessor: str = ("light_vllm.decoding.processor.output_processor:"
                            "ChatModelOutputProcessor")
    ModelInputBuilder: str = (
        "light_vllm.decoding.processor.model_input_builder:"
        "ChatModelPreProcessor")
    Worker: str = "light_vllm.decoding.worker.gpu_worker:Worker"
    Executor: str = "light_vllm.decoding.executor.gpu_executor"
    Scheduler: str = "light_vllm.decoding.scheduler:DecodingScheduler"
    AttnBackend: str = ("light_vllm.decoding.backends.attention.selector:"
                        "AttnBackend")
    attn_type: str = "DECODER"

    @classmethod
    def from_engine(cls, engine):
        workflow = cls()

        if engine.engine_config.scheduler_config.scheduling in ["sync"]:
            workflow.Executor += ":GPUExecutor"
        elif engine.engine_config.scheduler_config.scheduling in [
                "simple_async", "async", "double_buffer"
        ]:
            workflow.Executor += ":GPUAsyncExecutor"

        return workflow
