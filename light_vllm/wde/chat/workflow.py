from light_vllm.wde.core.workflow import Workflow


class ChatWorkflow(Workflow):
    EngineArgs: str = "light_vllm.wde.chat.arg_utils:ChatEngineArgs"
    InputProcessor: str = ("light_vllm.wde.chat.processor.input_processor:"
                           "ChatModelInputProcessor")
    RequestProcessor: str = ("light_vllm.wde.chat.processor.input_processor:"
                             "ChatModelRequestProcessor")
    OutputProcessor: str = ("light_vllm.wde.chat.processor.output_processor:"
                            "ChatModelOutputProcessor")
    ModelInputBuilder: str = (
        "light_vllm.wde.chat.processor.model_input_builder:"
        "ChatModelPreProcessor")
    Worker: str = "light_vllm.wde.decode_only.worker.gpu_worker:Worker"
    Executor: str = ("light_vllm.wde.decode_only.executor.gpu_executor:"
                     "GPUExecutor")
    Scheduler: str = "light_vllm.wde.decode_only.scheduler:Scheduler"
    AttnBackend: str = ("light_vllm.wde.decode_only.layers.attention.selector:"
                        "AttnBackend")
    attn_type: str = "DECODER"
