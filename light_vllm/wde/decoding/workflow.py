from light_vllm.wde.core.workflow import Workflow


class DecodeDecodingOnlyWorkflow(Workflow):
    EngineArgs: str = "light_vllm.wde.decoding.arg_utils:ChatEngineArgs"
    InputProcessor: str = ("light_vllm.wde.decoding.processor.input_processor:"
                           "ChatModelInputProcessor")
    RequestProcessor: str = (
        "light_vllm.wde.decoding.processor.input_processor:"
        "ChatModelRequestProcessor")
    OutputProcessor: str = (
        "light_vllm.wde.decoding.processor.output_processor:"
        "ChatModelOutputProcessor")
    ModelInputBuilder: str = (
        "light_vllm.wde.decoding.processor.model_input_builder:"
        "ChatModelPreProcessor")
    Worker: str = "light_vllm.wde.decoding.worker.gpu_worker:Worker"
    Executor: str = ("light_vllm.wde.decoding.executor.gpu_executor:"
                     "GPUExecutor")
    Scheduler: str = "light_vllm.wde.decoding.scheduler:Scheduler"
    AttnBackend: str = ("light_vllm.wde.decoding.layers.attention.selector:"
                        "AttnBackend")
    attn_type: str = "DECODER"
