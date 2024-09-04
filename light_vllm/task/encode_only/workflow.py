

from light_vllm.task.base.workflow import Workflow


class EncodeOnlyWorkflow(Workflow):
    EngineArgs: str = "light_vllm.task.encode_only.arg_utils:EngineArgs"
    InputProcessor: str = "light_vllm.task.encode_only.processor.input_processor:EncodeOnlyModelInputProcessor"
    RequestProcessor: str = "light_vllm.task.encode_only.processor.input_processor:EncodeOnlyModelRequestProcessor"
    OutputProcessor: str = "light_vllm.task.encode_only.processor.output_processor:EncodeOnlyModelOutputProcessor"
    ModelPreProcessor: str = "light_vllm.task.encode_only.processor.model_pre_processor:EncodeOnlyModelPreProcessor"
    Worker: str = "light_vllm.task.encode_only.worker.gpu_worker:Worker"
    Executor: str = "light_vllm.task.encode_only.executor.gpu_executor:GPUExecutor"
    Scheduler: str = "light_vllm.task.encode_only.scheduler:EncodeOnlyScheduler"
