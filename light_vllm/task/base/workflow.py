

"""
Workflow:

LLMEngine.add_request
    Input(request_id, prompt, params, arrival_time) -> InputProcessor -> Request
    scheduler.add_request(request: Request)

LLMEngine.step
    Request -> RequestProcessor -> SequenceGroup (lazy RequestProcessor)
    seq_group_metadata_list, scheduler_outputs = scheduler.schedule()

    List[SequenceGroupMetadata], SchedulerOutputs -> ModelPreProcessor -> ExecuteInput

    ExecuteInput -> Executor -> List[ExecuteOutput]

    List[ExecuteOutput] -> OutputProcessor -> RequestOutput
    RequestOutput -> return to downstream
"""


class Workflow:
    Executor: str = "light_vllm.task.base.executor.gpu_executor:GPUExecutor"
    Scheduler: str = "light_vllm.core.scheduler:Scheduler"
    Tokenizer: str = "light_vllm.inputs.tokenizer:Tokenizer"
    EngineArgs: str = "light_vllm.task.base.arg_utils:EngineArgs"
    InputProcessor: str = "light_vllm.task.base.processor.input_processor:InputProcessor"
    RequestProcessor: str = "light_vllm.task.base.processor.input_processor:RequestProcessor"
    OutputProcessor: str = "light_vllm.task.base.processor.output_processor:OutputProcessor"
    ModelPreProcessor: str = "light_vllm.base.chat.processor.model_pre_processor:PreProcessor"
    Worker: str = "light_vllm.task.base.worker.gpu_worker:Worker"
    GetAttnBackend: str


