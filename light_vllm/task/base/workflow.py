

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

