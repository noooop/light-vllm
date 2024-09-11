

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
    EngineArgs: str
    Scheduler: str
    GetAttnBackend: str
    Tokenizer: str = "light_vllm.inputs.tokenizer:Tokenizer"
    InputProcessor: str
    RequestProcessor: str
    OutputProcessor: str
    ModelInputBuilder: str
    Executor: str
    Worker: str



