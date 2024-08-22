

"""
Workflow:

AnyInput(*args, **kwargs) -> InputProcessor -> Request
scheduler.add_request(request:Request)

engine.step
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
    Worker: str = "light_vllm.task.base.worker.gpu_worker:Worker"

