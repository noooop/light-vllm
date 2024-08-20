

"""
chat workflow:
                tokenizer.encode
PromptInput -> ChatModelInputProcessor -> ChatInput -> +sampling_params,request_id -> ChatRequest

ChatRequest -> ChatModelSequenceProcessor -> SequenceGroup

scheduler.add_seq_group(seq_group: SequenceGroup)

engine.step
    seq_group_metadata_list, scheduler_outputs = scheduler.schedule()
    List[SequenceGroupMetadata], SchedulerOutputs -> ExecuteModelRequest -> model_executor.execute_model -> List[SamplerOutput]
    List[SamplerOutput] -> ChatModelOutputProcessor -> RequestOutput
    RequestOutput -> return to downstream
"""


class ChatWorkflow:
    Request: str = "light_vllm.task.chat.workflow.inputs:ChatRequest"
    InputProcessor: str = "light_vllm.task.chat.workflow.input_processor:ChatModelInputProcessor"
    SequenceProcessor: str = "light_vllm.task.chat.workflow.input_processor:ChatModelSequenceProcessor"
    OutputProcessor: str = "light_vllm.task.chat.workflow.output_processor:ChatModelOutputProcessor"
    Executor: str = "light_vllm.executor.gpu_executor:GPUExecutor"
    Scheduler: str = "light_vllm.core.scheduler:Scheduler"
    EngineArgs: str = "light_vllm.engine.arg_utils:EngineArgs"
    Tokenizer: str = "light_vllm.inputs.tokenizer:Tokenizer"
    ExecuteModelRequest = "light_vllm.sequence:ExecuteModelRequest"
