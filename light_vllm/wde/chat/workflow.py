

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

from light_vllm.wde.core.workflow import Workflow


class ChatWorkflow(Workflow):
    EngineArgs: str = "light_vllm.wde.chat.arg_utils:ChatEngineArgs"
    InputProcessor: str = "light_vllm.wde.chat.processor.input_processor:ChatModelInputProcessor"
    RequestProcessor: str = "light_vllm.wde.chat.processor.input_processor:ChatModelRequestProcessor"
    OutputProcessor: str = "light_vllm.wde.chat.processor.output_processor:ChatModelOutputProcessor"
    ModelPreProcessor: str = "light_vllm.wde.chat.processor.model_pre_processor:ChatModelPreProcessor"
    Worker: str = "light_vllm.wde.chat.worker.gpu_worker:Worker"
    Scheduler: str = "light_vllm.wde.chat.scheduler:Scheduler"
