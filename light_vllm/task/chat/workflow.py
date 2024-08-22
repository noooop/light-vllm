

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

from light_vllm.task.base.workflow import Workflow


class ChatWorkflow(Workflow):
    InputProcessor: str = "light_vllm.task.chat.processor.input_processor:ChatModelInputProcessor"
    SequenceProcessor: str = "light_vllm.task.chat.processor.input_processor:ChatModelSequenceProcessor"
    OutputProcessor: str = "light_vllm.task.chat.processor.output_processor:ChatModelOutputProcessor"
    Request: str = "light_vllm.task.chat.schema.inputs:ChatRequest"

    Worker: str = "light_vllm.worker.worker:Worker"

