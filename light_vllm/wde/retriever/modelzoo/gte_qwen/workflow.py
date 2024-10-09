from light_vllm.wde.decode_only.workflow import DecodeOnlyWorkflow


class Qwen2Workflow(DecodeOnlyWorkflow):
    EngineArgs: str = ("light_vllm.wde.retriever.modelzoo."
                       "gte_qwen.arg_utils:Qwen2EngineArgs")
