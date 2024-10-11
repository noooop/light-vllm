from light_vllm.prefill_only.workflow import PrefillOnlyWorkflow


class EncodeOnlyWorkflow(PrefillOnlyWorkflow):
    EngineArgs: str = ("light_vllm.encode_only.arg_utils:"
                       "EncodeOnlyEngineArgs")
    attn_type: str = "ENCODER"
