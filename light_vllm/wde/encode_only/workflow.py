from light_vllm.wde.prefill_only.workflow import PrefillOnlyWorkflow


class EncodeOnlyWorkflow(PrefillOnlyWorkflow):
    EngineArgs: str = ("light_vllm.wde.encode_only.arg_utils:"
                       "EncodeOnlyEngineArgs")
    attn_type: str = "ENCODER"
