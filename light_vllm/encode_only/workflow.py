from light_vllm.prefill_only.workflow import PrefillOnlyWorkflow


class EncodeOnlyWorkflow(PrefillOnlyWorkflow):
    EngineArgs: str = ("light_vllm.encode_only.arg_utils:"
                       "EncodeOnlyEngineArgs")
    OutputProcessor: str = ("light_vllm.encode_only.processor."
                            "output_processor:EncodeOnlyOutputProcessor")
    attn_type: str = "ENCODER"
