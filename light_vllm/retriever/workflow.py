from light_vllm.decode_only.workflow import DecodeOnlyWorkflow
from light_vllm.encode_only.workflow import EncodeOnlyWorkflow


class RetrieverEncodeOnlyWorkflow(EncodeOnlyWorkflow):
    OutputProcessor: str = ("light_vllm.retriever.processor."
                            "output_processor:RetrieverOutputProcessor")


class RetrieverDecodeOnlyWorkflow(DecodeOnlyWorkflow):
    EngineArgs: str = ("light_vllm.retriever.arg_utils:"
                       "RetrieverDecodeOnlyEngineArgs")
