from light_vllm.wde.decode_only.workflow import DecodeOnlyWorkflow
from light_vllm.wde.encode_only.workflow import EncodeOnlyWorkflow


class RetrieverEncodeOnlyWorkflow(EncodeOnlyWorkflow):
    OutputProcessor: str = ("light_vllm.wde.retriever.processor."
                            "output_processor:RetrieverModelOutputProcessor")


class RetrieverDecodeOnlyWorkflow(DecodeOnlyWorkflow):
    EngineArgs: str = ("light_vllm.wde.retriever.arg_utils:"
                       "RetrieverDecodeOnlyEngineArgs")
