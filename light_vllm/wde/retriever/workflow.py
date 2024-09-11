

from light_vllm.wde.encode_only.workflow import EncodeOnlyWorkflow


class RetrieverWorkflow(EncodeOnlyWorkflow):
    OutputProcessor: str = "light_vllm.wde.retriever.processor.output_processor:RetrieverModelOutputProcessor"
