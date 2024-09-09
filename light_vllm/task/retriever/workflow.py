

from light_vllm.task.encode_only.workflow import EncodeOnlyWorkflow


class RetrieverWorkflow(EncodeOnlyWorkflow):
    OutputProcessor: str = "light_vllm.task.retriever.processor.output_processor:RetrieverModelOutputProcessor"
