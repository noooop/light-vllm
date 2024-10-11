from light_vllm.encode_only.workflow import EncodeOnlyWorkflow


class RerankerWorkflow(EncodeOnlyWorkflow):
    InputProcessor: str = ("light_vllm.reranker.processor."
                           "input_processor:RerankerInputProcessor")
    RequestProcessor: str = ("light_vllm.reranker.processor."
                             "input_processor:RerankerRequestProcessor")
    OutputProcessor: str = ("light_vllm.reranker.processor."
                            "output_processor:RerankerOutputProcessor")
