
from typing import List
from light_vllm.task.base.processor.output_processor import OutputProcessor
from light_vllm.task.retriever.schema.outputs import EmbeddingRequestOutput


class RetrieverModelOutputProcessor(OutputProcessor):
    def __init__(self):
        pass

    @classmethod
    def from_engine(cls, engine):
        return cls()

    def __call__(self, scheduler_outputs, execute_output) -> List[EmbeddingRequestOutput]:
        request_outputs = []
        for request, outputs in zip(scheduler_outputs.scheduled_requests, execute_output):
            prompt_token_ids = request.input.prompt_token_ids
            request_outputs.append(EmbeddingRequestOutput(
                request_id=request.request_id,
                prompt_token_ids=prompt_token_ids,
                finished=True,
                outputs=outputs
            ))
        return request_outputs