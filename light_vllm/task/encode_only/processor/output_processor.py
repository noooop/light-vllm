
from typing import List
from light_vllm.task.base.processor.output_processor import OutputProcessor
from light_vllm.task.encode_only.schema.outputs import EncodeOnlyRequestOutput


class EncodeOnlyModelOutputProcessor(OutputProcessor):
    def __init__(self):
        pass

    @classmethod
    def from_engine(cls, engine):
        return cls()

    def __call__(self, scheduler_outputs, execute_output) -> List[EncodeOnlyRequestOutput]:
        request_outputs = []
        for request, outputs in zip(scheduler_outputs.scheduled_requests, execute_output):

            request_outputs.append(EncodeOnlyRequestOutput(
                request_id=request.request_id,
                prompt_token_ids=[],
                finished=True,
                outputs=outputs
            ))
        return request_outputs
