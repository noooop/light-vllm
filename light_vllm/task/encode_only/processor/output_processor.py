
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
        offset = 0
        for request in scheduler_outputs.scheduled_requests:
            prompt_token_ids = request.input.prompt_token_ids
            n_tokens = len(prompt_token_ids)
            request_outputs.append(
                EncodeOnlyRequestOutput(
                    request_id=request.request_id,
                    prompt_token_ids=prompt_token_ids,
                    finished=True,
                    outputs=execute_output[offset: offset+n_tokens]
                )
            )
            offset += n_tokens
        return request_outputs
