
from typing import List
import torch
from light_vllm.engine.llm_engine import LLMEngine
from light_vllm.wde.core.processor.output_processor import OutputProcessor
from light_vllm.wde.encode_only.schema.engine_io import EncodeOnlyRequestOutput, EncodeOnlySchedulerOutput


class EncodeOnlyModelOutputProcessor(OutputProcessor):
    def __init__(self):
        pass

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls()

    def __call__(self,
                 scheduler_output: EncodeOnlySchedulerOutput,
                 execute_output: torch.Tensor) -> List[EncodeOnlyRequestOutput]:
        request_outputs = []
        offset = 0
        for request in scheduler_output.scheduled_requests:
            prompt_token_ids = request.inputs.prompt_token_ids
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
