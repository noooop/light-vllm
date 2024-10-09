from typing import List

import torch

from light_vllm.wde.core.llm_engine import LLMEngine
from light_vllm.wde.core.processor.output_processor import OutputProcessor
from light_vllm.wde.encode_only.schema.engine_io import (
    EncodeOnlySchedulerOutput)
from light_vllm.wde.reranker.schema.engine_io import RerankerRequestOutput


class RerankerOutputProcessor(OutputProcessor):

    def __init__(self):
        pass

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls()

    def __call__(self, scheduler_output: EncodeOnlySchedulerOutput,
                 execute_output: torch.Tensor) -> List[RerankerRequestOutput]:
        request_outputs = []
        for i, request in enumerate(scheduler_output.requests):
            prompt_token_ids = request.inputs.prompt_token_ids
            request_outputs.append(
                RerankerRequestOutput(request_id=request.request_id,
                                      prompt_token_ids=prompt_token_ids,
                                      finished=True,
                                      score=float(execute_output[i])))
        return request_outputs
