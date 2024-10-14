from typing import List

import torch

from light_vllm.encode_only.processor.output_processor import (
    EncodeOnlyOutputProcessor)
from light_vllm.prefill_only.schema.engine_io import PrefillOnlySchedulerOutput
from light_vllm.retriever.schema.engine_io import EmbeddingRequestOutput


class RetrieverOutputProcessor(EncodeOnlyOutputProcessor):

    def __call__(self, scheduler_output: PrefillOnlySchedulerOutput,
                 execute_output: torch.Tensor) -> List[EmbeddingRequestOutput]:
        request_outputs = []
        for request, outputs in zip(scheduler_output.scheduled_requests,
                                    execute_output):
            prompt_token_ids = request.inputs.prompt_token_ids
            request_outputs.append(
                EmbeddingRequestOutput(request_id=request.request_id,
                                       prompt_token_ids=prompt_token_ids,
                                       finished=True,
                                       outputs=outputs))
        return request_outputs
