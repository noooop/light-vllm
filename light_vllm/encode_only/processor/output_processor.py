from typing import List

from light_vllm.core.llm_engine import LLMEngine
from light_vllm.core.processor.output_processor import (OutputProcessor,
                                                        RequestOutput)
from light_vllm.encode_only.schema.engine_io import EncodeOnlyRequestOutput
from light_vllm.encode_only.schema.execute_io import EncodeOnlyExecuteOutput
from light_vllm.prefill_only.schema.engine_io import PrefillOnlySchedulerOutput


class EncodeOnlyOutputProcessor(OutputProcessor):

    def __init__(self):
        pass

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls()

    def __call__(
            self, scheduler_output: PrefillOnlySchedulerOutput,
            execute_output: EncodeOnlyExecuteOutput) -> List[RequestOutput]:
        if execute_output.pooled_output is not None:
            request_outputs = []
            for request, outputs in zip(scheduler_output.scheduled_requests,
                                        execute_output.pooled_output):
                prompt_token_ids = request.inputs.prompt_token_ids
                request_outputs.append(
                    EncodeOnlyRequestOutput(request_id=request.request_id,
                                            arrival_time=request.arrival_time,
                                            prompt_token_ids=prompt_token_ids,
                                            finished=True,
                                            outputs=outputs))
            return request_outputs
        else:
            request_outputs = []
            offset = 0
            for request in scheduler_output.scheduled_requests:
                prompt_token_ids = request.inputs.prompt_token_ids
                n_tokens = len(prompt_token_ids)
                request_outputs.append(
                    EncodeOnlyRequestOutput(
                        request_id=request.request_id,
                        arrival_time=request.arrival_time,
                        prompt_token_ids=prompt_token_ids,
                        finished=True,
                        outputs=execute_output.last_hidden_states[offset]))
                offset += n_tokens
            return request_outputs
