from typing import List

from light_vllm.core.llm_engine import LLMEngine
from light_vllm.core.processor.output_processor import OutputProcessor
from light_vllm.prefill_only.schema.engine_io import PrefillOnlySchedulerOutput
from light_vllm.reranker.schema.engine_io import RerankerRequestOutput
from light_vllm.reranker.schema.execute_io import RerankerExecuteOutput


class RerankerOutputProcessor(OutputProcessor):

    def __init__(self):
        pass

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls()

    def __call__(
            self, scheduler_output: PrefillOnlySchedulerOutput,
            execute_output: RerankerExecuteOutput
    ) -> List[RerankerRequestOutput]:
        scores = execute_output.scores.view(-1, ).numpy().tolist()
        request_outputs = []
        for i, request in enumerate(scheduler_output.scheduled_requests):
            prompt_token_ids = request.inputs.prompt_token_ids
            request_outputs.append(
                RerankerRequestOutput(request_id=request.request_id,
                                      prompt_token_ids=prompt_token_ids,
                                      finished=True,
                                      score=float(scores[i])))
        return request_outputs
