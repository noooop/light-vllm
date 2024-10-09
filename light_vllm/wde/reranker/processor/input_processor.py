import time
from typing import Optional, Sequence

from light_vllm.wde.core.inputs.tokenizer import Tokenizer
from light_vllm.wde.core.llm_engine import LLMEngine
from light_vllm.wde.core.processor.input_processor import (InputProcessor,
                                                           RequestProcessor)
from light_vllm.wde.core.schema.engine_io import Params, ValidationError
from light_vllm.wde.encode_only.schema.engine_io import (
    EncodeOnlyInput, EncodeOnlySchedulableRequest)
from light_vllm.wde.reranker.schema.engine_io import (Pairs, RerankerInputs,
                                                      RerankerRequest)


class RerankerInputProcessor(InputProcessor):

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls()

    def __call__(self,
                 request_id: str,
                 inputs: Optional[RerankerInputs] = None,
                 params: Optional[Params] = None,
                 arrival_time: Optional[float] = None) -> RerankerRequest:
        if not arrival_time:
            arrival_time = time.time()

        if isinstance(inputs, Sequence):
            if len(inputs) != 2:
                raise ValidationError("Reranker model input must be pairs.")
            inputs = Pairs(query=inputs[0], passage=inputs[1])

        request = RerankerRequest(request_id=str(request_id),
                                  inputs=inputs,
                                  arrival_time=arrival_time)
        return request


class RerankerRequestProcessor(RequestProcessor):

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls(engine.tokenizer)

    def __call__(self,
                 request: RerankerRequest) -> EncodeOnlySchedulableRequest:
        text_pair = (request.inputs.query, request.inputs.passage)
        prompt_token_ids = self.tokenizer.encode(text_pair)
        schedulable_request = EncodeOnlySchedulableRequest(
            request_id=request.request_id,
            inputs=EncodeOnlyInput(prompt_token_ids=prompt_token_ids,
                                   prompt=None),
            arrival_time=request.arrival_time)
        return schedulable_request
