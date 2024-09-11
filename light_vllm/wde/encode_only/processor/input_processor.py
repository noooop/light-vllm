from typing import Optional

import time
from light_vllm.inputs.tokenizer import Tokenizer
from light_vllm.wde.core.llm_engine import LLMEngine
from light_vllm.wde.core.schema.engine_io import Params, PromptInput
from light_vllm.wde.encode_only.schema.engine_io import EncodeOnlyInput, EncodeOnlyRequest, EncodeOnlySchedulableRequest
from light_vllm.wde.core.processor.input_processor import InputProcessor, RequestProcessor


class EncodeOnlyModelInputProcessor(InputProcessor):
    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls()

    def __call__(self,
                 request_id: str,
                 inputs: Optional[PromptInput] = None,
                 params: Optional[Params] = None,
                 arrival_time: Optional[float] = None) -> EncodeOnlyRequest:
        if not arrival_time:
            arrival_time = time.time()
        request = EncodeOnlyRequest(request_id=str(request_id),
                                    inputs=inputs,
                                    arrival_time=arrival_time)
        return request


class EncodeOnlyModelRequestProcessor(RequestProcessor):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls(engine.tokenizer)

    def __call__(self, request: EncodeOnlyRequest) -> EncodeOnlySchedulableRequest:
        inputs = request.inputs

        if isinstance(inputs, str):
            inputs = {"prompt": inputs}

        if "prompt_token_ids" not in inputs:
            tokenizer = self.tokenizer

            prompt_token_ids = tokenizer.encode(inputs["prompt"])
        else:
            prompt_token_ids = inputs["prompt_token_ids"]

        schedulable_request = EncodeOnlySchedulableRequest(
            request_id=request.request_id,
            inputs=EncodeOnlyInput(
                prompt_token_ids=prompt_token_ids,
                prompt=inputs.get("prompt")
            ),
            arrival_time=request.arrival_time
        )

        return schedulable_request
