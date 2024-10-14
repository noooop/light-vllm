import pytest

from light_vllm.core.inputs.tokenizer import Tokenizer
from light_vllm.core.processor.input_processor import (TextInputProcessor,
                                                       TextRequestProcessor)
from light_vllm.core.schema.engine_io import TextOnlyInputs, TokensPrompt


@pytest.fixture(scope="session")
def request_id():
    return "0"


TOKENIZER_NAMES = ["facebook/opt-125m", "gpt2"]


@pytest.mark.parametrize("tokenizer_name", TOKENIZER_NAMES)
def test_request_processor(request_id: str, tokenizer_name: str):

    tokenizer = Tokenizer(tokenizer_name=tokenizer_name)
    input_processor = TextInputProcessor()
    request_processor = TextRequestProcessor(tokenizer)

    prompt = "test"
    request = input_processor(request_id, prompt)

    assert request.inputs == {"prompt": prompt}

    schedulable_request = request_processor(request)

    assert isinstance(schedulable_request.inputs, TextOnlyInputs)
    assert len(schedulable_request.inputs.prompt_token_ids) > 0

    prompt_token_ids = [0]
    request = input_processor(request_id,
                              TokensPrompt(prompt_token_ids=prompt_token_ids))

    schedulable_request = request_processor(request)

    assert isinstance(schedulable_request.inputs, TextOnlyInputs)
    assert len(schedulable_request.inputs.prompt_token_ids) > 0
