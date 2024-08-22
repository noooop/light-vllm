
from typing import Any, Dict, Optional

import time
from light_vllm.utils import Counter
from light_vllm.inputs.tokenizer import Tokenizer
from light_vllm.config import ModelConfig, CacheConfig
from light_vllm.layers.sampling_params import SamplingParams
from light_vllm.task.chat.schema.inputs import PromptInput, ChatInput, ChatRequest
from light_vllm.task.base.schema.sequence import Sequence, SequenceGroup
from light_vllm.task.base.processor.input_processor import InputProcessor


class ChatModelPromptProcessor(object):
    """
    PromptInput -> ChatModelPromptProcessor -> ChatInput
    """
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input: PromptInput) -> ChatInput:
        if isinstance(input, str):
            input = {"prompt": input}

        if "prompt_token_ids" not in input:
            tokenizer = self.tokenizer

            prompt_token_ids = tokenizer.encode(input["prompt"])
        else:
            prompt_token_ids = input["prompt_token_ids"]

        chat_input = ChatInput(prompt_token_ids=prompt_token_ids,
                               prompt=input.get("prompt"))
        return chat_input


class ChatModelSequenceProcessor(object):
    """
    ChatRequest -> ChatModelSequenceProcessor -> SequenceGroup
    """
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 tokenizer: Tokenizer,
                 seq_counter: Counter):
        self.block_size = cache_config.block_size
        self.eos_token_id = tokenizer.eos_token_id
        self.max_logprobs = model_config.max_logprobs
        self.seq_counter = seq_counter

        self.generation_config_fields = self._load_generation_config_dict(
            model_config)

    def _load_generation_config_dict(self, model_config: ModelConfig) -> Dict[str, Any]:
        from light_vllm.models.transformers_utils.config import try_get_generation_config
        config = try_get_generation_config(
            model_config.model,
            trust_remote_code=model_config.trust_remote_code,
            revision=model_config.revision,
        )

        if config is None:
            return {}

        return config.to_diff_dict()

    def __call__(self, chat_request: ChatRequest, arrival_time: Optional[float] = None) -> SequenceGroup:
        """Creates a SequenceGroup with SamplingParams."""

        sampling_params = chat_request["sampling_params"]

        if arrival_time is None:
            arrival_time = time.time()

        block_size = self.block_size
        eos_token_id = self.eos_token_id
        seq_id = next(self.seq_counter)

        seq = Sequence(seq_id, chat_request["input"], block_size, eos_token_id)

        max_logprobs = self.max_logprobs
        if (sampling_params.logprobs
            and sampling_params.logprobs > max_logprobs) or (
                sampling_params.prompt_logprobs
                and sampling_params.prompt_logprobs > max_logprobs):
            raise ValueError(f"Cannot request more than "
                             f"{max_logprobs} logprobs.")

        # Defensive copy of SamplingParams, which are used by the sampler,
        # this doesn't deep-copy LogitsProcessor objects
        sampling_params = sampling_params.clone()

        sampling_params.update_from_generation_config(
            self.generation_config_fields, seq.eos_token_id)

        # Create the sequence group.
        seq_group = SequenceGroup(
            request_id=chat_request["request_id"],
            seqs=[seq],
            arrival_time=arrival_time,
            sampling_params=sampling_params)

        return seq_group


class ChatModelInputProcessor(InputProcessor):
    """
    PromptInput -> ChatModelInputProcessor -> ChatRequest

    PromptInput -> ChatModelPromptProcessor -> ChatInput
    ChatInput + request_id + arrival_time -> ChatRequest
    ChatRequest -> ChatModelSequenceProcessor -> ChatRequest
    """
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 tokenizer: Tokenizer,
                 seq_counter: Counter):
        self.prompt_processor = ChatModelPromptProcessor(tokenizer)
        self.sequence_processor = ChatModelSequenceProcessor(model_config, cache_config, tokenizer, seq_counter)

    def __call__(
            self,
            request_id: str,
            prompt: PromptInput,
            params: SamplingParams,
            arrival_time: Optional[float] = None) -> ChatRequest:
        input = self.prompt_processor(prompt)

        request = ChatRequest(request_id=str(request_id), input=input, sampling_params=params)
        seq_group = self.sequence_processor(request, arrival_time)
        request["input"] = seq_group
        return request