import time
from typing import Any, Dict, Optional

from light_vllm.core.inputs.tokenizer import Tokenizer
from light_vllm.core.llm_engine import LLMEngine
from light_vllm.core.processor.input_processor import (InputProcessor,
                                                       RequestProcessor)
from light_vllm.core.schema.engine_io import (PromptInput, TextPrompt,
                                              TokensPrompt)
from light_vllm.decoding.backends.sampling_params import SamplingParams
from light_vllm.decoding.config import CacheConfig, ModelConfig
from light_vllm.decoding.schema.engine_io import (ChatInput, ChatRequest,
                                                  ChatSchedulableRequest)
from light_vllm.decoding.schema.sequence import Sequence, SequenceGroup
from light_vllm.utils import Counter


class ChatModelInputProcessor(InputProcessor):

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls()

    def __call__(self,
                 request_id: str,
                 inputs: Optional[PromptInput] = None,
                 params: Optional[SamplingParams] = None,
                 arrival_time: Optional[float] = None) -> ChatRequest:
        if not arrival_time:
            arrival_time = time.time()
        request = ChatRequest(request_id=str(request_id),
                              inputs=inputs,
                              params=params,
                              arrival_time=arrival_time)
        return request


class ChatModelPromptProcessor(object):
    """
    PromptInput -> ChatPromptProcessor -> ChatInput
    """

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, inputs: PromptInput) -> ChatInput:
        if isinstance(inputs, str):
            inputs = {"prompt": inputs}
        elif isinstance(input, TextPrompt):
            inputs = {"prompt": inputs.prompt}
        elif isinstance(input, TokensPrompt):
            inputs = {"prompt_token_ids", inputs.prompt_token_ids}

        if "prompt_token_ids" not in inputs:
            tokenizer = self.tokenizer

            prompt_token_ids = tokenizer.encode(inputs["prompt"])
        else:
            prompt_token_ids = inputs["prompt_token_ids"]

        chat_input = ChatInput(prompt_token_ids=prompt_token_ids,
                               prompt=inputs.get("prompt"))
        return chat_input


class ChatModelSequenceProcessor(object):
    """
    ChatRequest -> ChatSequenceProcessor -> SequenceGroup
    """

    def __init__(self, model_config: ModelConfig, cache_config: CacheConfig,
                 tokenizer: Tokenizer, seq_counter: Counter):
        self.block_size = cache_config.block_size
        self.eos_token_id = tokenizer.eos_token_id
        self.max_logprobs = model_config.max_logprobs
        self.seq_counter = seq_counter

        self.generation_config_fields = self._load_generation_config_dict(
            model_config)

    def _load_generation_config_dict(
            self, model_config: ModelConfig) -> Dict[str, Any]:
        from light_vllm.core.models.transformers_utils.config import (
            try_get_generation_config)
        config = try_get_generation_config(
            model_config.model,
            trust_remote_code=model_config.trust_remote_code,
            revision=model_config.revision,
        )

        if config is None:
            return {}

        return config.to_diff_dict()

    def __call__(self,
                 chat_request: ChatRequest,
                 arrival_time: Optional[float] = None) -> SequenceGroup:
        """Creates a SequenceGroup with SamplingParams."""

        sampling_params = chat_request.params

        if arrival_time is None:
            arrival_time = time.time()

        block_size = self.block_size
        eos_token_id = self.eos_token_id
        seq_id = next(self.seq_counter)

        seq = Sequence(seq_id, chat_request.inputs, block_size, eos_token_id)

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
        seq_group = SequenceGroup(request_id=chat_request.request_id,
                                  seqs=[seq],
                                  arrival_time=arrival_time,
                                  sampling_params=sampling_params)

        return seq_group


class ChatModelRequestProcessor(RequestProcessor):
    """
    ChatRequest -> ChatModelRequestProcessor -> ChatSchedulableRequest

    PromptInput -> ChatModelPromptProcessor -> ChatInput
    ChatRequest -> ChatModelSequenceProcessor -> SequenceGroup
    """

    def __init__(self, model_config: ModelConfig, cache_config: CacheConfig,
                 tokenizer: Tokenizer, seq_counter: Counter):
        self.prompt_processor = ChatModelPromptProcessor(tokenizer)
        self.sequence_processor = ChatModelSequenceProcessor(
            model_config, cache_config, tokenizer, seq_counter)

    @classmethod
    def from_engine(cls, engine):
        engine.seq_counter = Counter()

        return cls(engine.engine_config.model_config,
                   engine.engine_config.cache_config, engine.tokenizer,
                   engine.seq_counter)

    def __call__(self, request: ChatRequest) -> ChatSchedulableRequest:
        request.inputs = self.prompt_processor(request.inputs)
        seq_group = self.sequence_processor(request, request.arrival_time)
        return ChatSchedulableRequest(request_id=request.request_id,
                                      seq_group=seq_group,
                                      arrival_time=request.arrival_time)
