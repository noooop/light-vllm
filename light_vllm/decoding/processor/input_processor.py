from typing import Any, Dict

from light_vllm.core.inputs.tokenizer import Tokenizer
from light_vllm.core.processor.input_processor import TextRequestProcessor
from light_vllm.core.schema.engine_io import (TextRequest,
                                              TextSchedulableRequest)
from light_vllm.decoding.config import CacheConfig, ModelConfig
from light_vllm.decoding.schema.engine_io import ChatSchedulableRequest
from light_vllm.decoding.schema.sequence import Sequence, SequenceGroup
from light_vllm.utils import Counter


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

    def __call__(self, request: TextSchedulableRequest) -> SequenceGroup:
        """Creates a SequenceGroup with SamplingParams."""

        sampling_params = request.params
        arrival_time = request.arrival_time

        block_size = self.block_size
        eos_token_id = self.eos_token_id
        seq_id = next(self.seq_counter)

        seq = Sequence(seq_id, request.inputs, block_size, eos_token_id)

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
        seq_group = SequenceGroup(request_id=request.request_id,
                                  seqs=[seq],
                                  arrival_time=arrival_time,
                                  sampling_params=sampling_params)

        return seq_group


class ChatModelRequestProcessor(TextRequestProcessor):
    """
    ChatRequest -> ChatModelRequestProcessor -> ChatSchedulableRequest

    PromptInput -> ChatModelPromptProcessor -> ChatInput
    ChatRequest -> ChatModelSequenceProcessor -> SequenceGroup
    """

    def __init__(self, model_config: ModelConfig, cache_config: CacheConfig,
                 tokenizer: Tokenizer, seq_counter: Counter):
        super().__init__(tokenizer)
        self.sequence_processor = ChatModelSequenceProcessor(
            model_config, cache_config, tokenizer, seq_counter)

    @classmethod
    def from_engine(cls, engine):
        engine.seq_counter = Counter()

        return cls(engine.engine_config.model_config,
                   engine.engine_config.cache_config, engine.tokenizer,
                   engine.seq_counter)

    def __call__(self, request: TextRequest) -> ChatSchedulableRequest:
        schedulable_request = TextRequestProcessor.__call__(self, request)

        seq_group = self.sequence_processor(schedulable_request)
        print(seq_group)
        return ChatSchedulableRequest(request_id=request.request_id,
                                      seq_group=seq_group,
                                      arrival_time=request.arrival_time)
