from typing import Optional, Type

from vllm.config import (ModelConfig, SchedulerConfig,
                         TokenizerPoolConfig)

from .base_tokenizer_group import AnyTokenizer, BaseTokenizerGroup
from .tokenizer_group import TokenizerGroup


def init_tokenizer_from_configs(model_config: ModelConfig,
                                scheduler_config: SchedulerConfig):
    init_kwargs = dict(tokenizer_id=model_config.tokenizer,
                       max_num_seqs=scheduler_config.max_num_seqs,
                       max_input_length=None,
                       tokenizer_mode=model_config.tokenizer_mode,
                       trust_remote_code=model_config.trust_remote_code,
                       revision=model_config.tokenizer_revision)

    return get_tokenizer_group(**init_kwargs)


def get_tokenizer_group(**init_kwargs) -> BaseTokenizerGroup:
    return TokenizerGroup.from_config(**init_kwargs)


__all__ = ["AnyTokenizer", "get_tokenizer_group", "BaseTokenizerGroup"]
