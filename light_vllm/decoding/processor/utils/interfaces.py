from abc import ABC, abstractmethod
from typing import List

from light_vllm.core.processor.tokenizer import Tokenizer
from light_vllm.decoding.config import SchedulerConfig
from light_vllm.decoding.processor.utils.stop_checker import StopChecker
from light_vllm.decoding.schema.execute_io import SequenceGroupOutput
from light_vllm.decoding.schema.sequence import SequenceGroup
from light_vllm.utils import Counter


class SequenceGroupOutputProcessor(ABC):
    """Interface for logic that processes new token ids in sequence groups,
    managing detokenization, stop checking, and freeing/forking sequences with
    the scheduler.

    This is highly coupled with the LLMEngine and should be seen as an extension
    of it. The logic is separated to simplify the LLMEngine class and allow
    separate implementations for single-step decoding (which supports beam
    search sequence forking) and multi-step decoding (which does not support
    beam search, but does support speculative decoding).
    """

    @staticmethod
    def create_output_processor(
        scheduler_config: SchedulerConfig,
        tokenizer: Tokenizer,
        seq_counter: Counter,
        stop_checker: "StopChecker",
    ):
        """Create an output processor.

        This returns a single-step output processor if num_lookahead_slots is
        zero, else returns a multi-step output processor.
        """
        if scheduler_config.num_lookahead_slots == 0:
            # Importing here to avoid cycle.
            from light_vllm.engine.output_processor.single_step import (
                SingleStepOutputProcessor)
            return SingleStepOutputProcessor(
                scheduler_config.max_model_len,
                tokenizer,
                seq_counter,
                stop_checker,
            )

    @abstractmethod
    def process_outputs(self, sequence_group: SequenceGroup,
                        outputs: List[SequenceGroupOutput]) -> None:
        """Process new token ids for the sequence group. Handles logic such as
        detokenization, stop checking, and freeing/forking sequences in the
        scheduler.
        """
        pass

    @abstractmethod
    def process_prompt_logprob(self, seq_group: SequenceGroup,
                               outputs: List[SequenceGroupOutput]) -> None:
        """Update prompt logprobs received from outputs to seq_group."""
        pass
