from typing import List, Tuple

from light_vllm.core.processor.tokenizer import Tokenizer
from light_vllm.decoding.processor.utils.interfaces import (
    SequenceGroupOutputProcessor)
from light_vllm.decoding.processor.utils.stop_checker import StopChecker
from light_vllm.decoding.schema.execute_io import SequenceGroupOutput
from light_vllm.decoding.schema.sequence import SequenceGroup
from light_vllm.logger import init_logger
from light_vllm.utils import Counter

logger = init_logger(__name__)


class SingleStepOutputProcessor(SequenceGroupOutputProcessor):
    """SequenceGroupOutputProcessor which handles "output processing" logic,
    which happens after the model returns generated token ids and before
    scheduling of the next batch. Output processing logic includes
    detokenization, and determining if a sequence is finished (e.g. via max len
    or eos token).

    The SingleStepOutputProcessor is specialized to the case where the model
    emits at most a single token per invocation, which precludes configurations
    such as speculative decoding or multi-step decoding. This enables beam
    search sampling, which requires forking/finishing/freeing sequences in a way
    that is currently difficult to schedule multiple steps ahead of time.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        seq_counter: Counter,
        stop_checker: StopChecker,
        max_model_len: int,
    ):
        self.tokenizer = tokenizer
        self.seq_counter = seq_counter
        self.stop_checker = stop_checker
        self.max_model_len = max_model_len

    def process_outputs(self, sequence_group: SequenceGroup,
                        outputs: List[SequenceGroupOutput]) -> Tuple[List]:
        """Append all new tokens to sequences in the sequence group. Fork any
        surviving beam candidates; free any unsurviving ones.

        Invokes detokenizer to detokenize new tokens, and also marks sequences
        as finished if they meet stop conditions.
        """
        assert (len(outputs) == 1
                ), f"{type(self)} does not support multiple outputs per step"
        return self._process_sequence_group_outputs(sequence_group, outputs[0])

    def process_prompt_logprob(self, seq_group: SequenceGroup,
                               outputs: List[SequenceGroupOutput]) -> None:
        assert len(outputs) == 1, ("Single step should only has 1 output.")
        output = outputs[0]
        prompt_logprobs = output.prompt_logprobs

        # If this is the first (or only) "chunk" of the prefill, we need
        # to prepend None to the list of prompt logprobs. The reason for this
        # is that for N prompt tokens, the Sampler will generate N-1 total
        # prompt logprobs during prefill since the token at idx 0 will not
        # have a logprob associated with it.
        if prompt_logprobs is not None:
            if not seq_group.prompt_logprobs:
                prompt_logprobs = [None] + prompt_logprobs
                seq_group.prompt_logprobs = []

            if seq_group.sampling_params.detokenize and self.tokenizer:
                self.tokenizer.decode_prompt_logprobs_inplace(
                    seq_group,
                    prompt_logprobs,
                    position_offset=len(seq_group.prompt_logprobs))

            seq_group.prompt_logprobs.extend(prompt_logprobs)

    def _process_sequence_group_outputs(
            self, seq_group: SequenceGroup,
            outputs: SequenceGroupOutput) -> Tuple[List, List]:
        seq_need_free = []
        seq_need_fork = []

        sampling_params = seq_group.sampling_params
        if sampling_params.n == 1:
            # only have one output sample
            sample = outputs.samples[0]
            # only have one sequence
            seq = seq_group.seqs[0]
            seq.append_token_id(sample.output_token, sample.logprobs)
            if sampling_params.detokenize and self.tokenizer:
                new_char_count = self.tokenizer.decode_sequence_inplace(
                    seq, sampling_params)
            else:
                new_char_count = 0
            self.stop_checker.maybe_stop_sequence(seq, new_char_count,
                                                  sampling_params)
            if seq.is_finished():
                seq_need_free.append(seq)
            return seq_need_fork, seq_need_free
