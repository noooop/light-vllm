
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (TYPE_CHECKING, Dict, List, Mapping, Optional, Set, Tuple,
                    Union)
from light_vllm.task.chat.workflow.sequence import Logprob, PromptLogprobs


class SequenceOutput:
    """The model output associated with a sequence.

    Args:
        parent_seq_id: The ID of the parent sequence (for forking in beam
            search).
        output_token: The output token ID.
        logprobs: The logprobs of the output token.
            (Token id -> logP(x_i+1 | x_0, ..., x_i))
    """

    def __init__(
        self,
        parent_seq_id: int,
        output_token: int,
        logprobs: Dict[int, Logprob],
    ) -> None:
        self.parent_seq_id = parent_seq_id
        self.output_token = output_token
        self.logprobs = logprobs

    def __repr__(self) -> str:
        return (f"SequenceOutput(parent_seq_id={self.parent_seq_id}, "
                f"output_token={self.output_token}, "
                f"logprobs={self.logprobs})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceOutput):
            raise NotImplementedError()
        equal = (self.parent_seq_id == other.parent_seq_id
                 and self.output_token == other.output_token)
        log_probs_equal = other.logprobs == self.logprobs
        return equal and log_probs_equal


class SequenceGroupOutput(ABC):
    """The base class for model outputs associated with a sequence group."""

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass


class CompletionSequenceGroupOutput(SequenceGroupOutput):
    """The model output associated with a completion sequence group."""

    def __init__(
        self,
        samples: List[SequenceOutput],
        prompt_logprobs: Optional[PromptLogprobs],
    ) -> None:
        self.samples = samples
        # Prompt logprob for each prompt query token.
        self.prompt_logprobs = prompt_logprobs

    def __repr__(self) -> str:
        return (f"CompletionSequenceGroupOutput(samples={self.samples}, "
                f"prompt_logprobs={self.prompt_logprobs})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompletionSequenceGroupOutput):
            raise NotImplementedError()
        return (self.samples == other.samples
                and self.prompt_logprobs == other.prompt_logprobs)


@dataclass
class SamplerOutput:
    """For each sequence group, we generate a list of SequenceOutput object,
    each of which contains one possible candidate for the next token.

    This data structure implements methods, so it can be used like a list, but
    also has optional fields for device tensors.
    """

    outputs: List[CompletionSequenceGroupOutput]

    # On-device tensor containing probabilities of each token.
    sampled_token_probs: Optional[torch.Tensor] = None

    # On-device tensor containing the logprobs of each token.
    logprobs: Optional["torch.Tensor"] = None

    # On-device tensor containing the sampled token ids.
    sampled_token_ids: Optional[torch.Tensor] = None

    # Spec decode metrics populated by workers.
    spec_decode_worker_metrics: Optional["SpecDecodeWorkerMetrics"] = None

    # Optional last hidden states from the model.
    hidden_states: Optional[torch.Tensor] = None

    def __getitem__(self, idx: int):
        return self.outputs[idx]

    def __setitem__(self, idx: int, value):
        self.outputs[idx] = value

    def __len__(self):
        return len(self.outputs)

    def __eq__(self, other: object):
        return isinstance(other,
                          self.__class__) and self.outputs == other.outputs

    def __repr__(self) -> str:
        """Show the shape of a tensor instead of its values to reduce noise.
        """
        sampled_token_probs_repr = ("None" if self.sampled_token_probs is None
                                    else self.sampled_token_probs.shape)
        sampled_token_ids_repr = ("None" if self.sampled_token_ids is None else
                                  self.sampled_token_ids.shape)
        return (
            f"SamplerOutput(outputs={self.outputs}, "
            f"sampled_token_probs={sampled_token_probs_repr}, "
            f"sampled_token_ids={sampled_token_ids_repr}, "
            f"spec_decode_worker_metrics={self.spec_decode_worker_metrics})")