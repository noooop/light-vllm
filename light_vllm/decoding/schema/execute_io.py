from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import msgspec
import torch

from light_vllm.core.schema.execute_io import (ExecuteInput, ExecuteOutput,
                                               ModelInput, WorkerInput)
from light_vllm.decoding.backends.sampling_metadata import (
    SamplingMetadata, SequenceGroupToSample)
from light_vllm.decoding.schema.sequence import (Logprob, PromptLogprobs,
                                                 SequenceGroupMetadata)


@dataclass
class DecodingExecuteModelInput:
    """The model execution request, containing CPU metadata only. The LLM
    engine should create an instance of this class for each request batch."""
    # The sequence group metadata list.
    seq_group_metadata_list: List[SequenceGroupMetadata]
    # Blocks to swap in. List of CPU -> GPU block number.
    blocks_to_swap_in: List[Tuple[int, int]] = field(default_factory=list)
    # Blocks to swap out. List of GPU -> CPU block number.
    blocks_to_swap_out: List[Tuple[int, int]] = field(default_factory=list)
    # Blocks to copy. Source to dest block.
    blocks_to_copy: List[Tuple[int, int]] = field(default_factory=list)

    def clone(
        self, seq_group_metadata_list: List[SequenceGroupMetadata]
    ) -> "DecodingExecuteModelInput":
        """Clone the request with a new sequence group metadata list."""
        return DecodingExecuteModelInput(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=self.blocks_to_swap_in.copy(),
            blocks_to_swap_out=self.blocks_to_swap_out.copy(),
            blocks_to_copy=self.blocks_to_copy.copy())


@dataclass
class DecodingModelInputForGPU(ModelInput):
    """
    This base class contains metadata needed for the base model forward pass
    but not metadata for possible additional steps, e.g., sampling. Model
    runners that run additional steps should subclass this method to add
    additional fields.
    """
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    seq_lens: Optional[List[int]] = None
    query_lens: Optional[List[int]] = None
    attn_metadata: Optional["AttentionMetadata"] = None


@dataclass
class DecodingModelInputForGPUWithSamplingMetadata(DecodingModelInputForGPU):
    """
    Used by the ModelRunner.
    """
    sampling_metadata: Optional["SamplingMetadata"] = None
    # Used for speculative decoding. We do not broadcast it because it is only
    # used by the driver worker.
    is_prompt: Optional[bool] = None


@dataclass
class DecodingWorkerInputForGPU(WorkerInput):
    """Local inputs to each worker. May contain device-specific data. These
    fields should be broadcastable to other workers.
    """

    num_seq_groups: Optional[int] = None
    blocks_to_swap_in: Optional[torch.Tensor] = None
    blocks_to_swap_out: Optional[torch.Tensor] = None
    blocks_to_copy: Optional[torch.Tensor] = None


@dataclass
class DecodingExecuteInput(ExecuteInput):
    worker_input: Optional[DecodingWorkerInputForGPU]
    model_input: Optional[DecodingModelInputForGPUWithSamplingMetadata]


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


from light_vllm.decoding.backends.sampling_params import SamplingType

# (num_token_ids, num_parent_ids) per sequence group.
SampleResultType = List[Tuple[List[int], List[int]]]

# Types of temporary data structures used for
# computing sample_result
SampleMetadataType = Dict[SamplingType, Tuple[List[int],
                                              List[SequenceGroupToSample]]]
MultinomialSamplesType = Dict[SamplingType, torch.Tensor]
SampleResultsDictType = Dict[int, Tuple[List[int], List[int]]]


# Encapsulates temporary data structures for computing
# sample_result.
#
# * For multi-step scheduling: must be returned
#   by `Sampler.forward()` and used later to compute the pythonized
#   sample_result
#
# * For single-step scheduling: consumed immediately
#   inside `Sampler.forward()` to compute pythonized sample_result.
@dataclass
class SampleResultArgsType:
    sample_metadata: SampleMetadataType
    multinomial_samples: MultinomialSamplesType
    sample_results_dict: SampleResultsDictType
    sampling_metadata: SamplingMetadata
    greedy_samples: Optional[torch.Tensor]
    beam_search_logprobs: Optional[torch.Tensor]


# Union of non-deferred (single-step scheduling)
# vs deferred (multi-step scheduling)
# sample result types
MaybeDeferredSampleResultType = Union[SampleResultType, SampleResultArgsType]

# Abbreviation of the _sample() return type
SampleReturnType = Tuple[MaybeDeferredSampleResultType, Optional[torch.Tensor]]


class SamplerOutput(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        array_like=True):  # type: ignore[call-arg]
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

    # Holds either (1) the pythonized sampler result (single-step scheduling)
    # or (2) what will be arguments for later deferred pythonization of the
    # sampler result (muliti-step scheduling)
    deferred_sample_results_args: Optional[SampleResultArgsType] = None

    # On-device tensor containing the sampled token ids.
    sampled_token_ids: Optional[torch.Tensor] = None
    # CPU tensor containing the sampled token ids. Used during multi-step to
    # return the sampled token ids from last rank to AsyncLLMEngine to be
    # 'broadcasted' to all other PP ranks for next step.
    sampled_token_ids_cpu: Optional[torch.Tensor] = None

    # Optional last hidden states from the model.
    hidden_states: Optional[torch.Tensor] = None

    # Optional prefill hidden states from the model
    # (used for models like EAGLE).
    prefill_hidden_states: Optional[torch.Tensor] = None

    # Time taken in the forward pass for this across all workers
    model_forward_time: Optional[float] = None

    # Time taken in the model execute function. This will include model forward,
    # block/sync across workers, cpu-gpu sync time and sampling time.
    model_execute_time: Optional[float] = None

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
        return (f"SamplerOutput(outputs={self.outputs}, "
                f"sampled_token_probs={sampled_token_probs_repr}, "
                f"sampled_token_ids={sampled_token_ids_repr}.")


@dataclass
class DecodingExecuteOutput(ExecuteOutput):
    pass
