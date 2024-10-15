import functools
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple, Type

from torch import nn
from transformers import PretrainedConfig
from typing_extensions import TypeVar

from light_vllm.logger import init_logger
from light_vllm.utils import print_warning_once

logger = init_logger(__name__)

C = TypeVar("C", bound=PretrainedConfig, default=PretrainedConfig)


@dataclass(frozen=True)
class InputContext:
    """
    Contains information about the model which may be used to
    modify the inputs.
    """

    model_config: "ModelConfig"
    """The configuration of the model."""

    def get_hf_config(self, hf_config_type: Type[C] = PretrainedConfig) -> C:
        """
        Get the HuggingFace configuration
        (:class:`transformers.PretrainedConfig`) of the model,
        additionally checking its type.

        Raises:
            TypeError: If the model is not of the specified type.
        """

        hf_config = self.model_config.hf_config
        if not isinstance(hf_config, hf_config_type):
            raise TypeError("Invalid type of HuggingFace config. "
                            f"Expected type: {hf_config_type}, but "
                            f"found type: {type(hf_config)}")

        return hf_config

    def get_hf_image_processor_config(self) -> Dict[str, Any]:
        """
        Get the HuggingFace image processor configuration of the model.
        """

        return self.model_config.hf_image_processor_config


N = TypeVar("N", bound=Type[nn.Module])


class DummyDataFactory(Protocol):

    def __call__(
        self,
        ctx: InputContext,
        seq_len: int,
        mm_counts: Mapping[str, int],
        **mm_processor_kwargs: Any,
    ) -> Tuple["SequenceData", Optional["MultiModalDataDict"]]:
        """
        Create dummy data to be inputted into the model.

        Note:
            :data:`InputProcessor` is not applied to the dummy data.

            The :code:`mm_processor_kwargs` are overrides provided at
            initialization time to values in the config whose values
            may affect the number of tokens per instance.
        """
        ...


class InputRegistry:
    """
    A registry to dispatch data processing
    according to the target model.
    """

    def __init__(self) -> None:
        self._dummy_factories_by_model_type: Dict[Type[nn.Module],
                                                  DummyDataFactory] = {}
        self._dummy_encoder_factories_by_model_type: Dict[
            Type[nn.Module], DummyDataFactory] = {}
        self._input_processors_by_model_type = {}

    def _default_dummy_data_factory(
        self,
        ctx: InputContext,
        seq_len: int,
    ) -> "SequenceData":
        """
        The default dummy data factory represents the longest possible text
        that can be inputted to the model.

        Note:
            :data:`InputProcessor` is not applied to the dummy data.
        """
        # Avoid circular import
        from light_vllm.decoding.schema.sequence import SequenceData

        dummy_seq_data = SequenceData.from_token_counts((0, seq_len))

        return dummy_seq_data

    def register_dummy_data(self, factory: DummyDataFactory):
        """
        Register a dummy data factory to a model class.

        During memory profiling, the provided function is invoked to create
        dummy data to be inputted into the model. The resulting memory usage
        should be an upper bound of what the model would use at inference time.
        """

        def wrapper(model_cls: N) -> N:
            if model_cls in self._dummy_factories_by_model_type:
                logger.warning(
                    "Model class %s already has dummy data "
                    "registered to %s. It is overwritten by the new one.",
                    model_cls, self)

            self._dummy_factories_by_model_type[model_cls] = factory

            return model_cls

        return wrapper

    def _get_dummy_data_factory(self, model_cls: Type[nn.Module]):
        return self._dummy_factories_by_model_type \
            .get(model_cls, self._default_dummy_data_factory)

    def dummy_data_for_profiling(
        self,
        model_config: "ModelConfig",
        seq_len: int,
        is_encoder_data: bool = False,
    ) -> "SequenceData":
        """
        Create dummy data for profiling the memory usage of a model.

        The model is identified by ``model_config``.

        See also:
            :ref:`enabling_multimodal_inputs`

        Note:
            This should be called after
            :meth:`~MultiModalRegistry.init_mm_limits_per_prompt`.
        """
        # Avoid circular import
        from light_vllm.core.loader.utils import get_model_architecture

        model_cls, _ = get_model_architecture(model_config)
        seq_data = self._default_dummy_data_factory(InputContext(model_config),
                                                    seq_len)

        # Having more tokens is over-conservative but otherwise fine
        num_tokens = seq_data.prompt_token_ids
        if len(num_tokens) < seq_len:
            if is_encoder_data:
                print_warning_once(
                    f"Expected at least {seq_len} dummy encoder tokens for "
                    f"profiling, but found {len(num_tokens)} tokens instead.")
            else:
                raise AssertionError(
                    f"Expected at least {seq_len} dummy tokens for profiling, "
                    f"but found {len(num_tokens)} tokens instead.")

        return seq_data

    def _default_input_processor(self, ctx: InputContext, inputs):
        """The default input processor is a no-op."""
        return inputs

    def register_input_processor(self, processor):
        """
        Register an input processor to a model class.

        The provided function is invoked on each input to the model. This
        happens before :meth:`~vllm.multimodal.MultiModalRegistry.map_input`.

        See also:
            :ref:`input_processing_pipeline`
        """

        def wrapper(model_cls: N) -> N:
            if model_cls in self._input_processors_by_model_type:
                logger.warning(
                    "Model class %s already has input processor "
                    "registered to %s. It is overwritten by the new one.",
                    model_cls, self)

            self._input_processors_by_model_type[model_cls] = processor

            return model_cls

        return wrapper

    def _get_model_input_processor(self, model_cls: Type[nn.Module]):
        return self._input_processors_by_model_type \
            .get(model_cls, self._default_input_processor)

    def process_input(self, model_config: "ModelConfig", inputs):
        """
        Apply an input processor to an instance of model inputs.

        The model is identified by ``model_config``.

        See also:
            :ref:`input_processing_pipeline`
        """
        # Avoid circular import
        from light_vllm.core.loader.utils import get_model_architecture

        model_cls, _ = get_model_architecture(model_config)
        processor = self._get_model_input_processor(model_cls)

        return processor(InputContext(model_config), inputs)

    def create_input_processor(self, model_config: "ModelConfig"):
        """
        Create an input processor (see :meth:`_process_input`) for a
        specific model.
        """
        return functools.partial(self.process_input, model_config)
