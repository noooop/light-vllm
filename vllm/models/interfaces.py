from typing import (ClassVar, Dict, List, Literal, Optional, Protocol, Type,
                    Union, overload, runtime_checkable)

from typing_extensions import TypeGuard

from vllm.config import SchedulerConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


@runtime_checkable
class HasInnerState(Protocol):
    """The interface required for all models that has inner state."""

    has_inner_state: ClassVar[Literal[True]] = True
    """
        A flag that indicates this model has inner state.
        Models that has inner state usually need access to the scheduler_config
        for max_num_seqs ,etc... (Currently only used by Jamba)
    """

    def __init__(self,
                 *,
                 scheduler_config: Optional[SchedulerConfig] = None) -> None:
        ...


@runtime_checkable
class _HasInnerStateType(Protocol):
    has_inner_state: ClassVar[Literal[True]]

    def __init__(self,
                 *,
                 scheduler_config: Optional[SchedulerConfig] = None) -> None:
        ...


@overload
def has_inner_state(model: object) -> TypeGuard[HasInnerState]:
    ...


@overload
def has_inner_state(model: Type[object]) -> TypeGuard[Type[HasInnerState]]:
    ...


def has_inner_state(
    model: Union[Type[object], object]
) -> Union[TypeGuard[Type[HasInnerState]], TypeGuard[HasInnerState]]:
    if isinstance(model, type):
        return isinstance(model, _HasInnerStateType)

    return isinstance(model, HasInnerState)
