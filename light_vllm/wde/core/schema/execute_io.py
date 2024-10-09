from abc import ABC
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelInput(ABC):
    pass


@dataclass
class WorkerInput:
    pass


@dataclass
class ExecuteInput(ABC):
    worker_input: Optional[WorkerInput]
    model_input: Optional[ModelInput]


class ExecuteOutput(ABC):
    pass
