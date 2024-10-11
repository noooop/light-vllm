from typing import Dict


class Workflow:
    EngineArgs: str
    Scheduler: str
    GetAttnBackend: str
    Tokenizer: str = "light_vllm.core.inputs.tokenizer:Tokenizer"
    InputProcessor: str
    RequestProcessor: str
    OutputProcessor: str
    ModelInputBuilder: str
    Executor: str
    Worker: str

    @classmethod
    def from_engine(cls, engine):
        return cls()

    @classmethod
    def from_engine_args(cls, engine_args: Dict):
        return cls