
from contextlib import contextmanager
from typing import (TYPE_CHECKING, Type, Union, ClassVar, Dict, Iterable, List, Optional)
from typing import Sequence as GenericSequence
from light_vllm.logger import init_logger

from light_vllm.wde.core.schema.engine_io import Params, Inputs, RequestOutput
from light_vllm.wde.core.workflow import Workflow
from light_vllm.wde.core.arg_utils import EngineArgs
from light_vllm.wde.core.config import EngineConfig


logger = init_logger(__name__)
_O = RequestOutput


def lazy_import(module):
    module_name, class_name = module.split(":")
    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class LLMEngine:
    DO_VALIDATE_OUTPUT: ClassVar[bool] = False
    """A flag to toggle whether to validate the type of request output."""

    @classmethod
    @contextmanager
    def enable_output_validation(cls):
        cls.DO_VALIDATE_OUTPUT = True

        yield

        cls.DO_VALIDATE_OUTPUT = False

    @classmethod
    def validate_output(
            cls,
            output: object,
            output_type: Type[_O],
    ) -> _O:
        do_validate = cls.DO_VALIDATE_OUTPUT

        if ((TYPE_CHECKING or do_validate)
                and not isinstance(output, output_type)):
            raise TypeError(f"Expected output of type {output_type}, "
                            f"but found type {type(output)}")

        return output

    @classmethod
    def validate_outputs(
            cls,
            outputs: GenericSequence[object],
            output_type: Type[_O],
    ) -> List[_O]:
        do_validate = cls.DO_VALIDATE_OUTPUT

        outputs_: List[_O]
        if TYPE_CHECKING or do_validate:
            outputs_ = []
            for output in outputs:
                if not isinstance(output, output_type):
                    raise TypeError(f"Expected output of type {output_type}, "
                                    f"but found type {type(output)}")

                outputs_.append(output)
        else:
            outputs_ = outputs

        return outputs_

    def __init__(self, engine_config: EngineConfig, workflow: Workflow) -> None:
        self.engine_config = engine_config
        self.engine_config.log_config()

        self.workflow = workflow
        self.attn_backend = lazy_import(self.workflow.GetAttnBackend).from_engine(self)
        self.executor = lazy_import(self.workflow.Executor).from_engine(self)
        self.tokenizer = lazy_import(self.workflow.Tokenizer).from_engine(self)
        self.model_inputs_builder = lazy_import(self.workflow.ModelInputBuilder).from_engine(self)

        if hasattr(self.executor, "initialize_kv_caches"):
            self.executor.initialize_kv_caches(self)

        self.input_processor = lazy_import(self.workflow.InputProcessor).from_engine(self)
        self.request_processor = lazy_import(self.workflow.RequestProcessor).from_engine(self)
        self.scheduler = lazy_import(self.workflow.Scheduler).from_engine(self)
        self.output_processor = lazy_import(self.workflow.OutputProcessor).from_engine(self)

    @classmethod
    def from_engine_args(
            cls,
            engine_args: Union[Dict, EngineArgs]
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""

        from light_vllm.models.transformers_utils.config import get_config
        from light_vllm.wde.core.loader.utils import get_model_workflow

        if isinstance(engine_args, EngineArgs):
            engine_args = engine_args.to_dict()

        hf_config = get_config(engine_args["model"],
                               engine_args.get("trust_remote_code", False),
                               engine_args.get("revision", None),
                               engine_args.get("code_revision", None))

        workflow_class = get_model_workflow(hf_config)
        workflow = lazy_import(workflow_class)

        engine_args = lazy_import(workflow.EngineArgs)(**engine_args)

        engine_config = engine_args.create_engine_config()
        engine = cls(engine_config, workflow)
        return engine

    def add_request(self,
                    request_id: str,
                    inputs: Optional[Union[str, Inputs]] = None,
                    params: Optional[Params] = None,
                    arrival_time: Optional[float] = None) -> None:
        request = self.input_processor(request_id, inputs, params, arrival_time)
        self.scheduler.add_request(request)

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        self.scheduler.abort_request(request_id)

    def step(self) -> List[RequestOutput]:
        scheduler_output = self.scheduler.schedule()
        if scheduler_output.is_empty():
            return []

        executor_input = self.model_inputs_builder(scheduler_output)
        executor_output = self.executor.execute_model(executor_input)
        request_outputs = self.output_processor(scheduler_output, executor_output)
        self.scheduler.free_finished_request()

        request_outputs = self.scheduler.remove_abort_request(request_outputs)
        return request_outputs

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_requests()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_requests()

    def __reduce__(self):
        # This is to ensure that the LLMEngine is not referenced in
        # the closure used to initialize Ray worker actors
        raise RuntimeError("LLMEngine should not be pickled!")

    def __del__(self):
        # Shutdown model executor when engine is garbage collected
        # Use getattr since __init__ can fail before the field is set
        if Executor := getattr(self, "Executor", None):
            Executor.shutdown()
