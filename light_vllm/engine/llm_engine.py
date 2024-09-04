
from contextlib import contextmanager
from typing import (TYPE_CHECKING, Type, Union, ClassVar, Dict, Iterable, List, Optional)
from typing import Sequence as GenericSequence

from light_vllm.version import __version__ as VLLM_VERSION
from light_vllm.logger import init_logger

from light_vllm.task.base.schema.inputs import Params, Prompt
from light_vllm.task.base.schema.outputs import RequestOutput
from light_vllm.task.base.workflow import Workflow
from light_vllm.task.base.arg_utils import EngineArgs
from light_vllm.task.base.config import EngineConfig


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
        # config
        self.engine_config = engine_config
        self._log_config()

        # workflow
        self.workflow = workflow

        # executor
        self.executor = lazy_import(self.workflow.Executor).from_engine(self)

        # tokenizer
        self.tokenizer = lazy_import(self.workflow.Tokenizer).from_engine(self)

        # model_pre_processor
        self.model_pre_processor = lazy_import(self.workflow.ModelPreProcessor).from_engine(self)

        if hasattr(self.engine_config, "cache_config") and self.engine_config.cache_config is not None:
            self._initialize_kv_caches()

        # input_processor
        self.input_processor = lazy_import(self.workflow.InputProcessor).from_engine(self)
        self.request_processor = lazy_import(self.workflow.RequestProcessor).from_engine(self)

        # scheduler
        self.scheduler = lazy_import(self.workflow.Scheduler).from_engine(self)

        # output_processor
        self.output_processor = lazy_import(self.workflow.OutputProcessor).from_engine(self)

    def _log_config(self):
        logger.info(
            "Initializing an LLM engine (v%s) with config: "
            "model=%r, tokenizer=%r, "
            "skip_tokenizer_init=%s, tokenizer_mode=%s, revision=%s, "
            "rope_scaling=%r, rope_theta=%r, tokenizer_revision=%s, "
            "trust_remote_code=%s, dtype=%s, max_seq_len=%d, "
            "download_dir=%r, load_format=%s, "
            "quantization=%s, "
            "enforce_eager=%s, kv_cache_dtype=%s, "
            "quantization_param_path=%s, device_config=%s, "
            "seed=%d, served_model_name=%s, use_v2_block_manager=%s, "
            "enable_prefix_caching=%s)",
            VLLM_VERSION,
            self.engine_config.model_config.model,
            self.engine_config.model_config.tokenizer,
            self.engine_config.model_config.skip_tokenizer_init,
            self.engine_config.model_config.tokenizer_mode,
            self.engine_config.model_config.revision,
            self.engine_config.model_config.rope_scaling,
            self.engine_config.model_config.rope_theta,
            self.engine_config.model_config.tokenizer_revision,
            self.engine_config.model_config.trust_remote_code,
            self.engine_config.model_config.dtype,
            self.engine_config.model_config.max_model_len,
            self.engine_config.load_config.download_dir,
            self.engine_config.load_config.load_format,
            self.engine_config.model_config.quantization,
            self.engine_config.model_config.enforce_eager,
            "None" if self.engine_config.cache_config is None else self.engine_config.cache_config.cache_dtype,
            self.engine_config.model_config.quantization_param_path,
            self.engine_config.device_config.device,
            self.engine_config.model_config.seed,
            self.engine_config.model_config.served_model_name,
            self.engine_config.scheduler_config.use_v2_block_manager,
            "None" if self.engine_config.cache_config is None else self.engine_config.cache_config.enable_prefix_caching,
        )

    def _initialize_kv_caches(self) -> None:
        """Initialize the KV cache in the worker(s).

        The workers will determine the number of blocks in both the GPU cache
        and the swap CPU cache.
        """
        self.executor.driver_worker.model_runner.prepare_model_input = self.model_pre_processor.prepare_model_input

        num_gpu_blocks, num_cpu_blocks = (
            self.executor.determine_num_available_blocks())

        if self.engine_config.cache_config.num_gpu_blocks_override is not None:
            num_gpu_blocks_override = self.engine_config.cache_config.num_gpu_blocks_override
            logger.info(
                "Overriding num_gpu_blocks=%d with "
                "num_gpu_blocks_override=%d", num_gpu_blocks,
                num_gpu_blocks_override)
            num_gpu_blocks = num_gpu_blocks_override

        self.engine_config.cache_config.num_gpu_blocks = num_gpu_blocks
        self.engine_config.cache_config.num_cpu_blocks = num_cpu_blocks

        self.executor.initialize_cache(num_gpu_blocks, num_cpu_blocks)

        del self.executor.driver_worker.model_runner.prepare_model_input

    @classmethod
    def from_engine_args(
            cls,
            engine_args: Union[Dict, EngineArgs]
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""

        from light_vllm.models.transformers_utils.config import get_config
        from light_vllm.models.loader.utils import get_model_workflow

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
                    prompt: Optional[Union[str, Prompt]] = None,
                    params: Optional[Params] = None,
                    arrival_time: Optional[float] = None) -> None:
        request = self.input_processor(request_id, prompt, params, arrival_time)
        self.scheduler.add_request(request)

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        self.scheduler.abort_request(request_id)

    def step(self) -> List[RequestOutput]:
        scheduler_outputs = self.scheduler.schedule()

        if scheduler_outputs.is_empty():
            return []

        execute_input = self.model_pre_processor(scheduler_outputs)
        execute_output = self.executor.execute_model(execute_input)
        request_outputs = self.output_processor(scheduler_outputs, execute_output)
        self.scheduler.free_finished_request()

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
