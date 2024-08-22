from contextlib import contextmanager
from typing import (TYPE_CHECKING, Any, ClassVar, Dict, Iterable, List,
                    Mapping, Optional)
from typing import Sequence as GenericSequence
from typing import Type, Union

from light_vllm.version import __version__ as VLLM_VERSION

from light_vllm.engine.arg_utils import EngineArgs
from light_vllm.task.base.schema.outputs import RequestOutput

from light_vllm.utils import Counter
from light_vllm.logger import init_logger


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

    def __init__(self, engine_config) -> None:
        # config
        self.engine_config = engine_config
        self.model_config = engine_config.model_config
        self.cache_config = engine_config.cache_config
        self.scheduler_config = engine_config.scheduler_config
        self.device_config = engine_config.device_config
        self.load_config = engine_config.load_config
        self._log_config()

        # workflow
        self.workflow = lazy_import(self.model_config.workflow)

        # executor
        self.executor = lazy_import(self.workflow.Executor)(
            model_config=self.model_config,
            cache_config=self.cache_config,
            scheduler_config=self.scheduler_config,
            device_config=self.device_config,
            load_config=self.load_config,
            workflow=self.workflow
        )

        # model_pre_processor
        self.model_pre_processor = lazy_import(self.workflow.ModelPreProcessor)(self.device_config,
                                                                                self.model_config,
                                                                                self.scheduler_config,
                                                                                self.cache_config,
                                                                                attn_backend=self.executor.driver_worker.model_runner.attn_backend,
                                                                                cuda_graph=self.executor.driver_worker.model_runner.cuda_graph)

        self._initialize_kv_caches()

        # scheduler
        self.scheduler = lazy_import(self.workflow.Scheduler)(self.scheduler_config, self.cache_config)

        # tokenizer
        self.tokenizer = lazy_import(self.workflow.Tokenizer)(**self._init_tokenizer_kwargs())

        # input_processor
        self.seq_counter = Counter()
        self.input_processor = lazy_import(self.workflow.InputProcessor)(self.model_config,
                                                                         self.cache_config,
                                                                         self.tokenizer,
                                                                         self.seq_counter)

        # output_processor
        self.output_processor = lazy_import(self.workflow.OutputProcessor)(self.scheduler_config,
                                                                           self.scheduler,
                                                                           self.tokenizer,
                                                                           self.seq_counter)

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
            self.model_config.model,
            self.model_config.tokenizer,
            self.model_config.skip_tokenizer_init,
            self.model_config.tokenizer_mode,
            self.model_config.revision,
            self.model_config.rope_scaling,
            self.model_config.rope_theta,
            self.model_config.tokenizer_revision,
            self.model_config.trust_remote_code,
            self.model_config.dtype,
            self.model_config.max_model_len,
            self.load_config.download_dir,
            self.load_config.load_format,
            self.model_config.quantization,
            self.model_config.enforce_eager,
            self.cache_config.cache_dtype,
            self.model_config.quantization_param_path,
            self.device_config.device,
            self.model_config.seed,
            self.model_config.served_model_name,
            self.scheduler_config.use_v2_block_manager,
            self.cache_config.enable_prefix_caching,
        )

    def _initialize_kv_caches(self) -> None:
        """Initialize the KV cache in the worker(s).

        The workers will determine the number of blocks in both the GPU cache
        and the swap CPU cache.
        """
        self.executor.driver_worker.model_runner.prepare_model_input = self.model_pre_processor.prepare_model_input

        num_gpu_blocks, num_cpu_blocks = (
            self.executor.determine_num_available_blocks())

        if self.cache_config.num_gpu_blocks_override is not None:
            num_gpu_blocks_override = self.cache_config.num_gpu_blocks_override
            logger.info(
                "Overriding num_gpu_blocks=%d with "
                "num_gpu_blocks_override=%d", num_gpu_blocks,
                num_gpu_blocks_override)
            num_gpu_blocks = num_gpu_blocks_override

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self.executor.initialize_cache(num_gpu_blocks, num_cpu_blocks)

        del self.executor.driver_worker.model_runner.prepare_model_input

    def _init_tokenizer_kwargs(self) -> Dict:
        init_kwargs = dict(tokenizer_name=self.model_config.tokenizer,
                           tokenizer_mode=self.model_config.tokenizer_mode,
                           trust_remote_code=self.model_config.trust_remote_code,
                           revision=self.model_config.tokenizer_revision)

        return init_kwargs

    @classmethod
    def from_engine_args(
            cls,
            engine_args: EngineArgs = None,
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""

        engine_config = engine_args.create_engine_config()
        engine = cls(engine_config)
        return engine

    def add_request(self, *args, **kwargs) -> None:
        request = self.input_processor(*args, **kwargs)
        self.scheduler.add_seq_group(request["input"])

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        self.scheduler.abort_seq_group(request_id)

    def step(self) -> List[RequestOutput]:
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()

        if not scheduler_outputs.is_empty():
            execute_input = self.model_pre_processor(seq_group_metadata_list, scheduler_outputs)
            execute_output = self.executor.execute_model(execute_input)
        else:
            execute_output = []

        request_outputs = self.output_processor(execute_output,
                                                scheduler_outputs.scheduled_seq_groups,
                                                scheduler_outputs.ignored_seq_groups,
                                                seq_group_metadata_list)

        self.scheduler.free_finished_seq_groups()

        return request_outputs

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_seq_groups()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_seqs()

    def __reduce__(self):
        # This is to ensure that the LLMEngine is not referenced in
        # the closure used to initialize Ray worker actors
        raise RuntimeError("LLMEngine should not be pickled!")

    def __del__(self):
        # Shutdown model executor when engine is garbage collected
        # Use getattr since __init__ can fail before the field is set
        if Executor := getattr(self, "Executor", None):
            Executor.shutdown()
