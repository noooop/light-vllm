import argparse
import dataclasses
import json
from dataclasses import dataclass
from typing import List, Optional, Union

from light_vllm.layers.quantization import QUANTIZATION_METHODS
from light_vllm.logger import init_logger
from light_vllm.utils import FlexibleArgumentParser
from light_vllm.wde.core.arg_utils import EngineArgs
from light_vllm.wde.core.config import (DeviceConfig, LoadConfig,
                                        filter_unexpected_fields)
from light_vllm.wde.encode_only.config import (EncodeOnlyEngineConfig,
                                               EncodeOnlyModelConfig,
                                               EncodeOnlySchedulerConfig)

logger = init_logger(__name__)


def nullable_str(val: str):
    if not val or val == "None":
        return None
    return val


@filter_unexpected_fields
@dataclass
class EncodeOnlyEngineArgs(EngineArgs):
    """Arguments for vLLM engine."""
    model: str
    served_model_name: Optional[Union[List[str]]] = None
    tokenizer: Optional[str] = None
    skip_tokenizer_init: bool = False
    tokenizer_mode: str = 'auto'
    trust_remote_code: bool = False
    download_dir: Optional[str] = None
    load_format: str = 'auto'
    dtype: str = 'auto'
    kv_cache_dtype: str = 'auto'
    quantization_param_path: Optional[str] = None
    disable_sliding_window: bool = False
    seed: int = 0
    max_model_len: Optional[int] = None
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: int = 256
    max_num_on_the_fly: int = 3
    scheduling: str = "async"
    disable_log_stats: bool = False
    revision: Optional[str] = None
    code_revision: Optional[str] = None
    rope_scaling: Optional[dict] = None
    rope_theta: Optional[float] = None
    tokenizer_revision: Optional[str] = None
    quantization: Optional[str] = None
    disable_custom_all_reduce: bool = False
    device: str = 'auto'
    model_loader_extra_config: Optional[dict] = None
    ignore_patterns: Optional[Union[str, List[str]]] = None

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = self.model

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """Shared CLI arguments for vLLM engine."""

        # Model arguments
        parser.add_argument(
            '--model',
            type=str,
            default='facebook/opt-125m',
            help='Name or path of the huggingface model to use.')
        parser.add_argument(
            '--tokenizer',
            type=nullable_str,
            default=EngineArgs.tokenizer,
            help='Name or path of the huggingface tokenizer to use. '
            'If unspecified, model name or path will be used.')
        parser.add_argument(
            '--skip-tokenizer-init',
            action='store_true',
            help='Skip initialization of tokenizer and detokenizer')
        parser.add_argument(
            '--revision',
            type=nullable_str,
            default=None,
            help='The specific model version to use. It can be a branch '
            'name, a tag name, or a commit id. If unspecified, will use '
            'the default version.')
        parser.add_argument(
            '--code-revision',
            type=nullable_str,
            default=None,
            help='The specific revision to use for the model code on '
            'Hugging Face Hub. It can be a branch name, a tag name, or a '
            'commit id. If unspecified, will use the default version.')
        parser.add_argument(
            '--tokenizer-revision',
            type=nullable_str,
            default=None,
            help='Revision of the huggingface tokenizer to use. '
            'It can be a branch name, a tag name, or a commit id. '
            'If unspecified, will use the default version.')
        parser.add_argument(
            '--tokenizer-mode',
            type=str,
            default=EngineArgs.tokenizer_mode,
            choices=['auto', 'slow'],
            help='The tokenizer mode.\n\n* "auto" will use the '
            'fast tokenizer if available.\n* "slow" will '
            'always use the slow tokenizer.')
        parser.add_argument('--trust-remote-code',
                            action='store_true',
                            help='Trust remote code from huggingface.')
        parser.add_argument('--download-dir',
                            type=nullable_str,
                            default=EngineArgs.download_dir,
                            help='Directory to download and load the weights, '
                            'default to the default cache dir of '
                            'huggingface.')
        parser.add_argument(
            '--load-format',
            type=str,
            default=EngineArgs.load_format,
            choices=[
                'auto', 'pt', 'safetensors', 'npcache', 'dummy', 'tensorizer',
                'bitsandbytes'
            ],
            help='The format of the model weights to load.\n\n'
            '* "auto" will try to load the weights in the safetensors format '
            'and fall back to the pytorch bin format if safetensors format '
            'is not available.\n'
            '* "pt" will load the weights in the pytorch bin format.\n'
            '* "safetensors" will load the weights in the safetensors format.\n'
            '* "npcache" will load the weights in pytorch format and store '
            'a numpy cache to speed up the loading.\n'
            '* "dummy" will initialize the weights with random values, '
            'which is mainly for profiling.\n'
            '* "tensorizer" will load the weights using tensorizer from '
            'CoreWeave. See the Tensorize vLLM Model script in the Examples '
            'section for more information.\n'
            '* "bitsandbytes" will load the weights using bitsandbytes '
            'quantization.\n')
        parser.add_argument(
            '--dtype',
            type=str,
            default=EngineArgs.dtype,
            choices=[
                'auto', 'half', 'float16', 'bfloat16', 'float', 'float32'
            ],
            help='Data type for model weights and activations.\n\n'
            '* "auto" will use FP16 precision for FP32 and FP16 models, and '
            'BF16 precision for BF16 models.\n'
            '* "half" for FP16. Recommended for AWQ quantization.\n'
            '* "float16" is the same as "half".\n'
            '* "bfloat16" for a balance between precision and range.\n'
            '* "float" is shorthand for FP32 precision.\n'
            '* "float32" for FP32 precision.')
        parser.add_argument(
            '--kv-cache-dtype',
            type=str,
            choices=['auto', 'fp8', 'fp8_e5m2', 'fp8_e4m3'],
            default=EngineArgs.kv_cache_dtype,
            help='Data type for kv cache storage. If "auto", will use model '
            'data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. '
            'ROCm (AMD GPU) supports fp8 (=fp8_e4m3)')
        parser.add_argument(
            '--quantization-param-path',
            type=nullable_str,
            default=None,
            help='Path to the JSON file containing the KV cache '
            'scaling factors. This should generally be supplied, when '
            'KV cache dtype is FP8. Otherwise, KV cache scaling factors '
            'default to 1.0, which may cause accuracy issues. '
            'FP8_E5M2 (without scaling) is only supported on cuda version'
            'greater than 11.8. On ROCm (AMD GPU), FP8_E4M3 is instead '
            'supported for common inference criteria.')
        parser.add_argument('--max-model-len',
                            type=int,
                            default=EngineArgs.max_model_len,
                            help='Model context length. If unspecified, will '
                            'be automatically derived from the model config.')
        parser.add_argument(
            '--guided-decoding-backend',
            type=str,
            default='outlines',
            choices=['outlines', 'lm-format-enforcer'],
            help='Which engine will be used for guided decoding'
            ' (JSON schema / regex etc) by default. Currently support '
            'https://github.com/outlines-dev/outlines and '
            'https://github.com/noamgat/lm-format-enforcer.'
            ' Can be overridden per request via guided_decoding_backend'
            ' parameter.')
        # Parallel arguments
        parser.add_argument(
            '--distributed-executor-backend',
            choices=['ray', 'mp'],
            default=EngineArgs.distributed_executor_backend,
            help='Backend to use for distributed serving. When more than 1 GPU '
            'is used, will be automatically set to "ray" if installed '
            'or "mp" (multiprocessing) otherwise.')
        parser.add_argument(
            '--worker-use-ray',
            action='store_true',
            help='Deprecated, use --distributed-executor-backend=ray.')
        parser.add_argument('--pipeline-parallel-size',
                            '-pp',
                            type=int,
                            default=EngineArgs.pipeline_parallel_size,
                            help='Number of pipeline stages.')
        parser.add_argument('--tensor-parallel-size',
                            '-tp',
                            type=int,
                            default=EngineArgs.tensor_parallel_size,
                            help='Number of tensor parallel replicas.')
        parser.add_argument(
            '--max-parallel-loading-workers',
            type=int,
            default=EngineArgs.max_parallel_loading_workers,
            help='Load model sequentially in multiple batches, '
            'to avoid RAM OOM when using tensor '
            'parallel and large models.')
        parser.add_argument(
            '--ray-workers-use-nsight',
            action='store_true',
            help='If specified, use nsight to profile Ray workers.')
        # KV cache arguments
        parser.add_argument('--block-size',
                            type=int,
                            default=EngineArgs.block_size,
                            choices=[8, 16, 32],
                            help='Token block size for contiguous chunks of '
                            'tokens.')

        parser.add_argument('--enable-prefix-caching',
                            action='store_true',
                            help='Enables automatic prefix caching.')
        parser.add_argument('--disable-sliding-window',
                            action='store_true',
                            help='Disables sliding window, '
                            'capping to sliding window size')
        parser.add_argument('--use-v2-block-manager',
                            action='store_true',
                            help='Use BlockSpaceMangerV2.')
        parser.add_argument(
            '--num-lookahead-slots',
            type=int,
            default=EngineArgs.num_lookahead_slots,
            help='Experimental scheduling config necessary for '
            'speculative decoding. This will be replaced by '
            'speculative config in the future; it is present '
            'to enable correctness tests until then.')

        parser.add_argument('--seed',
                            type=int,
                            default=EngineArgs.seed,
                            help='Random seed for operations.')
        parser.add_argument('--swap-space',
                            type=int,
                            default=EngineArgs.swap_space,
                            help='CPU swap space size (GiB) per GPU.')
        parser.add_argument(
            '--cpu-offload-gb',
            type=float,
            default=0,
            help='The space in GiB to offload to CPU, per GPU. '
            'Default is 0, which means no offloading. Intuitively, '
            'this argument can be seen as a virtual way to increase '
            'the GPU memory size. For example, if you have one 24 GB '
            'GPU and set this to 10, virtually you can think of it as '
            'a 34 GB GPU. Then you can load a 13B model with BF16 weight,'
            'which requires at least 26GB GPU memory. Note that this '
            'requires fast CPU-GPU interconnect, as part of the model is'
            'loaded from CPU memory to GPU memory on the fly in each '
            'model forward pass.')
        parser.add_argument(
            '--gpu-memory-utilization',
            type=float,
            default=EngineArgs.gpu_memory_utilization,
            help='The fraction of GPU memory to be used for the model '
            'executor, which can range from 0 to 1. For example, a value of '
            '0.5 would imply 50%% GPU memory utilization. If unspecified, '
            'will use the default value of 0.9.')
        parser.add_argument(
            '--num-gpu-blocks-override',
            type=int,
            default=None,
            help='If specified, ignore GPU profiling result and use this number'
            'of GPU blocks. Used for testing preemption.')
        parser.add_argument('--max-num-batched-tokens',
                            type=int,
                            default=EngineArgs.max_num_batched_tokens,
                            help='Maximum number of batched tokens per '
                            'iteration.')
        parser.add_argument('--max-num-seqs',
                            type=int,
                            default=EngineArgs.max_num_seqs,
                            help='Maximum number of sequences per iteration.')
        parser.add_argument(
            '--max-logprobs',
            type=int,
            default=EngineArgs.max_logprobs,
            help=('Max number of log probs to return logprobs is specified in'
                  ' SamplingParams.'))
        parser.add_argument('--disable-log-stats',
                            action='store_true',
                            help='Disable logging statistics.')
        # Quantization settings.
        parser.add_argument('--quantization',
                            '-q',
                            type=nullable_str,
                            choices=[*QUANTIZATION_METHODS, None],
                            default=EngineArgs.quantization,
                            help='Method used to quantize the weights. If '
                            'None, we first check the `quantization_config` '
                            'attribute in the model config file. If that is '
                            'None, we assume the model weights are not '
                            'quantized and use `dtype` to determine the data '
                            'type of the weights.')
        parser.add_argument('--rope-scaling',
                            default=None,
                            type=json.loads,
                            help='RoPE scaling configuration in JSON format. '
                            'For example, {"type":"dynamic","factor":2.0}')
        parser.add_argument('--rope-theta',
                            default=None,
                            type=float,
                            help='RoPE theta. Use with `rope_scaling`. In '
                            'some cases, changing the RoPE theta improves the '
                            'performance of the scaled model.')
        parser.add_argument('--enforce-eager',
                            action='store_true',
                            help='Always use eager-mode PyTorch. If False, '
                            'will use eager mode and CUDA graph in hybrid '
                            'for maximal performance and flexibility.')
        parser.add_argument('--max-context-len-to-capture',
                            type=int,
                            default=EngineArgs.max_context_len_to_capture,
                            help='Maximum context length covered by CUDA '
                            'graphs. When a sequence has context length '
                            'larger than this, we fall back to eager mode. '
                            '(DEPRECATED. Use --max-seq-len-to-capture instead'
                            ')')
        parser.add_argument('--max-seq-len-to-capture',
                            type=int,
                            default=EngineArgs.max_seq_len_to_capture,
                            help='Maximum sequence length covered by CUDA '
                            'graphs. When a sequence has context length '
                            'larger than this, we fall back to eager mode.')
        parser.add_argument('--disable-custom-all-reduce',
                            action='store_true',
                            default=EngineArgs.disable_custom_all_reduce,
                            help='See ParallelConfig.')
        parser.add_argument('--tokenizer-pool-size',
                            type=int,
                            default=EngineArgs.tokenizer_pool_size,
                            help='Size of tokenizer pool to use for '
                            'asynchronous tokenization. If 0, will '
                            'use synchronous tokenization.')
        parser.add_argument('--tokenizer-pool-type',
                            type=str,
                            default=EngineArgs.tokenizer_pool_type,
                            help='Type of tokenizer pool to use for '
                            'asynchronous tokenization. Ignored '
                            'if tokenizer_pool_size is 0.')
        parser.add_argument('--tokenizer-pool-extra-config',
                            type=nullable_str,
                            default=EngineArgs.tokenizer_pool_extra_config,
                            help='Extra config for tokenizer pool. '
                            'This should be a JSON string that will be '
                            'parsed into a dictionary. Ignored if '
                            'tokenizer_pool_size is 0.')
        # LoRA related configs
        parser.add_argument('--enable-lora',
                            action='store_true',
                            help='If True, enable handling of LoRA adapters.')
        parser.add_argument('--max-loras',
                            type=int,
                            default=EngineArgs.max_loras,
                            help='Max number of LoRAs in a single batch.')
        parser.add_argument('--max-lora-rank',
                            type=int,
                            default=EngineArgs.max_lora_rank,
                            help='Max LoRA rank.')
        parser.add_argument(
            '--lora-extra-vocab-size',
            type=int,
            default=EngineArgs.lora_extra_vocab_size,
            help=('Maximum size of extra vocabulary that can be '
                  'present in a LoRA adapter (added to the base '
                  'model vocabulary).'))
        parser.add_argument(
            '--lora-dtype',
            type=str,
            default=EngineArgs.lora_dtype,
            choices=['auto', 'float16', 'bfloat16', 'float32'],
            help=('Data type for LoRA. If auto, will default to '
                  'base model dtype.'))
        parser.add_argument(
            '--long-lora-scaling-factors',
            type=nullable_str,
            default=EngineArgs.long_lora_scaling_factors,
            help=('Specify multiple scaling factors (which can '
                  'be different from base model scaling factor '
                  '- see eg. Long LoRA) to allow for multiple '
                  'LoRA adapters trained with those scaling '
                  'factors to be used at the same time. If not '
                  'specified, only adapters trained with the '
                  'base model scaling factor are allowed.'))
        parser.add_argument(
            '--max-cpu-loras',
            type=int,
            default=EngineArgs.max_cpu_loras,
            help=('Maximum number of LoRAs to store in CPU memory. '
                  'Must be >= than max_num_seqs. '
                  'Defaults to max_num_seqs.'))
        parser.add_argument(
            '--fully-sharded-loras',
            action='store_true',
            help=('By default, only half of the LoRA computation is '
                  'sharded with tensor parallelism. '
                  'Enabling this will use the fully sharded layers. '
                  'At high sequence length, max rank or '
                  'tensor parallel size, this is likely faster.'))
        parser.add_argument('--enable-prompt-adapter',
                            action='store_true',
                            help='If True, enable handling of PromptAdapters.')
        parser.add_argument('--max-prompt-adapters',
                            type=int,
                            default=EngineArgs.max_prompt_adapters,
                            help='Max number of PromptAdapters in a batch.')
        parser.add_argument('--max-prompt-adapter-token',
                            type=int,
                            default=EngineArgs.max_prompt_adapter_token,
                            help='Max number of PromptAdapters tokens')
        parser.add_argument("--device",
                            type=str,
                            default=EngineArgs.device,
                            choices=[
                                "auto", "cuda", "neuron", "cpu", "openvino",
                                "tpu", "xpu"
                            ],
                            help='Device type for vLLM execution.')

        parser.add_argument(
            '--scheduler-delay-factor',
            type=float,
            default=EngineArgs.scheduler_delay_factor,
            help='Apply a delay (of delay factor multiplied by previous'
            'prompt latency) before scheduling next prompt.')
        parser.add_argument(
            '--enable-chunked-prefill',
            action=StoreBoolean,
            default=EngineArgs.enable_chunked_prefill,
            nargs="?",
            const="True",
            help='If set, the prefill requests can be chunked based on the '
            'max_num_batched_tokens.')

        parser.add_argument(
            '--speculative-model',
            type=nullable_str,
            default=EngineArgs.speculative_model,
            help=
            'The name of the draft model to be used in speculative decoding.')
        parser.add_argument(
            '--num-speculative-tokens',
            type=int,
            default=EngineArgs.num_speculative_tokens,
            help='The number of speculative tokens to sample from '
            'the draft model in speculative decoding.')
        parser.add_argument(
            '--speculative-draft-tensor-parallel-size',
            '-spec-draft-tp',
            type=int,
            default=EngineArgs.speculative_draft_tensor_parallel_size,
            help='Number of tensor parallel replicas for '
            'the draft model in speculative decoding.')

        parser.add_argument(
            '--speculative-max-model-len',
            type=int,
            default=EngineArgs.speculative_max_model_len,
            help='The maximum sequence length supported by the '
            'draft model. Sequences over this length will skip '
            'speculation.')

        parser.add_argument(
            '--speculative-disable-by-batch-size',
            type=int,
            default=EngineArgs.speculative_disable_by_batch_size,
            help='Disable speculative decoding for new incoming requests '
            'if the number of enqueue requests is larger than this value.')

        parser.add_argument(
            '--ngram-prompt-lookup-max',
            type=int,
            default=EngineArgs.ngram_prompt_lookup_max,
            help='Max size of window for ngram prompt lookup in speculative '
            'decoding.')

        parser.add_argument(
            '--ngram-prompt-lookup-min',
            type=int,
            default=EngineArgs.ngram_prompt_lookup_min,
            help='Min size of window for ngram prompt lookup in speculative '
            'decoding.')

        parser.add_argument(
            '--spec-decoding-acceptance-method',
            type=str,
            default=EngineArgs.spec_decoding_acceptance_method,
            choices=['rejection_sampler', 'typical_acceptance_sampler'],
            help='Specify the acceptance method to use during draft token '
            'verification in speculative decoding. Two types of acceptance '
            'routines are supported: '
            '1) RejectionSampler which does not allow changing the '
            'acceptance rate of draft tokens, '
            '2) TypicalAcceptanceSampler which is configurable, allowing for '
            'a higher acceptance rate at the cost of lower quality, '
            'and vice versa.')

        parser.add_argument(
            '--typical-acceptance-sampler-posterior-threshold',
            type=float,
            default=EngineArgs.typical_acceptance_sampler_posterior_threshold,
            help='Set the lower bound threshold for the posterior '
            'probability of a token to be accepted. This threshold is '
            'used by the TypicalAcceptanceSampler to make sampling decisions '
            'during speculative decoding. Defaults to 0.09')

        parser.add_argument(
            '--typical-acceptance-sampler-posterior-alpha',
            type=float,
            default=EngineArgs.typical_acceptance_sampler_posterior_alpha,
            help='A scaling factor for the entropy-based threshold for token '
            'acceptance in the TypicalAcceptanceSampler. Typically defaults '
            'to sqrt of --typical-acceptance-sampler-posterior-threshold '
            'i.e. 0.3')

        parser.add_argument(
            '--disable-logprobs-during-spec-decoding',
            type=bool,
            default=EngineArgs.disable_logprobs_during_spec_decoding,
            help='If set to True, token log probabilities are not returned '
            'during speculative decoding. If set to False, log probabilities '
            'are returned according to the settings in SamplingParams. If '
            'not specified, it defaults to True. Disabling log probabilities '
            'during speculative decoding reduces latency by skipping logprob '
            'calculation in proposal sampling, target sampling, and after '
            'accepted tokens are determined.')

        parser.add_argument('--model-loader-extra-config',
                            type=nullable_str,
                            default=EngineArgs.model_loader_extra_config,
                            help='Extra config for model loader. '
                            'This will be passed to the model loader '
                            'corresponding to the chosen load_format. '
                            'This should be a JSON string that will be '
                            'parsed into a dictionary.')
        parser.add_argument(
            '--ignore-patterns',
            action="append",
            type=str,
            default=[],
            help="The pattern(s) to ignore when loading the model."
            "Default to 'original/**/*' to avoid repeated loading of llama's "
            "checkpoints.")
        parser.add_argument(
            '--preemption-mode',
            type=str,
            default=None,
            help='If \'recompute\', the engine performs preemption by '
            'recomputing; If \'swap\', the engine performs preemption by '
            'block swapping.')

        parser.add_argument(
            "--served-model-name",
            nargs="+",
            type=str,
            default=None,
            help="The model name(s) used in the API. If multiple "
            "names are provided, the server will respond to any "
            "of the provided names. The model name in the model "
            "field of a response will be the first name in this "
            "list. If not specified, the model name will be the "
            "same as the `--model` argument. Noted that this name(s)"
            "will also be used in `model_name` tag content of "
            "prometheus metrics, if multiple names provided, metrics"
            "tag will take the first one.")
        parser.add_argument('--qlora-adapter-name-or-path',
                            type=str,
                            default=None,
                            help='Name or path of the QLoRA adapter.')

        parser.add_argument(
            '--otlp-traces-endpoint',
            type=str,
            default=None,
            help='Target URL to which OpenTelemetry traces will be sent.')

        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args

    def create_engine_config(self) -> EncodeOnlyEngineConfig:
        device_config = DeviceConfig(device=self.device)
        model_config = EncodeOnlyModelConfig(
            model=self.model,
            tokenizer=self.tokenizer,
            tokenizer_mode=self.tokenizer_mode,
            trust_remote_code=self.trust_remote_code,
            dtype=self.dtype,
            seed=self.seed,
            revision=self.revision,
            code_revision=self.code_revision,
            rope_scaling=self.rope_scaling,
            rope_theta=self.rope_theta,
            tokenizer_revision=self.tokenizer_revision,
            max_model_len=self.max_model_len,
            quantization=self.quantization,
            quantization_param_path=self.quantization_param_path,
            disable_sliding_window=self.disable_sliding_window,
            skip_tokenizer_init=self.skip_tokenizer_init,
            served_model_name=self.served_model_name)

        scheduler_config = EncodeOnlySchedulerConfig(
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs,
            max_model_len=model_config.max_model_len,
            max_num_on_the_fly=self.max_num_on_the_fly,
            scheduling=self.scheduling)

        load_config = LoadConfig(
            load_format=self.load_format,
            download_dir=self.download_dir,
            model_loader_extra_config=self.model_loader_extra_config,
            ignore_patterns=self.ignore_patterns,
        )

        return EncodeOnlyEngineConfig(model_config=model_config,
                                      scheduler_config=scheduler_config,
                                      device_config=device_config,
                                      load_config=load_config)
