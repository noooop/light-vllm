from typing import List, Optional, Union

from light_vllm.layers.pooling_params import PoolingParams
from light_vllm.layers.sampling_params import SamplingParams
from light_vllm.logger import init_logger
from light_vllm.lora.request import LoRARequest
from light_vllm.prompt_adapter.request import PromptAdapterRequest

logger = init_logger(__name__)


class RequestLogger:

    def __init__(self, *, max_log_len: Optional[int]) -> None:
        super().__init__()

        self.max_log_len = max_log_len

    def log_inputs(
        self,
        request_id: str,
        prompt: Optional[str],
        prompt_token_ids: Optional[List[int]],
        params: Optional[Union[SamplingParams, PoolingParams]],
        lora_request: Optional[LoRARequest],
        prompt_adapter_request: Optional[PromptAdapterRequest],
    ) -> None:
        max_log_len = self.max_log_len
        if max_log_len is not None:
            if prompt is not None:
                prompt = prompt[:max_log_len]

            if prompt_token_ids is not None:
                prompt_token_ids = prompt_token_ids[:max_log_len]

        logger.info(
            "Received request %s: prompt: %r, "
            "params: %s, prompt_token_ids: %s, "
            "lora_request: %s, prompt_adapter_request: %s.", request_id,
            prompt, params, prompt_token_ids, lora_request,
            prompt_adapter_request)
