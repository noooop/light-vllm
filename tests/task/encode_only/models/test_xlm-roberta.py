
import pytest

from typing import (Any, Callable, Dict, List, Optional, Tuple, TypedDict,
                    TypeVar, Union)

import gc
from light_vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, is_cpu
from transformers import (AutoModelForCausalLM, AutoTokenizer, BatchEncoding,
                          BatchFeature)
import torch
import torch.nn as nn
import torch.nn.functional as F
from light_vllm import LLM

_T = TypeVar("_T", nn.Module, torch.Tensor, BatchEncoding, BatchFeature)


def cleanup():
    gc.collect()
    if not is_cpu():
        torch.cuda.empty_cache()




class HfRunner:
    def wrap_device(self, input: _T) -> _T:
        if not is_cpu():
            # Check if the input is already on the GPU
            if hasattr(input, 'device') and input.device.type == "cuda":
                return input  # Already on GPU, no need to move
            return input.to("cuda")
        else:
            # Check if the input is already on the CPU
            if hasattr(input, 'device') and input.device.type == "cpu":
                return input  # Already on CPU, no need to move
            return input.to("cpu")

    def __init__(
        self,
        model_name: str,
        dtype: str = "half",
        *,
        model_kwargs: Optional[Dict[str, Any]] = None,
        auto_cls=AutoModelForCausalLM
    ) -> None:
        torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[dtype]

        self.model_name = model_name

        model_kwargs = model_kwargs if model_kwargs is not None else {}

        self.model = self.wrap_device(
            auto_cls.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                **model_kwargs,
            ))

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

    @torch.inference_mode
    def encode(self, prompts: List[str]) -> List[List[torch.Tensor]]:
        encoded_input = self.tokenizer(prompts,
                                       padding=True,
                                       truncation=True,
                                       return_tensors='pt').to("cuda")
        return self.model(**encoded_input).logits

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        cleanup()



class VllmRunner:
    def __init__(
        self,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        dtype: str = "half",
    ) -> None:
        self.model = LLM(
            model=model_name,
            tokenizer=tokenizer_name,
            trust_remote_code=True,
            dtype=dtype)

    def encode(self, prompts: List[str]) -> List[List[float]]:
        req_outputs = self.model.encode(prompts)
        outputs = []
        for req_output in req_outputs:
            embedding = req_output.outputs
            outputs.append(embedding)
        return torch.stack(outputs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        cleanup()


@pytest.fixture(scope="session")
def hf_runner():
    return HfRunner

@pytest.fixture(scope="session")
def vllm_runner():
    return VllmRunner


@pytest.fixture(scope="session")
def example_prompts():
    return [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]


MODELS = [
    'FacebookAI/xlm-roberta-base'
]

def compare_embeddings(embeddings1, embeddings2):
    similarities = [
        F.cosine_similarity(torch.tensor(e1), torch.tensor(e2), dim=0)
        for e1, e2 in zip(embeddings1, embeddings2)
    ]
    return similarities


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@torch.inference_mode
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.encode(example_prompts)

    with vllm_runner(model, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.encode(example_prompts)

    similarities = compare_embeddings(hf_outputs, vllm_outputs)
    all_similarities = torch.stack(similarities)
    tolerance = 1e-2
    assert torch.all((all_similarities <= 1.0 + tolerance)
                     & (all_similarities >= 1.0 - tolerance)
                     ), f"Not all values are within {tolerance} of 1.0"
