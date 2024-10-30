import os
import time
from pathlib import Path
from typing import List, NamedTuple, Optional

import PIL
import torch
from PIL.Image import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

home = Path(os.path.abspath(__file__)).parent

QUESTION = "What is the content of each image?"


class ModelRequestData(NamedTuple):
    llm: LLM
    prompt: str
    stop_token_ids: Optional[List[str]]
    image_data: List[Image]
    chat_template: Optional[str]


def load_internvl(question, images) -> ModelRequestData:
    model_name = 'OpenGVLab/InternVL2-8B'

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={"image": len(image_urls)},
        mm_processor_kwargs={"max_dynamic_patch": 4},
    )

    placeholders = "\n".join(f"Image-{i}: <image>\n"
                             for i, _ in enumerate(image_urls, start=1))
    messages = [{'role': 'user', 'content': f"{placeholders}\n{question}"}]

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B#service
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    return ModelRequestData(
        llm=llm,
        prompt=prompt,
        stop_token_ids=stop_token_ids,
        image_data=images,
        chat_template=None,
    )


###########################################

question = 'Please describe the image shortly.'
image_urls = [str(home / '001.jpg')]

images = [PIL.Image.open(path) for path in image_urls]

req_data = load_internvl(question, images)

sampling_params = SamplingParams(temperature=0.0,
                                 max_tokens=128,
                                 stop_token_ids=req_data.stop_token_ids)


def single_image():
    outputs = req_data.llm.generate(
        {
            "prompt": req_data.prompt,
            "multi_modal_data": {
                "image": req_data.image_data
            },
        },
        sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text


for i in range(3):
    single_image()

torch.cuda.synchronize()

N = 10
start = time.perf_counter()

for i in range(N):
    single_image()
    torch.cuda.synchronize()

end = time.perf_counter()

elapsed_time = end - start
print("single-image single-round conversation", elapsed_time / N)

###########################################

question = 'Describe the two images in detail.'
image_urls = [str(home / '001.jpg'), str(home / '002.jpg')]

images = [PIL.Image.open(path) for path in image_urls]

req_data = load_internvl(question, images)

sampling_params = SamplingParams(temperature=0.0,
                                 max_tokens=128,
                                 stop_token_ids=req_data.stop_token_ids)


def multi_image():
    outputs = req_data.llm.generate(
        {
            "prompt": req_data.prompt,
            "multi_modal_data": {
                "image": req_data.image_data
            },
        },
        sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text


for i in range(3):
    multi_image()

torch.cuda.synchronize()

N = 10
start = time.perf_counter()

for i in range(N):
    multi_image()
    torch.cuda.synchronize()

end = time.perf_counter()

elapsed_time = end - start
print("multi-image single-round conversation", elapsed_time / N)

###########################################
