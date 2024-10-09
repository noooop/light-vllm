import light_vllm
from light_vllm import LLM

print("light_vllm:", light_vllm.__version__)

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

llm = LLM(model="google-bert/bert-base-uncased")

outputs = llm.encode(prompts)
for output in outputs:
    print(output.outputs.shape)
