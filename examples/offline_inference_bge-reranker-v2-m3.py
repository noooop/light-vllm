import light_vllm
from light_vllm import LLM

print("light_vllm:", light_vllm.__version__)

pairs = [
    ['query', 'passage'], ['what is panda?', 'hi'],
    [
        'what is panda?',
        'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.'
    ]
]

llm = LLM(model="BAAI/bge-reranker-v2-m3")

outputs = llm.reranker(pairs)
for output in outputs:
    print(output.score)
