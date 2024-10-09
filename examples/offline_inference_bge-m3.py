try:
    import light_vllm
    from light_vllm import LLM

    print("light_vllm:", light_vllm.__version__)

except Exception:
    import vllm
    from vllm import LLM
    print("vllm:", vllm.__version__)

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create an LLM.
llm = LLM(model='BAAI/bge-m3')

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.encode(prompts)
# Print the outputs.
for output in outputs:
    print(output.outputs)
