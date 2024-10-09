TASK = "decode_only"
WORKFLOW = "light_vllm.wde.decode_only.workflow:DecodeOnlyWorkflow"

# Architecture -> (task, module, class, workflow).
DECODE_ONLY_MODELS = {
    "Qwen2ForCausalLM":
    (TASK, "qwen2", "Qwen2ForCausalLM",
     "light_vllm.wde.retriever.modelzoo.gte_qwen.workflow:Qwen2Workflow"),
}
