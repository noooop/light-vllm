

TASK = "chat"
WORKFLOW = "light_vllm.task.chat.workflow:ChatWorkflow"

# Architecture -> (task, module, class, workflow).
CHAT_MODELS = {
        "Qwen2ForCausalLM": (TASK, "qwen2", "Qwen2ForCausalLM", WORKFLOW),
}