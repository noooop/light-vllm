
TASK = "retriever"
WORKFLOW = "light_vllm.wde.retriever.workflow:RetrieverWorkflow"

# Architecture -> (wde, module, class, workflow).
RETRIEVER_MODELS = {
    "XLMRobertaModel": (TASK, "bgem3", "BGEM3Model", WORKFLOW),
}


