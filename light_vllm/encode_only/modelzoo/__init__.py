TASK = "encode_only"
WORKFLOW = "light_vllm.encode_only.workflow:EncodeOnlyWorkflow"

# Architecture -> (task, module, class, workflow).
ENCODE_ONLY_MODELS = {
    "XLMRobertaForMaskedLM":
    (TASK, "xlm_roberta", "XLMRobertaForMaskedLM", WORKFLOW),
    "BertForMaskedLM": (TASK, "bert", "BertForMaskedLM", WORKFLOW),
}
