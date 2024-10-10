import warnings

try:
    import light_vllm.commit_id
    __commit__ = light_vllm.commit_id.__commit__
except Exception as e:
    warnings.warn(f"Failed to read commit hash:\n{e}",
                  RuntimeWarning,
                  stacklevel=2)
    __commit__ = "COMMIT_HASH_PLACEHOLDER"

__version__ = "0.2.2"
