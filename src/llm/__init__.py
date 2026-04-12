from . import llm as _llm_impl
from .llm import (
    GLOBAL_CONFIG,
    LLMConfig,
    LiteLLMProvider,
    build_litellm_model_list,
    build_router_from_config_data,
    get_llm,
    init_from_config,
    _heal_json,
    _strip_code_fence,
    _strip_think_block,
)


def __getattr__(name: str):
    if name == "ROUTER":
        return _llm_impl.ROUTER
    if name == "DEFAULT_MODEL_NAME":
        return _llm_impl.DEFAULT_MODEL_NAME
    if name in ("DEFAULT_LOGICAL_MODEL", "DEFAULT_LITELLM_MODEL"):
        return _llm_impl.DEFAULT_MODEL_NAME
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GLOBAL_CONFIG",
    "LLMConfig",
    "ROUTER",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_LOGICAL_MODEL",
    "DEFAULT_LITELLM_MODEL",
    "LiteLLMProvider",
    "build_litellm_model_list",
    "build_router_from_config_data",
    "get_llm",
    "init_from_config",
    "_heal_json",
    "_strip_code_fence",
    "_strip_think_block",
]
