from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, get_origin
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
import litellm
from litellm.router import Router

T = TypeVar("T", bound=BaseModel)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
load_dotenv(dotenv_path=str(_PROJECT_ROOT / ".env"))

litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]

ROUTER: Router | None = None
DEFAULT_MODEL_NAME: str = "app"


@dataclass
class LLMConfig:
    """Per-call kwargs for ``router.completion`` (``model`` is the Router alias from YAML)."""

    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    litellm_params: dict | None = field(default_factory=dict)

GLOBAL_CONFIG = LLMConfig()

def build_litellm_model_list(config: dict[str, Any], model_name: str) -> list[dict[str, Any]]:
    """Merge ``providers`` into LiteLLM ``model_list`` rows (raw dicts; LiteLLM resolves ``os.environ/``)."""
    out: list[dict[str, Any]] = []
    for _provider, prov in (config.get("providers") or {}).items():
        if not isinstance(prov, dict):
            continue
        shared = {k: v for k, v in prov.items() if k != "models"}
        for entry in prov.get("models", []):
            if isinstance(entry, str):
                row = {"model": entry}
            else:
                row = dict(entry)
            merged = {**shared, **row}
            if not merged.get("model"):
                continue
            out.append({"model_name": model_name, "litellm_params": merged})
    return out


def build_router_from_config_data(config_data: dict[str, Any]) -> Router:
    rb = config_data.get("router") or {}
    if not rb:
        raise ValueError("config.yaml: missing top-level key ``router``")

    primary = str(rb.get("default_model_name") or "app")
    fb_alias = rb.get("fallback_model_name")
    fb_alias = str(fb_alias).strip() if fb_alias else None

    model_list = build_litellm_model_list(rb, primary)
    if fb_alias and (rb.get("fallback_providers") or {}):
        model_list.extend(
            build_litellm_model_list({**rb, "providers": rb["fallback_providers"]}, fb_alias)
        )

    if not model_list:
        raise ValueError("config.yaml: add at least one entry under router.providers.*.models")

    settings = dict(rb.get("settings") or {})

    # Optional explicit cross-alias chain; omit to rely on same-alias pooling + Router defaults.
    if "fallbacks" in rb:
        settings["fallbacks"] = rb["fallbacks"]

    return Router(model_list=model_list, **settings)


def init_from_config(config_path: str | None = None) -> None:
    global ROUTER, DEFAULT_MODEL_NAME

    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    rb = data.get("router") or {}
    DEFAULT_MODEL_NAME = str(rb.get("default_model_name") or "app")
    ROUTER = build_router_from_config_data(data)


def _strip_think_block(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _strip_code_fence(text: str) -> str:
    fenced = re.sub(r"^```(?:json)?\s*\n?(.*?)\n?```$", r"\1", text, flags=re.DOTALL).strip()
    if fenced != text:
        return fenced
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
    return text


def _heal_json(raw: str, schema: type[BaseModel]) -> str:

    list_fields = [
        name
        for name, fi in schema.model_fields.items()
        if get_origin(fi.annotation) is list
    ]
    key = list_fields[0] if len(list_fields) == 1 else None
    if not key:
        return raw

    stripped = raw.strip()
    try:
        candidate = json.loads(stripped)
        if isinstance(candidate, dict) and key in candidate:
            return raw
        if isinstance(candidate, dict):
            return json.dumps({key: [candidate]})
        if isinstance(candidate, list):
            return json.dumps({key: candidate})
        return raw
    except json.JSONDecodeError:
        pass

    try:
        items = json.loads(f"[{stripped}]")
        if isinstance(items, list) and all(isinstance(i, dict) for i in items):
            return json.dumps({key: items})
    except json.JSONDecodeError:
        pass

    return raw


class LiteLLMProvider:
    def __init__(self, config: LLMConfig):
        self.config = config
        if ROUTER is None:
            init_from_config()
        self._router = ROUTER

    def complete(
        self,
        messages: list[dict],
        schema: type[T] | None = None,
        **kwargs,
    ) -> str:
        if self._router is None:
            raise RuntimeError("Router not initialized; call init_from_config() from main.")

        kw: dict[str, Any] = {
            "model": (self.config.model or DEFAULT_MODEL_NAME).strip(),
            "messages": messages,
            **(self.config.litellm_params or {}),
        }
        t = kwargs.get("temperature", self.config.temperature)
        mt = kwargs.get("max_tokens", self.config.max_tokens)
        if t is not None:
            kw["temperature"] = t
        if mt is not None:
            kw["max_tokens"] = mt
        if schema is not None:
            kw["response_format"] = {"type": "json_object"}

        resp = self._router.completion(**kw)
        return resp.choices[0].message.content or ""


def get_llm(
    config: LLMConfig | None = None,
    llm_config_override: dict | None = None,
) -> LiteLLMProvider:
    if config is None:
        config = GLOBAL_CONFIG

    if llm_config_override:
        config = LLMConfig(
            model=llm_config_override.get("model", config.model),
            temperature=llm_config_override.get("temperature", config.temperature),
            max_tokens=llm_config_override.get("max_tokens", config.max_tokens),
            litellm_params={
                **(config.litellm_params or {}),
                **(llm_config_override.get("litellm_params") or {}),
            },
        )

    return LiteLLMProvider(config)
