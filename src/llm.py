from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar, Any

from dotenv import load_dotenv
from pydantic import BaseModel
import litellm
from litellm import token_counter, supports_vision, supports_reasoning
from litellm.router import Router

T = TypeVar("T", bound=BaseModel)

ROUTER_PRIMARY_ALIAS = "primary"

DEFAULT_LITELLM_MODEL = "gemini/gemini-2.0-flash-001"

DEFAULT_CROSS_PROVIDER_FALLBACKS: tuple[str, ...] = (
    "openrouter/google/gemini-2.0-flash-001",
    "ollama/qwen3:8b",
)

# Same ``model_name`` group → multiple deployments (load-balance / retry within group).
# Keys are exact LiteLLM ids for a chain slot; values are additional models tried in that
# group before Router moves to the next fallback-* group.
_DEFAULT_PROVIDER_FALLBACKS: dict[str, list[str]] = {
    "gemini/gemini-2.0-flash-001": [
        "gemini/gemini-2.0-flash",
        "gemini/gemini-2.0-flash-lite",
    ],
    "gemini/gemini-3.1-flash-lite-preview": [
        "gemini/gemini-2.0-flash-001",
        "gemini/gemini-2.0-flash",
        "gemini/gemini-2.0-flash-lite",
    ],
    "gemini/gemini-2.5-flash": [
        "gemini/gemini-2.5-flash-lite",
        "gemini/gemini-2.5-pro",
        "gemini/gemini-2.5-pro-preview-tts",
        "gemini/gemini-2.0-flash",
        "gemini/gemini-2.0-flash-lite",
    ],
    "ollama/qwen.qwen3-32b-v1:0": [
        "ollama/qwen.qwen3-next-80b-a3b",
        "ollama/qwen.qwen3-vl-235b-a22b",
    ],
    "openai/gpt-4": [
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "openai/gpt-4-turbo",
        "openai/gpt-4.1",
        "openai/gpt-4.1-mini",
    ],
}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=str(_PROJECT_ROOT / ".env"))

litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]

ROUTER: Router | None = None


@dataclass
class LLMConfig:
    """Settings merged into ``router.completion``; chain is built separately (CLI → ``ROUTER``)."""

    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    think: bool | None = True
    base_url: str | None = None
    api_key: str | None = None
    litellm_params: dict | None = field(default_factory=dict)
    cooldown_time: int | float = 60


def _deployment_litellm_params(
    deployment_model: str,
    alias: str,
    config: LLMConfig,
    cli_primary_model: str,
) -> dict[str, Any]:
    """Explicit api_key / api_base only on the deployment that matches the CLI primary id."""
    out: dict[str, Any] = {}
    if (
        alias == ROUTER_PRIMARY_ALIAS
        and deployment_model.strip() == cli_primary_model.strip()
    ):
        if config.api_key:
            out["api_key"] = config.api_key
        if config.base_url:
            out["api_base"] = config.base_url
    return out


def _group_models_for_slot(slot_model: str) -> list[str]:
    """Ordered deployments for one Router group (slot + provider-default alternates)."""
    mid = slot_model.strip()
    out: list[str] = []
    seen: set[str] = set()

    def add(m: str) -> None:
        m = m.strip()
        if m and m not in seen:
            seen.add(m)
            out.append(m)

    add(mid)
    for alt in _DEFAULT_PROVIDER_FALLBACKS.get(mid, []):
        add(alt)
    return out


def _think_litellm_params(litellm_model: str, think: bool | None) -> dict[str, Any]:
    """Provider-specific think / reasoning on each deployment (not on router.completion())."""
    if think is None or not think:
        return {}
    ml = litellm_model.lower()
    if "gemini" in ml:
        return {"extra_body": {"thinking_config": {"include_thoughts": True}}}
    if "ollama" in ml:
        return {"extra_body": {"think": True}}
    try:
        if supports_reasoning(litellm_model):
            return {"reasoning_effort": "low"}
    except Exception:
        pass
    return {}


def build_router(chain: list[str], config: LLMConfig) -> Router:
    """
    ``chain`` = CLI-ordered LiteLLM ids. Each position becomes a **group** (same
    ``model_name``) with one or more deployments (slot model + ``_DEFAULT_PROVIDER_FALLBACKS``).

    Router ``fallbacks`` wires groups: ``primary`` → ``fallback-1`` → …

    LiteLLM reads API keys from the environment per deployment unless ``config``
    pins ``api_key`` / ``base_url`` on the deployment whose model id matches the
    first chain entry (CLI primary).
    """
    if not chain:
        chain = [DEFAULT_LITELLM_MODEL]

    cli_primary = chain[0].strip()
    aliases = [ROUTER_PRIMARY_ALIAS] + [f"fallback-{i}" for i in range(1, len(chain))]
    model_list: list[dict[str, Any]] = []
    for alias, slot_model in zip(aliases, chain):
        for mid in _group_models_for_slot(slot_model):
            litellm_params: dict[str, Any] = {"model": mid}
            litellm_params.update(
                _deployment_litellm_params(mid, alias, config, cli_primary)
            )
            litellm_params.update(_think_litellm_params(mid, config.think))
            model_list.append({"model_name": alias, "litellm_params": litellm_params})

    fallback_aliases = aliases[1:]
    return Router(
        model_list=model_list,
        fallbacks=[{ROUTER_PRIMARY_ALIAS: fallback_aliases}] if fallback_aliases else [],
        num_retries=2,
        retry_after=3,
        cooldown_time=config.cooldown_time,
        timeout=120,
    )


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
    from typing import get_origin

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
    """
    Uses the module ``ROUTER`` when ``config.model`` matches ``GLOBAL_CONFIG.model``;
    otherwise builds a router for ``[model] + DEFAULT_CROSS_PROVIDER_FALLBACKS``
    (same intra-group expansion inside ``build_router`` as for the global router).
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        mid = (config.model or DEFAULT_LITELLM_MODEL).strip()
        gmid = (GLOBAL_CONFIG.model or DEFAULT_LITELLM_MODEL).strip()
        if ROUTER is not None and mid == gmid:
            self._router = ROUTER
        else:
            # One slot for ``mid`` (intra-group via _DEFAULT_PROVIDER_FALLBACKS) + same
            # cross-provider tail as CLI when model override differs from GLOBAL_CONFIG.
            chain = [mid] + list(DEFAULT_CROSS_PROVIDER_FALLBACKS)
            self._router = build_router(chain, config)

    def complete(
        self,
        messages: list[dict],
        schema: type[T] | None = None,
        **kwargs,
    ) -> str:
        primary_model = (self.config.model or DEFAULT_LITELLM_MODEL).strip()

        req_params: dict[str, Any] = {
            "model": ROUTER_PRIMARY_ALIAS,
            "messages": messages,
            **(self.config.litellm_params or {}),
        }

        temp = kwargs.get("temperature", self.config.temperature)
        if temp is not None:
            req_params["temperature"] = temp

        mt = kwargs.get("max_tokens", self.config.max_tokens)
        if mt is not None:
            req_params["max_tokens"] = mt

        try:
            count = token_counter(model=primary_model, messages=messages)
            print(f"[llm] {primary_model} — input tokens: {count}")
        except Exception:
            pass

        if schema is not None:
            req_params["response_format"] = {"type": "json_object"}

        has_images = any(
            isinstance(m.get("content"), list)
            and any(
                isinstance(p, dict) and p.get("type") == "image_url"
                for p in m["content"]
            )
            for m in messages
        )
        if has_images:
            try:
                if not supports_vision(primary_model):
                    print(f"[llm] WARNING: {primary_model} does not support vision — call may fail.")
            except Exception:
                pass

        resp = self._router.completion(**req_params)
        return resp.choices[0].message.content or ""


GLOBAL_CONFIG = LLMConfig()


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
            think=llm_config_override.get("think", config.think),
            base_url=llm_config_override.get("base_url", config.base_url),
            api_key=llm_config_override.get("api_key", config.api_key),
            litellm_params={
                **(config.litellm_params or {}),
                **(llm_config_override.get("litellm_params") or {}),
            },
            cooldown_time=llm_config_override.get("cooldown_time", config.cooldown_time),
        )

    return LiteLLMProvider(config)
