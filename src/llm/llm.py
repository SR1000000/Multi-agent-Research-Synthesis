from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, get_origin
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
import litellm
from litellm.router import Router
from litellm.types.router import DeploymentTypedDict

logger = logging.getLogger(__name__)

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
    """Per-call kwargs for ``router.completion`` (``model`` is the Router group alias from YAML)."""

    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    litellm_params: dict | None = field(default_factory=dict)

GLOBAL_CONFIG = LLMConfig()

def build_litellm_model_list(
    config: dict[str, Any],
    default_alias: str,
) -> list[DeploymentTypedDict]:
    """Merge ``providers`` into LiteLLM ``model_list`` rows.

    Each model row may set ``model_name`` to register that deployment under a Router group alias
    (e.g. ``writer``). 
    If omitted, ``default_alias`` is used (``default_model_name`` or ``fallback_model_name`` for the block).
    """
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
            alias_raw = merged.pop("model_name", None)
            if isinstance(alias_raw, str) and alias_raw.strip():
                effective_alias = alias_raw.strip()
            else:
                effective_alias = default_alias
            out.append({"model_name": effective_alias, "litellm_params": merged})
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
    """
    Load ``config.yaml`` and build a LiteLLM ``Router``.

    **Deployments** — Each ``model_list`` entry is one backend (``litellm_params``) plus a Router **alias**
    (``model_name`` in LiteLLM terms). YAML rows use the block default (``default_model_name`` or
    ``fallback_model_name``) unless the row sets ``model_name: <alias>`` (e.g. ``writer``, ``fast``).

    **Same alias (intra-group)** — Rows that share one alias (e.g. two Gemini models both under ``app``)
    are one pool. The Router moves between those deployments on its own: routing strategy, retries,
    cooldowns, rate limits. No ``router.fallbacks`` entry is required for that.

    **Different aliases (cross-group)** — To move from one logical name to another (e.g. ``app`` →
    ``fallback``), LiteLLM uses ``Router(fallbacks=...)``. We pass ``router.fallbacks`` from YAML when
    present. If you omit it, the ``fallback`` group still exists in ``model_list`` but nothing auto-switches
    to it; call ``router.completion(model="…")`` with another alias (or set ``LLMConfig.model``).
    """
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
        self.last_model_used: str | None = None

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
        self.last_model_used = getattr(resp, "model", None) or kw["model"]
        return resp.choices[0].message.content or ""

    def batch_complete(
        self,
        messages_list: list[list[dict]],
        **kwargs,
    ) -> list[str | Exception] | list[list[str | Exception]]:
        """
        Send batched completion calls.

        Modes:
        - Default (backwards compatible): many prompts -> one model (Router alias)
          Returns: list[str | Exception] aligned with messages_list

        - Opt-in: one prompt -> many models (Return ALL responses), per LiteLLM docs
          Trigger by passing `models=[...]` in kwargs.
          Returns: list[list[str | Exception]] where outer index is prompt index and
          inner index is model index.

        Args:
            messages_list: List of message lists, each representing one prompt

        Returns:
            See Modes above.
        """
        if self._router is None:
            raise RuntimeError("Router not initialized; call init_from_config() from main.")

        models = kwargs.get("models")

        # Opt-in: 1 prompt -> many models -> return all responses
        if models is not None:
            if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
                raise TypeError("batch_complete(models=...) expects a list[str]")

            outputs: list[list[str | Exception]] = []
            for messages in messages_list:
                try:
                    # Per LiteLLM docs:
                    # https://docs.litellm.ai/docs/completion/batching
                    resp_list = litellm.batch_completion_models_all_responses(
                        models=models,
                        messages=messages,
                    )
                    row: list[str | Exception] = []
                    for r in resp_list:
                        try:
                            row.append(r.choices[0].message.content or "")
                        except Exception as e:
                            row.append(e)
                    outputs.append(row)
                except Exception as e:
                    outputs.append([e for _ in models])
            return outputs

        # Default: many prompts -> one model (Router alias), i.e. LiteLLM `batch_completion`
        kw: dict[str, Any] = {
            "model": (self.config.model or DEFAULT_MODEL_NAME).strip(),
            **(self.config.litellm_params or {}),
        }
        t = kwargs.get("temperature", self.config.temperature)
        mt = kwargs.get("max_tokens", self.config.max_tokens)
        if t is not None:
            kw["temperature"] = t
        if mt is not None:
            kw["max_tokens"] = mt

        # Prevent duplicate kw collisions (kw already contains model)
        call_kw = dict(kw)
        model_alias = call_kw.pop("model")

        try:
            # Router-side batch completion: one model group alias, many prompts.
            results = self._router.batch_completion(model=model_alias, messages=messages_list, **call_kw)
        except AttributeError:
            logger.warning("Router.batch_completion not available; falling back to individual calls")
            results = []
            for messages in messages_list:
                try:
                    resp = self._router.completion(
                        model=model_alias,
                        messages=messages,
                        **call_kw,
                    )
                    results.append(resp)
                except Exception as e:
                    results.append(e)

        # Extract content from each response
        outputs = []
        for result in results:
            if isinstance(result, Exception):
                outputs.append(result)
            else:
                outputs.append(result.choices[0].message.content or "")

        return outputs


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
