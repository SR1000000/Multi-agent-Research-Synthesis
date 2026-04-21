from __future__ import annotations

import contextvars
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, get_origin
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
import litellm
import logging as _logging
_logging.getLogger("LiteLLM").setLevel(_logging.ERROR)
_logging.getLogger("LiteLLM Router").setLevel(_logging.ERROR)
from litellm.router import Router
from litellm.types.router import DeploymentTypedDict
from litellm.integrations.custom_logger import CustomLogger
from src.logging.logger import AgentLogger

T = TypeVar("T", bound=BaseModel)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.dev.yaml"
_SAMPLE_CONFIG_PATH = Path(__file__).resolve().parent / "config.sample.yaml"
load_dotenv(dotenv_path=str(_PROJECT_ROOT / ".env"))

# Langfuse Python SDK v2 reads LANGFUSE_HOST for the API URL, not LANGFUSE_BASE_URL.
# Mirror so a .env that only sets LANGFUSE_BASE_URL (e.g. regional cloud URL) still works.
_langfuse_base_url = os.environ.get("LANGFUSE_BASE_URL")
if _langfuse_base_url and "LANGFUSE_HOST" not in os.environ:
    os.environ["LANGFUSE_HOST"] = _langfuse_base_url.strip()


_agent_logger = AgentLogger()
current_agent_label: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_agent_label", default="LLM",
)
current_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_session_id", default=None,
)


class _FailureLogger(CustomLogger):
    """Logs every individual LiteLLM deployment failure as a clean one-liner."""

    def _format(self, kwargs: dict) -> str:
        agent = current_agent_label.get()
        alias = kwargs.get("model") or "unknown"
        actual = (kwargs.get("litellm_params") or {}).get("model")
        model = f"{alias} ({actual})" if actual and actual != alias else alias
        exc = kwargs.get("exception")
        exc_type = type(exc).__name__ if exc else "Error"
        status = getattr(exc, "status_code", None)
        status_part = f" [{status}]" if status else ""
        msg = str(exc) if exc else ""
        # Trim the message to the first sentence / newline to keep it short
        msg = msg.split("\n")[0][:120]
        return f"[{agent}] Deployment failed: {model} — {exc_type}{status_part}: {msg}"

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):  # noqa: ANN001
        _agent_logger.log(self._format(kwargs), level="warning")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):  # noqa: ANN001
        _agent_logger.log(self._format(kwargs), level="warning")


_failure_logger = _FailureLogger()
litellm.callbacks = [_failure_logger, "langfuse"]

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


@dataclass
class StructuredOutputMetadata:
    requested_mode: str | None = None
    mode_used: str | None = None
    used_native_schema: bool = False
    fallback_reason: str | None = None


class LLMCallError(RuntimeError):
    """Raised when the LiteLLM router exhausts all retries/fallbacks."""
    def __init__(self, model: str, cause: Exception) -> None:
        self.model = model                                               # router alias, e.g. "slides"
        self.actual_model: str | None = getattr(cause, "model", None)  # e.g. "gemini/gemini-3.1-flash-lite-preview"
        self.status_code: int | None = getattr(cause, "status_code", None)
        exc_type = type(cause).__name__
        status_part = f" [{self.status_code}]" if self.status_code else ""
        super().__init__(f"{exc_type}{status_part}")


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
            
            out_row = {"model_name": effective_alias, "litellm_params": merged}
            for key in ["rpm", "tpm", "tps", "weight", "order"]:
                if key in merged:
                    out_row[key] = merged.pop(key)
            out.append(out_row)
    return out


def build_router_from_config_data(config_data: dict[str, Any]) -> Router:
    rb = config_data.get("router") or {}
    if not rb:
        raise ValueError("Invalid configuration: missing top-level key ``router``")

    primary = str(rb.get("default_model_name") or "app")
    fb_alias = rb.get("fallback_model_name")
    fb_alias = str(fb_alias).strip() if fb_alias else None

    model_list = build_litellm_model_list(rb, primary)
    if fb_alias and (rb.get("fallback_providers") or {}):
        model_list.extend(
            build_litellm_model_list({**rb, "providers": rb["fallback_providers"]}, fb_alias)
        )

    if not model_list:
        raise ValueError("Invalid configuration: add at least one entry under router.providers.*.models")

    settings = dict(rb.get("settings") or {})

    # Optional explicit cross-alias chain; omit to rely on same-alias pooling + Router defaults.
    if "fallbacks" in rb:
        settings["fallbacks"] = rb["fallbacks"]

    return Router(model_list=model_list, **settings)


def init_from_config(config_path: str | None = None) -> None:
    """
    Load ``config.dev.yaml`` and build a LiteLLM ``Router``.

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
    if path == _DEFAULT_CONFIG_PATH and not path.exists():
        if not _SAMPLE_CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"Missing LLM config at {path} and sample config at {_SAMPLE_CONFIG_PATH}"
            )
        shutil.copyfile(_SAMPLE_CONFIG_PATH, path)
        _agent_logger.log(
            f"[LLM] Seeded missing config from sample: {path.name}",
            level="info",
        )

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


# Matches backslashes NOT followed by a valid JSON escape character.
# Valid: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
# Everything else (e.g. \l, \e, \c, \_) is illegal in JSON and commonly
# produced by LLMs writing raw LaTeX inside JSON string values.
_INVALID_JSON_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')


def _fix_latex_escapes(text: str) -> str:
    """Double-escape backslashes that form invalid JSON escape sequences.

    LLMs writing LaTeX math (e.g. \\epsilon, \\log, \\cdot) inside JSON
    string values often emit a single backslash, which is illegal JSON.
    This pass converts every such bare backslash to \\\\ so the JSON is
    parseable before it reaches Pydantic validation.
    """
    return _INVALID_JSON_ESCAPE.sub(r'\\\\', text)


def _parse_first_json_value(text: str) -> Any | None:
    """Parse the first JSON value and ignore any trailing junk."""
    try:
        value, _end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        return None
    return value


def _unwrap_schema_wrapper(candidate: Any, schema: type[BaseModel]) -> Any:
    """Unwrap a lone outer key when the inner object clearly matches the schema better."""
    if not isinstance(candidate, dict) or len(candidate) != 1:
        return candidate

    inner = next(iter(candidate.values()))
    if not isinstance(inner, dict):
        return candidate

    schema_keys = set(schema.model_fields)
    outer_overlap = len(set(candidate) & schema_keys)
    inner_overlap = len(set(inner) & schema_keys)
    return inner if inner_overlap > outer_overlap else candidate


def _heal_json(raw: str, schema: type[BaseModel]) -> str:

    list_fields = [
        name
        for name, fi in schema.model_fields.items()
        if get_origin(fi.annotation) is list
    ]
    key = list_fields[0] if len(list_fields) == 1 else None
    if not key:
        key = None

    stripped = _fix_latex_escapes(raw.strip())
    candidate = _parse_first_json_value(stripped)
    if candidate is not None:
        candidate = _unwrap_schema_wrapper(candidate, schema)
        if isinstance(candidate, dict):
            if key is None or key in candidate:
                return json.dumps(candidate)
            return json.dumps({key: [candidate]})
        if isinstance(candidate, list) and key is not None:
            return json.dumps({key: candidate})
        return raw

    if key is not None:
        try:
            items = json.loads(f"[{stripped}]")
            if isinstance(items, list) and all(isinstance(i, dict) for i in items):
                return json.dumps({key: items})
        except json.JSONDecodeError:
            pass

    return raw


def build_json_schema_response_format(schema: type[BaseModel]) -> dict[str, Any]:
    """Build a LiteLLM/OpenAI-style JSON Schema response format payload."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": schema.model_json_schema(),
            "strict": True,
        },
    }


def should_fallback_to_json_object(exc: Exception) -> bool:
    """Return True when a provider likely rejected native JSON Schema mode."""
    message = str(exc).lower()
    fallback_markers = (
        "json_schema",
        "response schema",
        "response_schema",
        "response format",
        "response_format",
        "structured output",
        "structured-output",
        "unsupported",
        "not supported",
        "unknown parameter",
        "invalid parameter",
        "invalid value",
        "extra_forbidden",
        "extra inputs are not permitted",
        "extra inputs not permitted",
    )
    return any(marker in message for marker in fallback_markers)


def _litellm_model_from_deployment(deployment: Any) -> str | None:
    """Extract ``litellm_params.model`` from a Router deployment object or dict."""
    if deployment is None:
        return None
    litellm_params = (
        deployment.get("litellm_params")
        if isinstance(deployment, dict)
        else getattr(deployment, "litellm_params", None)
    )
    if litellm_params is None:
        return None
    model_id = (
        litellm_params.get("model")
        if isinstance(litellm_params, dict)
        else getattr(litellm_params, "model", None)
    )
    if not model_id:
        return None
    return str(model_id).strip()


class LiteLLMProvider:
    def __init__(self, config: LLMConfig):
        self.config = config
        if ROUTER is None:
            init_from_config()
        self._router = ROUTER
        self.last_model_used: str | None = None
        self.last_structured_output_metadata: StructuredOutputMetadata | None = None

    def peek_router_litellm_model(self, messages: list[dict]) -> str | None:
        """Best-effort ``litellm_params['model']`` for the deployment the router would pick.

        Mirrors the same ``get_available_deployment`` path used for ``completion`` (routing,
        health, cooldown). Returns ``None`` if the router is unavailable or resolution fails.
        """
        if self._router is None:
            return None
        alias = (self.config.model or DEFAULT_MODEL_NAME).strip()
        try:
            deployment = self._router.get_available_deployment(alias, messages=messages)
        except Exception:
            return None
        return _litellm_model_from_deployment(deployment)

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
        self.last_structured_output_metadata = None

        # Inject session_id into LiteLLM metadata so the built-in Langfuse
        # callback tags every litellm-completion trace with the current session.
        session_id = current_session_id.get()
        if session_id:
            existing_meta = kw.get("metadata") or {}
            kw["metadata"] = {"session_id": session_id, **existing_meta}

        try:
            if schema is None:
                resp = self._router.completion(**kw)
                self.last_structured_output_metadata = StructuredOutputMetadata()
            else:
                native_kw = dict(kw)
                native_kw["response_format"] = build_json_schema_response_format(schema)
                try:
                    resp = self._router.completion(**native_kw)
                    self.last_structured_output_metadata = StructuredOutputMetadata(
                        requested_mode="native_schema",
                        mode_used="native_schema",
                        used_native_schema=True,
                    )
                except Exception as native_exc:
                    if not should_fallback_to_json_object(native_exc):
                        raise native_exc

                    json_object_kw = dict(kw)
                    json_object_kw["response_format"] = {"type": "json_object"}
                    resp = self._router.completion(**json_object_kw)
                    self.last_structured_output_metadata = StructuredOutputMetadata(
                        requested_mode="native_schema",
                        mode_used="json_object",
                        used_native_schema=False,
                        fallback_reason=str(native_exc).split("\n")[0][:200],
                    )
        except Exception as exc:
            raise LLMCallError(kw["model"], exc) from None
        self.last_model_used = getattr(resp, "model", None) or kw["model"]
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
