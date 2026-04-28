from __future__ import annotations

import asyncio
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
        # Trim the message to the first newline to keep it short, but don't truncate length
        msg = msg.split("\n")[0][:180]
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


def _normalize_provider_models(provider_name: str, provider_config: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(provider_config, dict):
        raise ValueError(f"Invalid provider config for `{provider_name}`: expected mapping")

    raw_models = provider_config.get("models")
    if not isinstance(raw_models, dict) or not raw_models:
        raise ValueError(
            f"Invalid provider config for `{provider_name}`: `providers.{provider_name}.models` "
            "must be a non-empty mapping"
        )

    normalized: dict[str, dict[str, Any]] = {}
    for model_name, model_config in raw_models.items():
        clean_name = str(model_name).strip()
        if not clean_name:
            raise ValueError(f"Invalid empty model name under provider `{provider_name}`")
        if model_config is None:
            normalized[clean_name] = {}
            continue
        if not isinstance(model_config, dict):
            raise ValueError(
                f"Invalid provider model config for `{provider_name}/{clean_name}`: expected mapping or null"
            )
        normalized[clean_name] = dict(model_config)
    return normalized


def _parse_group_model_ref(ref: Any) -> tuple[str, str]:
    if not isinstance(ref, str):
        raise ValueError(f"Invalid group model reference `{ref}`: expected string `<provider>/<model_name>`")
    clean_ref = ref.strip()
    provider, sep, model_name = clean_ref.partition("/")
    if not sep or not provider.strip() or not model_name.strip():
        raise ValueError(f"Invalid group model reference `{ref}`: expected string `<provider>/<model_name>`")
    return provider.strip(), model_name.strip()


def build_litellm_model_list(config_data: dict[str, Any]) -> list[DeploymentTypedDict]:
    providers = config_data.get("providers")
    groups = config_data.get("groups")
    if not isinstance(providers, dict) or not providers:
        raise ValueError("Invalid configuration: missing top-level key `providers`")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Invalid configuration: missing top-level key `groups`")

    provider_catalog: dict[str, dict[str, Any]] = {}
    provider_shared: dict[str, dict[str, Any]] = {}
    for provider_name, provider_config in providers.items():
        if not isinstance(provider_config, dict):
            raise ValueError(f"Invalid provider config for `{provider_name}`: expected mapping")
        provider_catalog[provider_name] = _normalize_provider_models(provider_name, provider_config)
        shared = provider_config.get("shared") or {}
        if not isinstance(shared, dict):
            raise ValueError(
                f"Invalid provider config for `{provider_name}`: `providers.{provider_name}.shared` must be a mapping"
            )
        provider_shared[provider_name] = dict(shared)

    out: list[DeploymentTypedDict] = []
    for group_name, group_config in groups.items():
        if not isinstance(group_config, dict):
            raise ValueError(f"Invalid group config for `{group_name}`: expected mapping")

        group_models = group_config.get("models")
        if not isinstance(group_models, list) or not group_models:
            raise ValueError(
                f"Invalid group config for `{group_name}`: `groups.{group_name}.models` must be a non-empty list"
            )

        seen_refs: set[str] = set()
        for ref in group_models:
            provider_name, model_name = _parse_group_model_ref(ref)
            normalized_ref = f"{provider_name}/{model_name}"
            if normalized_ref in seen_refs:
                raise ValueError(f"Duplicate model `{normalized_ref}` declared in group `{group_name}`")
            seen_refs.add(normalized_ref)

            if provider_name not in provider_catalog:
                raise ValueError(
                    f"Group `{group_name}` references unknown provider `{provider_name}` in `{normalized_ref}`"
                )
            model_config = provider_catalog[provider_name].get(model_name)
            if model_config is None:
                raise ValueError(
                    f"Group `{group_name}` references undeclared model `{normalized_ref}`; "
                    f"declare `{model_name}` under `providers.{provider_name}.models` first"
                )

            merged = {
                **provider_shared[provider_name],
                **model_config,
                "model": normalized_ref,
            }
            out_row: DeploymentTypedDict = {
                "model_name": group_name,
                "litellm_params": merged,
            }
            for key in ["rpm", "tpm", "tps", "weight", "max_parallel_requests"]:
                if key in merged:
                    out_row[key] = merged[key]
            out.append(out_row)

    return out


def build_router_from_config_data(config_data: dict[str, Any]) -> Router:
    litellm_config = config_data.get("litellm")
    groups = config_data.get("groups")
    if not isinstance(litellm_config, dict):
        raise ValueError("Invalid configuration: missing top-level key `litellm`")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Invalid configuration: missing top-level key `groups`")

    model_list = build_litellm_model_list(config_data)

    if not model_list:
        raise ValueError("Invalid configuration: add at least one entry under `groups.*.models`")

    settings = dict(litellm_config.get("router") or {})

    fallbacks: list[dict[str, list[str]]] = []
    for group_name, group_config in groups.items():
        if not isinstance(group_config, dict):
            continue
        raw_fallbacks = group_config.get("fallbacks") or []
        if not raw_fallbacks:
            continue
        if not isinstance(raw_fallbacks, list):
            raise ValueError(
                f"Invalid group config for `{group_name}`: `groups.{group_name}.fallbacks` must be a list"
            )
        clean_fallbacks: list[str] = []
        for fallback_group in raw_fallbacks:
            target = str(fallback_group).strip()
            if not target:
                raise ValueError(f"Invalid empty fallback target in group `{group_name}`")
            if target not in groups:
                raise ValueError(f"Group `{group_name}` references unknown fallback group `{target}`")
            clean_fallbacks.append(target)
        if clean_fallbacks:
            fallbacks.append({group_name: clean_fallbacks})

    if fallbacks:
        settings["fallbacks"] = fallbacks

    return Router(model_list=model_list, **settings)


def init_from_config(config_path: str | None = None) -> None:
    """
    Load ``config.dev.yaml`` and build a LiteLLM ``Router``.

    Config is split into three top-level sections:
    - ``providers``: allowed provider-local model catalog plus shared provider params
    - ``groups``: logical Router groups listing concrete ``<provider>/<model_name>`` entries
    - ``litellm``: Router runtime settings such as retries, cooldowns, and default group

    Group model references are validated before Router build so unknown providers, undeclared
    provider models, and unknown fallback groups fail fast during startup instead of surfacing
    later as LiteLLM routing errors.
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

    litellm_config = data.get("litellm") or {}
    DEFAULT_MODEL_NAME = str(litellm_config.get("default_group") or "app")
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


def inline_schema_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve all JSON Schema $ref pointers and remove $defs.

    Pydantic generates $defs + $ref for nested models. Groq (and some other
    providers) reject schemas containing $defs/$ref, so we inline every
    reference before sending the schema to the provider.
    """
    defs = schema.get("$defs", {})

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                # $ref is always "#/$defs/ModelName"
                ref_name = node["$ref"].split("/")[-1]
                resolved = defs.get(ref_name)
                if resolved is None:
                    # Return the node as-is if the reference cannot be resolved to avoid infinite recursion.
                    return node
                # Recursively resolve in case the target also has $refs
                return _resolve(dict(resolved))
            return {k: _resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    return _resolve(schema)


def enforce_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Patch every object node in an inlined schema for strict-mode compatibility.

    Groq (and OpenAI strict mode) require:
      - ``additionalProperties: false`` on every object
      - Every key in ``properties`` must appear in ``required``

    Pydantic's ``model_json_schema()`` omits both, so this pass adds them.
    Fields with a ``default`` value are kept in ``required`` — strict mode
    does not allow optional fields, but the model will still emit a value.
    """
    def _patch(node: Any) -> Any:
        if isinstance(node, list):
            return [_patch(item) for item in node]
        if not isinstance(node, dict):
            return node

        # Recurse first so nested objects are patched before we inspect them
        patched: dict[str, Any] = {k: _patch(v) for k, v in node.items()}

        if patched.get("type") == "object" and "properties" in patched:
            props = patched["properties"]
            # All declared properties must be required in strict mode
            existing_required: list[str] = list(patched.get("required") or [])
            all_keys = list(props.keys())
            merged_required = existing_required + [k for k in all_keys if k not in existing_required]
            patched["required"] = merged_required
            patched["additionalProperties"] = False

        return patched

    return _patch(schema)


def build_json_schema_response_format(schema: type[BaseModel]) -> dict[str, Any]:
    """Build a LiteLLM/OpenAI-style JSON Schema response format payload.

    Calls inline_schema_refs() to resolve $defs/$ref and enforce_strict_schema()
    to add the ``additionalProperties: false`` / ``required`` constraints that
    Groq strict mode demands, before sending to the provider.
    """
    inlined = inline_schema_refs(schema.model_json_schema())
    strict_schema = enforce_strict_schema(inlined)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": strict_schema,
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
        "invalid grammar",
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

    async def batch_complete(
        self,
        messages_batch: list[list[dict]],
        **kwargs,
    ) -> list[str | Exception]:
        if self._router is None:
            raise RuntimeError("Router not initialized; call init_from_config() from main.")
        if not messages_batch:
            return []

        kw: dict[str, Any] = {
            **(self.config.litellm_params or {}),
        }
        t = kwargs.get("temperature", self.config.temperature)
        mt = kwargs.get("max_tokens", self.config.max_tokens)
        if t is not None:
            kw["temperature"] = t
        if mt is not None:
            kw["max_tokens"] = mt

        session_id = current_session_id.get()
        if session_id:
            existing_meta = kw.get("metadata") or {}
            kw["metadata"] = {"session_id": session_id, **existing_meta}

        alias = (self.config.model or DEFAULT_MODEL_NAME).strip()

        try:
            responses = await self._router.abatch_completion_one_model_multiple_requests(
                model=alias,
                messages=messages_batch,
                **kw,
            )
        except Exception as exc:
            raise LLMCallError(alias, exc) from None

        parsed: list[str | Exception] = []
        for response in responses:
            if isinstance(response, Exception):
                parsed.append(response)
                continue
            self.last_model_used = getattr(response, "model", None) or alias
            parsed.append(response.choices[0].message.content or "")
        return parsed

    def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict[str, Any]],
        **kwargs,
    ) -> dict[str, Any]:
        if self._router is None:
            raise RuntimeError("Router not initialized; call init_from_config() from main.")

        kw: dict[str, Any] = {
            "model": (self.config.model or DEFAULT_MODEL_NAME).strip(),
            "messages": messages,
            "tools": tools,
            **(self.config.litellm_params or {}),
        }
        t = kwargs.get("temperature", self.config.temperature)
        mt = kwargs.get("max_tokens", self.config.max_tokens)
        if t is not None:
            kw["temperature"] = t
        if mt is not None:
            kw["max_tokens"] = mt

        session_id = current_session_id.get()
        if session_id:
            existing_meta = kw.get("metadata") or {}
            kw["metadata"] = {"session_id": session_id, **existing_meta}

        try:
            resp = self._router.completion(**kw)
        except Exception as exc:
            raise LLMCallError(kw["model"], exc) from None

        self.last_model_used = getattr(resp, "model", None) or kw["model"]
        message = resp.choices[0].message
        content = getattr(message, "content", None) or ""
        tool_calls = getattr(message, "tool_calls", None) or []
        if not isinstance(tool_calls, list):
            tool_calls = []
        return {"content": content, "tool_calls": tool_calls}


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
