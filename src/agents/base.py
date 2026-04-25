import json
import time
from contextvars import ContextVar
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, TypeVar
from pydantic import BaseModel, ValidationError
from src.llm.llm import (
    get_llm,
    _strip_think_block,
    _strip_code_fence,
    _heal_json,
    DEFAULT_MODEL_NAME,
    LLMCallError,
    StructuredOutputMetadata,
    current_agent_label,
    current_session_id,
)
from src.logging.logger import AgentLogger
from src.agents.prompts.common import build_structured_retry_turns
from src.tools.registry import execute_tool_call, get_tool_prompt_snippets, get_tool_schemas
from src.tools.rag import format_tool_result_for_llm

# Generic placeholder for "the Pydantic model this call expects back." Binding it
# to BaseModel lets typed callers get their concrete schema type returned instead
# of a plain BaseModel.
T = TypeVar("T", bound=BaseModel)

# Propagated from slide writer / critic dispatch state so RAG can tag retrieval rows by plan.
current_plan_generation: ContextVar[int] = ContextVar("current_plan_generation", default=0)


@dataclass
class StructuredOutputResult:
    """Return envelope for structured LLM calls.

    Keeping the parsed model with provider metadata and retry count gives callers
    a single object for both the useful result and debugging/observability data.
    """
    parsed: BaseModel
    metadata: StructuredOutputMetadata
    attempts_used: int


class BaseLLMAgent:
    def __init__(
        self,
        role: str,
        *,
        system_prompt: str,
        log_display: str | None = None,
        tools_for_agent: dict[str, dict[str, Any]] | None = None,
    ):
        self.role = role
        self.system_prompt = system_prompt
        self._log_display = log_display if log_display is not None else role
        self._logger = AgentLogger()
        self._last_model_used: str | None = None  # Used for logging validation errors
        self._tools_for_agent = tools_for_agent or {}
        self._last_structured_output_metadata = StructuredOutputMetadata()

    def _set_session_id(self, state: dict) -> None:
        """Propagate session_id from the node's state into the module-level ContextVar.
        Called at the start of every agent run() so that the LiteLLM Langfuse callback
        always tags traces with the correct session, regardless of whether this node
        runs in the main thread or a parallel worker thread.
        """
        sid = state.get("session_id") if isinstance(state, dict) else None
        if sid:
            current_session_id.set(sid)

    def _set_plan_generation(self, state: dict) -> None:
        """Tag tool execution with the active presentation plan generation (replan epoch)."""
        if not isinstance(state, dict):
            return
        g = state.get("plan_generation")
        if g is not None:
            current_plan_generation.set(int(g))

    def _build_messages(
        self,
        turns: list[dict],
        *,
        system_prompt_override: str | None = None,
    ) -> list[dict]:
        """Build the full message list by prepending the system prompt.

        LiteLLM handles system message translation for all providers including
        Gemini (which doesn't accept role=system natively) — no manual separation
        needed here.

        For prompt caching (Anthropic/Gemini), wrap the system content as a list:
            {'role': 'system', 'content': [
                {'type': 'text', 'text': '...', 'cache_control': {'type': 'ephemeral'}}
            ]}
        """
        system_prompt = system_prompt_override or self.system_prompt
        snippets = get_tool_prompt_snippets(self._tools_for_agent)
        if snippets:
            system_prompt = f"{system_prompt}\n\n" + "\n\n".join(snippets)
        return [{'role': 'system', 'content': system_prompt}, *turns]

    def _call_raw(
        self,
        turns: list[dict],
        schema: type[T] | None = None,
        model: str | None = None,
        llm_config_override: dict | None = None,
        system_prompt_override: str | None = None,
    ) -> str:
        """
        Single LLM completion call. Transport reliability (retries, fallbacks,
        cooldowns, timeouts — per LiteLLM Router) is handled inside
        ``LiteLLMProvider.complete()``; this method does not add a second retry layer.

        Schema validation retries (with correction prompts) live in ``_call_structured``.

        Args:
            turns:               User/assistant conversation turns.
            schema:              Pydantic schema — activates JSON mode.
                                 Parsing and validation happen in ``_call_structured``, not here.
            model:               Router group alias: ``router.default_model_name`` when omitted, or any alias
                                 defined in YAML (including per-row ``model_name`` on a provider model).
            llm_config_override: Dict of LLMConfig field overrides for this call.

        Returns:
            Raw string response from the model (may contain think blocks or
            code fences — callers strip those themselves).
        """
        messages = self._build_messages(turns, system_prompt_override=system_prompt_override)
        override = dict(llm_config_override) if llm_config_override else {}
        if model is not None:
            override["model"] = model
        llm = get_llm(llm_config_override=override if override else None)
        alias = (llm.config.model or DEFAULT_MODEL_NAME).strip()
        intended_model_name = llm.peek_router_litellm_model(messages) or alias
        intended_structured = "text" if schema is None else "native_schema"
        self._logger.log(
            f"[{self._log_display}] LLM request started "
            f"(model_name={intended_model_name}, structured_output={intended_structured})"
        )
        token = current_agent_label.set(self._log_display)
        try:
            t0 = time.perf_counter()
            try:
                content = llm.complete(messages, schema=schema)
            except LLMCallError as exc:
                elapsed_s = time.perf_counter() - t0
                model_label = (
                    f"{exc.model} ({exc.actual_model})" if exc.actual_model else exc.model
                )
                self._logger.log(
                    f"[{self._log_display}] LLM call failed after {elapsed_s:.2f}s "
                    f"(model={model_label}): {exc}",
                    level="error",
                )
                raise
            elapsed_s = time.perf_counter() - t0
            actual_model = llm.last_model_used or (model or DEFAULT_MODEL_NAME)
            self._last_model_used = actual_model
            self._last_structured_output_metadata = (
                llm.last_structured_output_metadata or StructuredOutputMetadata()
            )
            label = f"default ({actual_model})" if model is None else f"{model} ({actual_model})"
            metadata = self._last_structured_output_metadata
            if schema is None:
                used_structured = "text"
            else:
                used_structured = metadata.mode_used or "unknown"
            fallback_note = (
                f", fallback={metadata.fallback_reason}"
                if schema is not None and metadata.fallback_reason
                else ""
            )
            self._logger.log(
                f"[{self._log_display}] LLM request completed in {elapsed_s:.2f}s "
                f"(model={label}, structured_output={used_structured}{fallback_note})"
            )
            return content
        finally:
            current_agent_label.reset(token)

    def _call_structured(
        self,
        turns: list[dict],
        schema: type[T],
        *,
        max_retries: int = 2,
        model: str | None = None,
        llm_config_override: dict | None = None,
        runtime_validator: Callable[[T], list[str]] | None = None,
        system_prompt_override: str | None = None,
    ) -> StructuredOutputResult:
        """Call the LLM and validate the response against a Pydantic schema.

        This wraps ``_call_raw`` with the structured-output responsibilities:
        strip provider artifacts, heal small JSON issues, validate the schema,
        optionally run semantic/runtime checks, and retry with a correction prompt
        when validation fails.
        """
        current_turns = list(turns)
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            raw = self._call_raw(
                current_turns,
                schema=schema,
                model=model,
                llm_config_override=llm_config_override,
                system_prompt_override=system_prompt_override,
            )
            clean = _strip_code_fence(_strip_think_block(raw))
            healed = _heal_json(clean, schema)
            healed_changed = healed != clean
            metadata = self._last_structured_output_metadata

            try:
                parsed = schema.model_validate_json(healed)
            except ValidationError as exc:
                last_error = exc
                retry_note = (
                    " Retrying with correction prompt."
                    if attempt < max_retries
                    else " No retries left; propagating error."
                )
                dump_path = self._logger.dump_validation_error(
                    self._log_display,
                    attempt,
                    max_retries + 1,
                    exc,
                    clean,
                    model=self._last_model_used,
                    stage="schema_validation",
                    structured_output=asdict(metadata),
                    healed_json_changed=healed_changed,
                )
                location = f" See: {dump_path}" if dump_path else ""
                self._logger.log(
                    f"[{self._log_display}] Schema validation error "
                    f"(attempt {attempt + 1}/{max_retries + 1}).{retry_note}{location}"
                )
                if attempt == max_retries:
                    break
                current_turns = build_structured_retry_turns(
                    current_turns,
                    clean,
                    str(exc),
                    schema,
                )
                continue

            if runtime_validator is not None:
                failures = runtime_validator(parsed)
                if failures:
                    failure_summary = "\n".join(f"  - {failure}" for failure in failures)
                    last_error = ValueError(failure_summary)
                    retry_note = (
                        " Retrying with correction prompt."
                        if attempt < max_retries
                        else " No retries left; propagating error."
                    )
                    dump_path = self._logger.dump_validation_error(
                        self._log_display,
                        attempt,
                        max_retries + 1,
                        ValueError(failure_summary),
                        clean,
                        model=self._last_model_used,
                        stage="runtime_validation",
                        structured_output=asdict(metadata),
                        healed_json_changed=healed_changed,
                    )
                    location = f" See: {dump_path}" if dump_path else ""
                    self._logger.log(
                        f"[{self._log_display}] Runtime validation error "
                        f"(attempt {attempt + 1}/{max_retries + 1}).{retry_note}{location}"
                    )
                    if attempt == max_retries:
                        break
                    current_turns = build_structured_retry_turns(
                        current_turns,
                        parsed.model_dump_json(),
                        failure_summary,
                        schema,
                    )
                    continue

            return StructuredOutputResult(
                parsed=parsed,
                metadata=metadata,
                attempts_used=attempt + 1,
            )

        raise last_error

    def _call(
        self,
        turns: list[dict],
        *,
        schema: type[T] | None = None,
        max_retries: int = 2,
        model: str | None = None,
        llm_config_override: dict | None = None,
        system_prompt_override: str | None = None,
        use_tools: bool = False,
        session_id: str | None = None,
        max_tool_calls: int = 4,
    ) -> str | T | dict[str, Any]:
        """
        Unified LLM entrypoint.

        Modes:
          - text: default
          - structured: set ``schema=``
          - tool loop: set ``use_tools=True``

        Tool mode currently returns the legacy dict payload used by writer flows.
        Structured output and tool mode are intentionally exclusive for now.
        """
        if use_tools and schema is not None:
            raise ValueError("Structured output with tool calls is not supported by BaseLLMAgent._call")

        if use_tools:
            return self._call_tools(
                turns,
                session_id=session_id,
                max_tool_calls=max_tool_calls,
                model=model,
                llm_config_override=llm_config_override,
                system_prompt_override=system_prompt_override,
            )

        if schema is None:
            raw = self._call_raw(
                turns,
                schema=None,
                model=model,
                llm_config_override=llm_config_override,
                system_prompt_override=system_prompt_override,
            )
            return _strip_think_block(raw)
        return self._call_structured(
            turns,
            schema,
            max_retries=max_retries,
            model=model,
            llm_config_override=llm_config_override,
            system_prompt_override=system_prompt_override,
        ).parsed

    def _call_tools(
        self,
        turns: list[dict],
        *,
        session_id: str | None = None,
        max_tool_calls: int = 4,
        model: str | None = None,
        llm_config_override: dict | None = None,
        system_prompt_override: str | None = None,
    ) -> dict[str, Any]:
        """Run an LLM turn that may ask for registered tools.

        The method owns the provider tool-call loop: expose schemas, execute each
        requested tool locally, append role="tool" results back into the message
        history, and return the final model content plus tool/retrieval logs for
        downstream writer flows.
        """
        if not self._tools_for_agent:
            return {
                "content": self._call(
                    turns,
                    model=model,
                    llm_config_override=llm_config_override,
                    system_prompt_override=system_prompt_override,
                ),
                "tool_calls": [],
                "tool_results": [],
                "retrieval_queries": [],
            }

        tools = get_tool_schemas(self._tools_for_agent)
        if not tools:
            return {
                "content": self._call(
                    turns,
                    model=model,
                    llm_config_override=llm_config_override,
                    system_prompt_override=system_prompt_override,
                ),
                "tool_calls": [],
                "tool_results": [],
                "retrieval_queries": [],
            }

        messages = self._build_messages(turns, system_prompt_override=system_prompt_override)
        override = dict(llm_config_override) if llm_config_override else {}
        if model is not None:
            override["model"] = model
        llm = get_llm(llm_config_override=override if override else None)

        tool_calls_used = 0
        tool_call_log: list[dict[str, Any]] = []
        tool_result_log: list[dict[str, Any]] = []
        retrieval_queries: list[str] = []
        
        # ReAct loop: model can iteratively request tools; we execute calls, append
        # tool outputs back into messages, and stop when no tool_calls are returned
        # (or when max_tool_calls budget is reached for this turn).
        while True:
            token = current_agent_label.set(self._log_display)
            try:
                resp = llm.complete_with_tools(messages, tools=tools)
            except LLMCallError as exc:
                model_label = (
                    f"{exc.model} ({exc.actual_model})" if exc.actual_model else exc.model
                )
                self._logger.log(
                    f"[{self._log_display}] LLM call failed (model={model_label}): {exc}",
                    level="error",
                )
                raise
            finally:
                current_agent_label.reset(token)

            content = _strip_think_block(resp.get("content") or "")
            tool_calls = resp.get("tool_calls") or []
            if not tool_calls:
                return {
                    "content": content,
                    "tool_calls": tool_call_log,
                    "tool_results": tool_result_log,
                    "retrieval_queries": retrieval_queries,
                }

            assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
            assistant_tool_calls: list[dict[str, Any]] = []
            messages.append(assistant_message)

            for call in tool_calls:
                if tool_calls_used >= max_tool_calls:
                    messages.append(
                        {
                            "role": "user",
                            "content": "Tool call limit reached. Continue without additional tool calls.",
                        }
                    )
                    break

                call_id = getattr(call, "id", None)
                if call_id is None and isinstance(call, dict):
                    call_id = call.get("id")
                fn = getattr(call, "function", None)
                if fn is None and isinstance(call, dict):
                    fn = call.get("function")
                fn = fn or {}
                name = getattr(fn, "name", None)
                if name is None and isinstance(fn, dict):
                    name = fn.get("name")
                arguments = getattr(fn, "arguments", None)
                if arguments is None and isinstance(fn, dict):
                    arguments = fn.get("arguments")
                if not name:
                    continue
                
                # Normalize and persist assistant tool_calls in message history so
                # subsequent role="tool" messages have a valid parent call id.
                assistant_tool_calls.append(
                    {
                        "id": call_id or f"tool_call_{tool_calls_used}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments or {}),
                        },
                    }
                )

                result = execute_tool_call(
                    tools_for_agent=self._tools_for_agent,
                    tool_name=name,
                    arguments_raw=arguments,
                    context={
                        "session_id": session_id,
                        "agent_type": self.role,
                        "plan_generation": current_plan_generation.get(),
                    },
                )

                tool_call_log.append(
                    {
                        "agent": self.role,
                        "tool_name": name,
                        "arguments": arguments if isinstance(arguments, dict) else (arguments or "{}"),
                    }
                )
                tool_result_log.append(
                    {
                        "agent": self.role,
                        "tool_name": name,
                        "ok": bool(result.get("ok")),
                        "error": result.get("error"),
                        "payload": result,
                    }
                )
                result_query = result.get("query")
                if isinstance(result_query, str) and result_query.strip():
                    retrieval_queries.append(result_query)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id or f"tool_call_{tool_calls_used}",
                        "name": name,
                        "content": format_tool_result_for_llm(result),
                    }
                )
                tool_calls_used += 1

            if assistant_tool_calls:
                assistant_message["tool_calls"] = assistant_tool_calls
