from __future__ import annotations

import json
from typing import Any

from src.tools.rag import build_retrieve_artifacts_tool


def build_tool_registry(*, retriever: Any, research_db: Any) -> dict[str, dict[str, Any]]:
    rag_tool = build_retrieve_artifacts_tool(retriever=retriever, research_db=research_db)
    return {rag_tool["name"]: rag_tool}


def get_tool_schemas(
    tools_for_agent: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    schemas: list[dict[str, Any]] = []
    for tool in tools_for_agent.values():
        schemas.append(tool["schema"])
    return schemas


def execute_tool_call(
    *,
    tools_for_agent: dict[str, dict[str, Any]],
    tool_name: str,
    arguments_raw: str | dict[str, Any] | None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tool = tools_for_agent.get(tool_name)
    if not tool:
        return {
            "ok": False,
            "tool_name": tool_name,
            "error": f"Tool '{tool_name}' is not available for this agent.",
            "artifacts": [],
        }

    try:
        if arguments_raw is None:
            arguments = {}
        elif isinstance(arguments_raw, str):
            arguments = json.loads(arguments_raw) if arguments_raw.strip() else {}
        else:
            arguments = arguments_raw
    except Exception as exc:
        return {
            "ok": False,
            "tool_name": tool_name,
            "error": f"Invalid tool arguments JSON: {exc}",
            "artifacts": [],
        }

    try:
        return tool["handler"](arguments, context)
    except Exception as exc:
        return {
            "ok": False,
            "tool_name": tool_name,
            "error": f"Tool execution failed: {exc}",
            "artifacts": [],
        }


def resolve_agent_tools(
    tool_registry: dict[str, dict[str, Any]],
    allowed_tool_names: list[str],
) -> dict[str, dict[str, Any]]:
    return {
        name: tool_registry[name]
        for name in allowed_tool_names
        if name in tool_registry
    }


def get_tool_prompt_snippets(tools_for_agent: dict[str, dict[str, Any]]) -> list[str]:
    snippets: list[str] = []
    for tool in tools_for_agent.values():
        snippet = tool.get("prompt_snippet")
        if isinstance(snippet, str) and snippet.strip():
            snippets.append(snippet.strip())
    return snippets
