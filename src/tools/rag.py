from __future__ import annotations

import json
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.memory.research.database import ResearchDatabase
from src.retriever import Retriever

DEFAULT_CONTEXT_K = 8


class RetrieveArtifactsArgs(BaseModel):
    query: str = Field(min_length=1, description="Natural language query for retrieval.")
    k: int = Field(default=DEFAULT_CONTEXT_K, ge=1, le=20, description="Maximum artifacts to return.")
    strategy: Literal["fusion", "semantic", "keyword"] = Field(
        default="fusion",
        description="Retrieval strategy to use.",
    )


def _render_text(kind: str, item_id: str, document_id: str, text: str) -> str:
    if kind == "equation":
        return f"equation id={item_id} document_id={document_id}\nLaTeX: {text}"
    return f"{kind} id={item_id} document_id={document_id}\n{text}"


def retrieve_artifacts(
    args: RetrieveArtifactsArgs,
    *,
    retriever: Retriever,
    research_db: ResearchDatabase,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    context = context or {}
    query = args.query.strip()
    if not query:
        return {
            "ok": False,
            "tool_name": "retrieve_artifacts",
            "query": args.query,
            "latency_ms": 0,
            "error": "Query must not be empty.",
            "artifacts": [],
            "provenance": {"strategy": args.strategy, "top_k": args.k, "document_ids": []},
        }

    if args.strategy == "semantic":
        items = retriever.semantic_retrieve(query, args.k)
    elif args.strategy == "keyword":
        items = retriever.keywords_retrieve(query, args.k)
    else:
        items = retriever.fusion_retrieve(query, args.k)

    artifacts: list[dict[str, Any]] = []
    doc_ids: set[str] = set()
    for it in items:
        doc_ids.add(it.document_id)
        artifact: dict[str, Any] = {
            "kind": it.kind,
            "id": it.id,
            "document_id": it.document_id,
            "score": it.score,
            "payload": {"text": it.text},
            "render_text": _render_text(it.kind, it.id, it.document_id, it.text),
        }
        if it.kind == "image":
            image = research_db.get_image(it.id)
            if image:
                artifact["payload"] = {
                    "caption": image.caption,
                    "storage_path": image.storage_path,
                    "mime_type": image.mime_type,
                    "page": image.page,
                }
                artifact["render_text"] = (
                    f"image id={it.id} document_id={it.document_id}\n"
                    f"caption={image.caption or ''}\n"
                    f"storage_path={image.storage_path or ''}"
                )
        elif it.kind == "table":
            table = research_db.get_table(it.id)
            if table:
                artifact["payload"] = {
                    "content": table.content,
                    "caption": table.title,
                    "page": table.page,
                }
        elif it.kind == "equation":
            equation = research_db.get_equation(it.id)
            if equation:
                artifact["payload"] = {
                    "latex": equation.latex_or_text,
                    "caption": equation.caption,
                    "page": equation.page,
                }
        artifacts.append(artifact)

    session_id = context.get("session_id")
    agent_type = context.get("agent_type", "writer")
    if session_id:
        for it in items:
            research_db.save_retrieved_chunk(
                item_id=it.id,
                kind=it.kind,
                document_id=it.document_id,
                text_content=it.text,
                score=it.score,
                session_id=session_id,
                agent_type=agent_type,
                query=query,
            )

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return {
        "ok": True,
        "tool_name": "retrieve_artifacts",
        "query": query,
        "latency_ms": elapsed_ms,
        "error": None,
        "artifacts": artifacts,
        "provenance": {
            "strategy": args.strategy,
            "top_k": args.k,
            "document_ids": sorted(doc_ids),
        },
    }


def build_retrieve_artifacts_tool(
    *,
    retriever: Retriever,
    research_db: ResearchDatabase,
) -> dict[str, Any]:
    schema = {
        "type": "function",
        "function": {
            "name": "retrieve_artifacts",
            "description": (
                "Retrieve the most relevant source artifacts from ingested documents. "
                "Use this when you need source-of-truth evidence before writing."
            ),
            "parameters": RetrieveArtifactsArgs.model_json_schema(),
        },
    }

    def _handler(arguments: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
        validated = RetrieveArtifactsArgs.model_validate(arguments)
        return retrieve_artifacts(
            validated,
            retriever=retriever,
            research_db=research_db,
            context=context,
        )

    return {
        "name": "retrieve_artifacts",
        "schema": schema,
        "handler": _handler,
        "prompt_snippet": (
            "Tool available: `retrieve_artifacts(query, k, strategy)`.\n"
            "- For this workflow, call this tool at least once before finalizing any draft to fact check yourself.\n"
            "- Start by calling it with the user query to gather initial evidence.\n"
            "- Prefer strategy='fusion' unless you need strict semantic/keyword behavior.\n"
            "- Use returned artifacts as source-of-truth and cite only retrieved evidence."
        ),
    }


def format_tool_result_for_llm(result: dict[str, Any], max_artifacts: int = 8) -> str:
    if not result.get("ok"):
        return json.dumps(result)
    trimmed = dict(result)
    artifacts = list(trimmed.get("artifacts", []))
    trimmed["artifacts"] = artifacts[:max_artifacts]
    return json.dumps(trimmed, ensure_ascii=True)
