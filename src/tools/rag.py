from __future__ import annotations

import json
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.memory.research.database import ResearchDatabase
from src.retriever import Retriever, RetrievedItem

DEFAULT_CONTEXT_K = 8


class RetrieveArtifactsArgs(BaseModel):
    query: str = Field(min_length=1, description="Natural language query for retrieval.")
    k: int = Field(default=DEFAULT_CONTEXT_K, ge=1, le=20, description="Maximum artifacts to return.")
    strategy: Literal["fusion", "semantic", "keyword"] = Field(
        default="fusion",
        description="Retrieval strategy to use.",
    )

class RetrieveArtifactsResult(BaseModel):
    ok: bool
    tool_name: Literal["retrieve_artifacts"]
    query: str
    latency_ms: int
    error: str | None = None
    artifacts: list[ArtifactRecord]
    provenance: RetrieveArtifactsProvenance

class ArtifactRecord(BaseModel):
    kind: Literal["chunk", "table", "equation", "image"]
    id: str
    document_id: str
    score: float | None = None
    content: str

class RetrieveArtifactsProvenance(BaseModel):
    strategy: Literal["fusion", "semantic", "keyword"]
    top_k: int
    document_ids: list[str]
    call_id: str | None = None

OUTPUT_SCHEMA = RetrieveArtifactsResult.model_json_schema()


def _build_artifact_content(record: dict[str, Any], *, normalized: bool = False) -> str:
    kind = str(record.get("kind", "chunk"))
    lines: list[str] = []
    caption = ""
    contextualized = ""
    raw_value = ""

    if normalized:
        raw_value = str(record.get("text") or "")
        contextualized = str(record.get("contextualized_text") or "")
        caption = str(record.get("caption") or "")
        if kind in {"image", "table", "equation"} and caption.strip():
            lines.append(f"caption: {caption}")
        if contextualized.strip():
            lines.append(f"contextualized description: {contextualized}")
        lines.append(f"value: {raw_value}")
        return "\n".join(lines)

    if kind == "chunk":
        raw_value = str(record.get("text") or record.get("chunk_text") or "")
        contextualized = str(
            record.get("contextualized_text")
            or record.get("chunk_contextualized_text")
            or ""
        )
    elif kind == "table":
        raw_value = str(record.get("table_content") or record.get("content") or "")
        caption = str(record.get("table_caption") or record.get("caption") or "")
        contextualized = str(
            record.get("table_contextualized_text")
            or record.get("contextualized_text")
            or ""
        )
    elif kind == "equation":
        raw_value = str(record.get("equation_text") or record.get("text") or "")
        caption = str(record.get("equation_caption") or record.get("caption") or "")
        contextualized = str(
            record.get("equation_contextualized_text")
            or record.get("contextualized_text")
            or ""
        )
    elif kind == "image":
        raw_value = str(record.get("image_storage_path") or record.get("storage_path") or "")
        caption = str(record.get("image_caption") or record.get("caption") or "")
        contextualized = str(
            record.get("image_contextualized_text")
            or record.get("contextualized_text")
            or ""
        )

    if kind in {"image", "table", "equation"} and caption.strip():
        lines.append(f"caption: {caption}")
    if contextualized.strip():
        lines.append(f"contextualized description: {contextualized}")
    lines.append(f"value: {raw_value}")
    return "\n".join(lines)


def retrieve_artifacts(
    args: RetrieveArtifactsArgs,
    *,
    retriever: Retriever,
    research_db: ResearchDatabase,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    context = {} if context is None else context
    query = args.query.strip()
    if not query:
        return {
            "ok": False,
            "tool_name": "retrieve_artifacts",
            "query": args.query,
            "latency_ms": 0,
            "error": "Query must not be empty.",
            "artifacts": [],
            "provenance": {
                "strategy": args.strategy,
                "top_k": args.k,
                "document_ids": [],
                "call_id": None,
            },
        }

    call_id = uuid.uuid4().hex
    if context is not None:
        context["last_retrieval_call_id"] = call_id

    if args.strategy == "semantic":
        items: list[RetrievedItem] = retriever.semantic_retrieve(query, args.k)
    elif args.strategy == "keyword":
        items = retriever.keywords_retrieve(query, args.k)
    else:
        items = retriever.fusion_retrieve(query, args.k)

    normalized_rows = research_db.load_normalized_artifacts_for_keys(items)

    session_id = context.get("session_id")
    agent_type = context.get("agent_type", "writer")
    if session_id:
        try:
            research_db.save_session_retrieval_batch(
                session_id=session_id,
                call_id=call_id,
                items=items,
                query=query,
                strategy=args.strategy,
                agent_type=agent_type,
            )
        except Exception as exc:
            research_db._logger.log(
                (
                    "[retrieve_artifacts] Failed to persist retrieval ledger "
                    f"call_id={call_id} session_id={session_id}: {type(exc).__name__}: {exc}"
                ),
                level="warning",
            )

    artifacts: list[dict[str, Any]] = []
    doc_ids: set[str] = set()
    for row in normalized_rows:
        kind = str(row.get("kind", "chunk"))
        item_id = str(row.get("artifact_id") or "")
        document_id = str(row.get("document_id") or "")
        score_raw = row.get("score")
        score = float(score_raw) if score_raw is not None else None
        doc_ids.add(document_id)
        content = _build_artifact_content(row, normalized=True)
        artifact: dict[str, Any] = {
            "kind": kind,
            "id": item_id,
            "document_id": document_id,
            "score": score,
            "content": content,
        }
        artifacts.append(artifact)

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
            "call_id": call_id,
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
        "output_schema": OUTPUT_SCHEMA,
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
