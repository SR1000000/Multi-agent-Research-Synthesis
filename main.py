import asyncio
import argparse
import os
import re
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.graph import build_graph
from src.llm.llm import current_session_id, init_from_config
from src.logging.logger import AgentLogger, VALIDATION_ERRORS_DIR
from src.memory import get_database
from src.memory.objectstore import (
    DEFAULT_OBJECT_STORE_CONFIG,
    LocalObjectStore,
    R2ObjectStore,
)
from src.processing.export.pandoc_builder import PandocBuilder
from src.processing.chunker import get_text_chunker
from src.processing.document import DocProcessor
from src.processing.embedder.provider import get_text_embedder
from src.processing.context.contextualizer import Contextualizer, ContextConfig
from src.processing.context.document import DocumentContextualizer, DocumentContextConfig
from src.state import MAX_CYCLES, make_initial_review_state

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"
from src.retriever import Retriever
from src.tools.registry import build_tool_registry

DEFAULT_QUERY      = "Explain this paper to an audience of laypeople"
DEFAULT_SOURCE_PDF = "./.samples/Transformers.pdf"

_PROCESSOR_BACKEND_ALIASES = {
    "llama": "llama_parse",
    "llama_parse": "llama_parse",
}

_TEXT_SPLITTER_ALIASES = {
    "none":     None,
    "semantic": "semantic",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-agent research presentation synthesizer"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help="Presentation query / audience description (default: %(default)s)",
    )
    parser.add_argument(
        "--llm-config",
        type=str,
        default=None,
        metavar="PATH",
        help="YAML file with providers and LiteLLM Router settings (default: src/llm/config.dev.yaml)",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        nargs="+",
        metavar="PATH",
        default=[DEFAULT_SOURCE_PDF],
        help="One or more PDF paths to analyse (default: %(default)s)",
    )
    parser.add_argument(
        "--processor",
        type=str,
        choices=sorted(_PROCESSOR_BACKEND_ALIASES.keys()),
        default="llama_parse",
        help=(
            "Document processor backend. `llama_parse` is the only supported "
            "processor in the current build; `llama` is accepted as a legacy alias "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--text-splitter",
        type=str,
        choices=sorted(_TEXT_SPLITTER_ALIASES.keys()),
        default="semantic",
        help=(
            "Text splitter backend for llama_parse document chunking "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Pause after each document extraction and require user confirmation to continue",
    )
    parser.add_argument(
        "--logging",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable Langfuse logging",
    )
    parser.add_argument(
        "--max-slides",
        type=int,
        default=15,
        help="Soft target for number of slides (Planner may adjust based on content density; default: %(default)s)",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=MAX_CYCLES,
        help="Maximum number of critic/rewrite cycles (default: %(default)s)",
    )

    parser.add_argument(
        "--object-store",
        type=str,
        choices=["local", "r2"],
        default=None,
        help="Object store type: 'local' for local filesystem, 'r2' for Cloudflare R2 (default: R2 with local fallback)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        metavar="PATH",
        help="Directory where the generated PPTX will be written (default: %(default)s)",
    )
    parser.add_argument(
        "--reference-doc",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a .pptx or .potx template passed to Pandoc's --reference-doc option "
            "(default: src/processing/export/reference.pptx bundled template)"
        ),
    )
    parser.add_argument(
        "--skip-supervisor",
        action="store_true",
        default=False,
        help="Skip supervisor/critic review cycles and export proto-slides directly via Pandoc",
    )
    parser.add_argument(
        "--force-replan",
        action="store_true",
        default=False,
        help=(
            "Test/debug only: at critic/rewrite cap, force up to two replans then allow "
            "normal acceptance. For exercising replan orchestration without the Supervisor "
            "LLM choosing replan."
        ),
    )
    parser.add_argument(
        "--force-accept-first-plan",
        action="store_true",
        default=False,
        help=(
            "If the critic/rewrite cycle cap is reached on the first presentation plan "
            "and the supervisor would replan or reject, force acceptance and proceed to export."
        ),
    )
    parser.add_argument(
        "--no-cache-control",
        action="store_true",
        default=False,
        help="Disable prompt cache_control sent to the LLM provider (contextualizer)",
    )
    parser.add_argument(
        "--no-context-batching",
        action="store_true",
        default=False,
        help=(
            "Disable batch LLM calls in contextualization; "
            "items are sent to the LLM one at a time sequentially"
        ),
    )
    return parser.parse_args()


def _get_callbacks(args, logger: AgentLogger, session_id: str):
    callbacks = []

    if args.logging is False:
        os.environ["LANGFUSE_ENABLED"] = "false"
        from langfuse.decorators import langfuse_context
        langfuse_context.configure(enabled=False)
    else:
        # Set the ContextVar so every router.completion() call in llm.py
        # injects session_id into its metadata, which the LiteLLM Langfuse
        # callback reads to tag litellm-completion traces to this session.
        current_session_id.set(session_id)
        langfuse_handler = logger.get_langgraph_handler(session_id=session_id)
        callbacks.append(langfuse_handler)
    return callbacks, logger


def _configure_llm(args: argparse.Namespace) -> None:
    init_from_config(config_path=args.llm_config)


def _make_object_store(args: argparse.Namespace, logger: AgentLogger) -> Any:
    """Create the object store selected for this run."""
    if args.object_store == "local":
        logger.log("Using LocalObjectStore for image storage (explicitly selected)", level="info")
        return LocalObjectStore(config=DEFAULT_OBJECT_STORE_CONFIG)

    if args.object_store == "r2":
        logger.log("Using R2ObjectStore for image storage (explicitly selected)", level="info")
        return R2ObjectStore(config=DEFAULT_OBJECT_STORE_CONFIG)

    try:
        logger.log("Using R2ObjectStore for image storage (default behavior)", level="info")
        return R2ObjectStore(config=DEFAULT_OBJECT_STORE_CONFIG)
    except Exception as exc:
        logger.log(
            f"R2ObjectStore initialization failed: {exc}. Falling back to LocalObjectStore.",
            level="warning",
        )
        return LocalObjectStore(config=DEFAULT_OBJECT_STORE_CONFIG)


def _process_document(
    pdf_path_str: str,
    args: argparse.Namespace,
    logger: AgentLogger,
    db: Any,
    object_store: Any,
    embedder: Any,
) -> tuple[Any, str]:
    pdf_path = Path(pdf_path_str)
    if not pdf_path.exists():
        sys.exit(f"error: PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        sys.exit(f"error: file does not have a .pdf extension: {pdf_path}")

    _t0 = time.perf_counter()

    processor_backend = _PROCESSOR_BACKEND_ALIASES[args.processor]
    chunker_name      = _TEXT_SPLITTER_ALIASES[args.text_splitter]

    text_chunker = get_text_chunker(chunker_name) if chunker_name else None
    contextualizer = Contextualizer(
        config=ContextConfig(
            model="context",
            cache_control=not args.no_cache_control,
            use_batch=not args.no_context_batching,
        ),
        object_store=object_store,
        logger=logger,
    )
    document_contextualizer = DocumentContextualizer(
        config=DocumentContextConfig(model="context"),
        logger=logger,
    )
    processor = DocProcessor(
        backend=processor_backend,
        text_chunker=text_chunker,
        db=db,
        contextualizer=contextualizer,
        document_contextualizer=document_contextualizer,
        embedder=embedder,
        logger=logger,
        object_store=object_store,
    )
    artifacts = asyncio.run(processor.process_document(str(pdf_path)))

    _pdf_elapsed = time.perf_counter() - _t0

    if artifacts and artifacts.chunk_count > 0:
        print(
            f"[preprocessing] {pdf_path.name} completed in {_pdf_elapsed:.2f}s",
            flush=True,
        )

    if args.interactive:
        try:
            response = input(
                f"Finished processing {pdf_path.name}. Press Enter to continue, or 'q' to quit: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            sys.exit("\nAborted.")
        if response == "q":
            sys.exit("Execution stopped by user.")

    status = (
        "Processed"
        if artifacts and artifacts.chunk_count > 0
        else "FAILED TO PROCESS (running without this document)"
    )

    if artifacts:
        msg = (
            f"[preprocessing] {pdf_path.name}: {status} "
            f"(images={artifacts.image_count}, tables={artifacts.table_count}, "
            f"equations={artifacts.equation_count}, chunks={artifacts.chunk_count})"
        )
    else:
        msg = f"[preprocessing] {pdf_path.name}: {status}"

    return artifacts, msg


def _build_initial_state(
    args: argparse.Namespace,
    preprocessing_messages: list[str],
    doc_ids: list[str],
    paper_titles: list[str],
    session_id: str,
) -> dict:
    return {
        "query":             args.query or DEFAULT_QUERY,
        "session_id":        session_id,
        "created_at":        datetime.now(timezone.utc).isoformat(),
        "doc_ids":           doc_ids,
        "paper_titles":      paper_titles,
        "max_slides":        args.max_slides,
        "skip_supervisor":   args.skip_supervisor,
        "plan_number":       1,
        "force_replan_at_max_cycles":        args.force_replan,
        "force_accept_first_plan_at_cap":    args.force_accept_first_plan,
        "slide_numbers":     [],
        "presentation_plan": None,
        "review":            make_initial_review_state(max_cycles=args.max_cycles),
        "retrieval_queries": [],
        'tool_calls':       [],
        'tool_results':     [],
        "slides_written":    [],
        "critic_results":    [],
        "review_summaries":  [],
        "messages":          preprocessing_messages,
        "errors":            [],
    }

def _sanitize_filename(name: str) -> str:
    """Keep only alphanumeric, spaces, hyphens and underscores. Trim to length."""
    if not name:
        return ""
    # Replace invalid chars with underscore
    safe = "".join(ch if ch.isalnum() or ch in (" ", "-", "_") else "_" for ch in name)
    # Collapse multiple underscores/spaces and switch spaces to underscores
    safe = re.sub(r"[ _]+", "_", safe).strip("_")
    return safe[:64]


def _partial_deck_warnings(messages: list[str]) -> list[str]:
    """Extract final warnings that indicate skipped groups or a partial deck export."""
    return [
        msg for msg in messages
        if "RETRIES EXHAUSTED" in msg or "PARTIAL DECK" in msg
    ]


def _report_final_warnings(messages: list[str]) -> None:
    """Print a compact end-of-run warning summary for partial deck outcomes."""
    warnings = _partial_deck_warnings(messages)
    if not warnings:
        return

    print("\n--- Final Warnings ---")
    for msg in warnings:
        print(msg)


def main() -> None:
    args       = _parse_args()
    logger     = AgentLogger()

    reference_doc: Path | None = None
    if args.reference_doc:
        _ref = Path(args.reference_doc).expanduser().resolve()
        if not _ref.exists():
            logger.log(
                f"[export] --reference-doc not found: {_ref}. Continuing without template.",
                level="warning",
            )
        elif _ref.suffix.lower() not in {".pptx", ".potx"}:
            logger.log(
                f"[export] --reference-doc should point to a .pptx or .potx file, got: {_ref}. "
                "Continuing without template.",
                level="warning",
            )
        else:
            reference_doc = _ref

    session_id = str(uuid.uuid4())
    output_dir = Path(args.output_dir).expanduser().resolve()

    # Clear validation error dumps from previous run
    if VALIDATION_ERRORS_DIR.exists():
        shutil.rmtree(VALIDATION_ERRORS_DIR)
    VALIDATION_ERRORS_DIR.mkdir(exist_ok=True)

    _configure_llm(args)
    callbacks, logger = _get_callbacks(args, logger, session_id)
    object_store = _make_object_store(args, logger)

    with get_database() as db:
        # Clear deck generation artifacts so each run starts clean (ingestion tables unchanged).
        db.clear_proto_slides()
        db.clear_slide_review_events()

        doc_ids: list[str] = []
        paper_titles: list[str] = []
        preprocessing_messages: list[str] = []
        seen_doc_ids: set[str] = set()

        embedder = get_text_embedder()
        retriever = Retriever(db, embedder)
        
        tool_registry = build_tool_registry(retriever=retriever, research_db=db)
        agent_tool_allowlist = {
            "slide_writer": ["retrieve_artifacts"],
            "planner": [],
            "critic": [],
            "supervisor": [],
            "parse_supervisor": [],
            "research_to_slide": [],
        }

        for pdf_path_str in args.pdf:
            artifacts, msg = _process_document(
                pdf_path_str,
                args,
                logger,
                db,
                object_store,
                embedder,
            )
            preprocessing_messages.append(msg)
            if not artifacts:
                continue

            if artifacts.doc_id in seen_doc_ids:
                preprocessing_messages.append(
                    f"[preprocessing] {Path(pdf_path_str).name}: duplicate document skipped "
                    f"(doc_id={artifacts.doc_id})"
                )
                continue

            seen_doc_ids.add(artifacts.doc_id)
            doc_ids.append(artifacts.doc_id)
            title = (
                Path(artifacts.source_path).stem
                if getattr(artifacts, "source_path", None)
                else artifacts.doc_id
            )
            paper_titles.append(title)

        if not doc_ids:
            sys.exit("error: No documents were successfully processed. Exiting.")

        initial_state = _build_initial_state(
            args, preprocessing_messages, doc_ids, paper_titles, session_id
        )

        graph = build_graph(
            tool_registry=tool_registry,
            agent_tool_allowlist=agent_tool_allowlist,
        )
        
        final_state = initial_state
        try:
            # Use streaming to capture the state at each step, allowing us to recover logs if a crash occurs.
            for event in graph.stream(
                initial_state,
                config={"callbacks": callbacks, "recursion_limit": 100},
                stream_mode="values",
            ):
                final_state = event
        except Exception as exc:
            print(f"\n[!] Graph encountered an error mid-flight: {exc}")
            print("    Attempting to recover partial logs...")

            ve_files = list(VALIDATION_ERRORS_DIR.glob("*.json"))
            if ve_files:
                print(f"\n[validation] {len(ve_files)} error dump(s) written to {VALIDATION_ERRORS_DIR}/")

            print("\n--- Agent Log ---")
            for msg in final_state.get("messages", []):
                print(msg)

        raw_name = paper_titles[0] if paper_titles else session_id
        presentation_plan = final_state.get("presentation_plan")
        plan_title = ""
        plan_subtitle = ""
        if presentation_plan and hasattr(presentation_plan, "title") and presentation_plan.title:
            raw_name = presentation_plan.title
            plan_title = presentation_plan.title
            plan_subtitle = getattr(presentation_plan, "subtitle", "") or ""
        safe_name = _sanitize_filename(raw_name) or session_id

        # Determine the prefix based on existing .pptx files in the output directory
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            pptx_count = 0
        else:
            pptx_count = len(list(output_dir.glob("*.pptx")))

        prefix = f"{pptx_count + 1} - "
        pptx_path = output_dir / f"{prefix}{safe_name}.pptx"
        if final_state.get("review", {}).get("export_ready"):
            try:
                out = PandocBuilder(
                    output_path=pptx_path,
                    db=db,
                    title=plan_title,
                    subtitle=plan_subtitle,
                    object_store=object_store,
                    reference_doc=reference_doc,
                ).build()
                print(f"\n[export] Presentation saved -> {out}")
            except ValueError as exc:
                print(f"\n[export] Could not generate PPTX: {exc}")
        else:
            print("\n[export] Skipped because supervisor did not accept the deck.")

        _report_final_warnings(final_state.get("messages", []))

    if logger:
        logger.flush()


if __name__ == "__main__":
    main()
