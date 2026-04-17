import argparse
import shutil
import sys
from typing import Any
import time
from pathlib import Path
import os
import re
from src.memory import get_database
from src.graph import build_graph
from src.llm.llm import init_from_config, current_session_id
from src.processing.chunker import get_text_chunker
from src.processing.document import DocProcessor
from src.processing.embedder.provider import get_text_embedder
import uuid
from datetime import datetime, timezone
from src.logging.logger import AgentLogger, VALIDATION_ERRORS_DIR
from src.memory.objectstore import LocalObjectStore, R2ObjectStore, DEFAULT_OBJECT_STORE_CONFIG

OUTPUT_DIR = Path(__file__).parent / "output"  # PowerPoint files land here

DEFAULT_QUERY      = "Explain this paper to an audience of laypeople"
DEFAULT_SOURCE_PDF = "./.samples/Transformers.pdf"

_PROCESSOR_BACKEND_ALIASES = {
    "llama":      "llama_parse",
    "llama_parse": "llama_parse",
    "docling":    "docling",
    "lighton":    "lighton",
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
        default="llama",
        help="Document processor backend (default: %(default)s)",
    )
    parser.add_argument(
        "--text-splitter",
        type=str,
        choices=sorted(_TEXT_SPLITTER_ALIASES.keys()),
        default="none",
        help=(
            "Text splitter backend for document chunking "
            "(defaults to 'semantic' for llama_parse) (default: %(default)s)"
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
        "--object-store",
        type=str,
        choices=["local", "r2"],
        default=None,
        help="Object store type: 'local' for local filesystem, 'r2' for Cloudflare R2 (default: R2 with local fallback)",
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

def _process_document(args: argparse.Namespace, logger: AgentLogger, db: Any) -> tuple[Any, str]:
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(f"error: PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        sys.exit(f"error: file does not have a .pdf extension: {pdf_path}")

    _t0 = time.perf_counter()

    # Initialize object store based on CLI argument
    if args.object_store == "local":
        object_store = LocalObjectStore(config=DEFAULT_OBJECT_STORE_CONFIG)
        logger.log("Using LocalObjectStore for image storage (explicitly selected)", level="info")
    elif args.object_store == "r2":
        r2_config = DEFAULT_OBJECT_STORE_CONFIG
        object_store = R2ObjectStore(config=r2_config)
        logger.log("Using R2ObjectStore for image storage (explicitly selected)", level="info")
    else:  # Default behavior
        try:
            r2_config = DEFAULT_OBJECT_STORE_CONFIG
            object_store = R2ObjectStore(config=r2_config)
            logger.log("Using R2ObjectStore for image storage (default behavior)", level="info")
        except Exception as e:
            logger.log(f"R2ObjectStore initialization failed: {str(e)}. Falling back to LocalObjectStore.", level="warning")
            object_store = LocalObjectStore(config=DEFAULT_OBJECT_STORE_CONFIG)

    db       = get_database()
    embedder = get_text_embedder()
    processor_backend = _PROCESSOR_BACKEND_ALIASES[args.processor]
    chunker_name      = _TEXT_SPLITTER_ALIASES[args.text_splitter]

    # default to 'semantic' chunking if using LlamaParser and no specific splitter chosen
    if not chunker_name and processor_backend == "llama_parse":
        chunker_name = "semantic"

    text_chunker = get_text_chunker(chunker_name) if chunker_name else None
    processor = DocProcessor(
        backend=processor_backend,
        text_chunker=text_chunker,
        db=db,
        embedder=embedder,
        logger=logger,
        object_store=object_store,
    )
    artifacts = processor.process_document(str(pdf_path))

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
        "slide_numbers":     [],
        "presentation_plan": None,
        "slides_written":    [],
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
    return safe[:150]


def _partial_deck_warnings(messages: list[str]) -> list[str]:
    """Extract final warnings that indicate skipped groups or a partial deck export."""
    return [
        msg for msg in messages
        if "RETRIES EXHAUSTED" in msg or "PARTIAL DECK" in msg
    ]


def main() -> None:
    args       = _parse_args()
    logger     = AgentLogger()
    session_id = str(uuid.uuid4())

    # Clear validation error dumps from previous run
    if VALIDATION_ERRORS_DIR.exists():
        shutil.rmtree(VALIDATION_ERRORS_DIR)
    VALIDATION_ERRORS_DIR.mkdir(exist_ok=True)

    _configure_llm(args)
    callbacks, logger = _get_callbacks(args, logger, session_id)

    object_store = _make_object_store(args, logger)

    # ------------------------------------------------------------------
    # Ingest each PDF into research.db
    # ------------------------------------------------------------------
    doc_ids:              list[str] = []
    paper_titles:         list[str] = []
    preprocessing_messages: list[str] = []

    for pdf_path_str in args.pdf:
        artifacts, msg = _process_document(pdf_path_str, args, logger, object_store)
        preprocessing_messages.append(msg)
        if artifacts:
            doc_ids.append(artifacts.doc_id)
            title = (
                artifacts.paper_metadata.title
                if artifacts.paper_metadata and artifacts.paper_metadata.title
                else Path(pdf_path_str).stem
            )
            paper_titles.append(title)

    if not doc_ids:
        sys.exit("error: No documents were successfully processed. Exiting.")

    # ------------------------------------------------------------------
    # Build and stream the graph
    # ------------------------------------------------------------------
    initial_state = _build_initial_state(
        args, preprocessing_messages, doc_ids, paper_titles, session_id
    )

    graph = build_graph()

    final_state = initial_state
    try:
        # Use streaming to capture the state at each step, allowing us to recover logs if a crash occurs
        for event in graph.stream(
            initial_state,
            config={"callbacks": callbacks},
            stream_mode="values",
        ):
            final_state = event
    except Exception as e:
        print(f"\n[!] Graph encountered an error mid-flight: {e}")
        print("    Attempting to recover partial logs...")

    ve_files = list(VALIDATION_ERRORS_DIR.glob("*.json"))
    if ve_files:
        print(f"\n[validation] {len(ve_files)} error dump(s) written to {VALIDATION_ERRORS_DIR}/")

    print("\n--- Agent Log ---")
    for msg in final_state.get("messages", []):
        print(msg)

    partial_warnings = _partial_deck_warnings(final_state.get("messages", []))
    if partial_warnings:
        print("\n[warning] Export will proceed with a partial deck.")
        for msg in partial_warnings:
            print(msg)

    partial_warnings = _partial_deck_warnings(final_state.get("messages", []))
    if partial_warnings:
        print("\n[warning] Export will proceed with a partial deck.")
        for msg in partial_warnings:
            print(msg)

    # ------------------------------------------------------------------
    # Export PPTX
    # ------------------------------------------------------------------
    from src.processing.export.pandoc_builder import PandocBuilder

    # Use first paper title, or fall back to session_id
    raw_name  = paper_titles[0] if paper_titles else session_id
    safe_name = _sanitize_filename(raw_name) or session_id

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pptx_path = OUTPUT_DIR / f"{safe_name}.pptx"
    try:
        with WIPDatabase() as wip_db:
            out = PandocBuilder(output_path=pptx_path, db=wip_db).build()
        print(f"\n[export] Presentation saved → {out}")
    except ValueError as exc:
        print(f"\n[export] Could not generate PPTX: {exc}")

    if logger:
        logger.flush()


if __name__ == "__main__":
    main()
