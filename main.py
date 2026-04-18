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

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"

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
) -> tuple[Any, str]:
    pdf_path = Path(pdf_path_str)
    if not pdf_path.exists():
        sys.exit(f"error: PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        sys.exit(f"error: file does not have a .pdf extension: {pdf_path}")

    _t0 = time.perf_counter()
    embedder = get_text_embedder()
    processor_backend = _PROCESSOR_BACKEND_ALIASES[args.processor]
    chunker_name      = _TEXT_SPLITTER_ALIASES[args.text_splitter]

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
        # Clear generated proto slides for a fresh deck.
        db.clear_proto_slides()

        doc_ids: list[str] = []
        paper_titles: list[str] = []
        preprocessing_messages: list[str] = []
        seen_doc_ids: set[str] = set()

        for pdf_path_str in args.pdf:
            artifacts, msg = _process_document(
                pdf_path_str,
                args,
                logger,
                db,
                object_store,
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

        graph = build_graph()
        final_state = initial_state
        try:
            # Use streaming to capture the state at each step, allowing us to recover logs if a crash occurs.
            for event in graph.stream(
                initial_state,
                config={"callbacks": callbacks},
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
        safe_name = _sanitize_filename(raw_name) or session_id
        pptx_path = output_dir / f"{safe_name}.pptx"
        try:
            out = PandocBuilder(output_path=pptx_path, db=db).build()
            print(f"\n[export] Presentation saved -> {out}")
        except ValueError as exc:
            print(f"\n[export] Could not generate PPTX: {exc}")

        _report_final_warnings(final_state.get("messages", []))

    if logger:
        logger.flush()


if __name__ == "__main__":
    main()
