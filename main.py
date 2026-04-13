import argparse
import sys
from typing import Any
import time
from pathlib import Path
import os
import re
from src.memory import get_database
from src.graph import build_graph
from src.llm.llm import init_from_config
from src.processing.chunker import get_text_chunker
from src.processing.document import DocProcessor
from src.processing.embedder.provider import get_text_embedder
import uuid
from datetime import datetime, timezone
from src.logging.logger import AgentLogger
from src.memory.wip.database import WIPDatabase
from src.memory.objectstore import LocalObjectStore, R2ObjectStore, DEFAULT_OBJECT_STORE_CONFIG

OUTPUT_DIR = Path(__file__).parent / "output"  # PowerPoint files land here

DEFAULT_QUERY = "Explain what Transformers are and how they are so important to AI"
DEFAULT_SOURCE_PDF = "./.samples/Transformers.pdf"

_PROCESSOR_BACKEND_ALIASES = {
    "llama": "llama_parse",
    "llama_parse": "llama_parse",
    "docling": "docling",
    "lighton": "lighton",
}

_TEXT_SPLITTER_ALIASES = {
    "none": None,
    "semantic": "semantic",
}

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-agent research synthesis")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="Research query")
    parser.add_argument(
        "--llm-config",
        type=str,
        default=None,
        metavar="PATH",
        help="YAML file with providers and LiteLLM Router settings (default: src/llm/config.yaml)",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        metavar="PATH",
        default=DEFAULT_SOURCE_PDF,
        help="Path to the PDF to analyse (default: %(default)s)"
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
        help="Text splitter backend for document chunking (defaults to 'semantic' for llama_parse) (default: %(default)s)",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Pause after document extraction and require user confirmation to continue",
    )
    parser.add_argument(
        "--logging",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable Langfuse logging"
    )
    parser.add_argument(
        "--slides",
        action="store_true",
        help="Run the slide synthesis pipeline instead of research synthesis"
    )
    parser.add_argument(
        "--max-slides",
        type=int,
        default=12,
        help="Maximum number of slides to generate (default: %(default)s)",
    )
    parser.add_argument(
        "--object-store",
        type=str,
        choices=["local", "r2"],
        default=None,  # Default behavior remains as before (R2 with fallback to local)
        help="Object store type: 'local' for local filesystem, 'r2' for Cloudflare R2 (default: R2 with local fallback)"
    )
    return parser.parse_args()

def _get_callbacks(args, logger: AgentLogger):
    callbacks = []

    if args.logging is False:
        os.environ["LANGFUSE_ENABLED"] = "false"
        from langfuse.decorators import langfuse_context
        langfuse_context.configure(enabled=False)
    else:
        langfuse_handler = logger.get_langgraph_handler()
        callbacks.append(langfuse_handler)
    return callbacks, logger

def _configure_llm(args: argparse.Namespace) -> None:
    init_from_config(config_path=args.llm_config)

def _process_document(args: argparse.Namespace, logger: AgentLogger) -> tuple[Any, str]:
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

    db = get_database()
    embedder = get_text_embedder()
    processor_backend = _PROCESSOR_BACKEND_ALIASES[args.processor]
    chunker_name = _TEXT_SPLITTER_ALIASES[args.text_splitter]

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
        print(f"[preprocessing] PDF extraction/pipeline completed in {_pdf_elapsed:.2f}s", flush=True)

    if args.interactive:
        try:
            response = input("Press Enter to continue, or type 'q' to quit: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            sys.exit("\nAborted.")
        if response == "q":
            sys.exit("Execution stopped by user.")

    status = "Processed" if artifacts and artifacts.chunk_count > 0 else "FAILED TO PROCESS (running without documents)"

    if artifacts:
        preprocessing_message = (
            f"[preprocessing] {status} multimodal artifacts "
            f"(images={artifacts.image_count}, tables={artifacts.table_count}, "
            f"equations={artifacts.equation_count}, chunks={artifacts.chunk_count})"
        )
    else:
        preprocessing_message = f"[preprocessing] {status}"

    return artifacts, preprocessing_message

def _build_initial_state(args: argparse.Namespace, preprocessing_message: str, artifacts: Any) -> dict:
    return {
        'query':            args.query or DEFAULT_QUERY,
        'session_id':       str(uuid.uuid4()),
        'created_at':       datetime.now(timezone.utc).isoformat(),
        'revision_count':   0,
        'replan_count':     0,
        'plan':             None,
        'draft':            None,
        'critique':         None,
        'max_slides':       args.max_slides,
        'slide_numbers':    [],
        'document_context': "",
        'source_chunks':    artifacts.source_chunks if artifacts else [],
        'doc_id':           artifacts.doc_id if artifacts else "unknown",
        'paper_title':      (artifacts.paper_metadata.title if artifacts and artifacts.paper_metadata else "") or "",
        'revision_history': [],
        'replan_history':   [],
        'messages':         [preprocessing_message],
        'errors':           [],
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

def main() -> None:
    args = _parse_args()
    logger = AgentLogger()

    # Reset WIP database for the new run
    with WIPDatabase() as db:
        db.reset()

    _configure_llm(args)
    callbacks, logger = _get_callbacks(args, logger)

    artifacts, preprocessing_message = _process_document(args, logger)
    initial_state = _build_initial_state(args, preprocessing_message, artifacts)

    graph = build_graph(slides_mode=args.slides)
    
    final_state = initial_state
    try:
        # Use streaming to capture the state at each step, allowing us to recover logs if a crash occurs
        for event in graph.stream(
            initial_state, 
            config={"callbacks": callbacks},
            stream_mode="values"
        ):
            final_state = event
    except Exception as e:
        print(f"\n[!] Research Graph encountered an error mid-flight: {e}")
        print("    Attempting to recover partial logs...")

    print("\n--- Agent Log ---")
    for msg in final_state.get("messages", []):
        print(msg)


    if args.slides:
        from src.processing.export.pptx_builder import PptxBuilder
        
        # Use paper title if available, fallback to doc_id or session_id
        raw_name = final_state.get('paper_title') or final_state.get('doc_id') or final_state['session_id']
        safe_name = _sanitize_filename(raw_name)
        if not safe_name:
            safe_name = final_state['session_id']
            
        pptx_path = OUTPUT_DIR / f"{safe_name}.pptx"
        try:
            with WIPDatabase() as wip_db:
                out = PptxBuilder(output_path=pptx_path, db=wip_db).build()
            print(f"\n[export] Presentation saved → {out}")
        except ValueError as exc:
            print(f"\n[export] Could not generate PPTX: {exc}")
    else:
        print("\n--- Final Draft (Last Known State) ---")
        final_draft = final_state.get('draft')
        if final_draft:
            print(final_draft['document'])
        else:
            print('(no draft produced)')

    if logger:
        logger.flush()


if __name__ == "__main__":
    main()

