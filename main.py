import argparse
import sys
from typing import Any
import time
from pathlib import Path
import os
from src.memory import get_database
from src.graph import build_graph
import src.llm
from src.processing.chunker import get_text_chunker
from src.processing.document import DocProcessor
import uuid
from datetime import datetime, timezone
from src.llm import GLOBAL_CONFIG, Provider
from src.logging.logger import AgentLogger

DEFAULT_QUERY = "Explain what Transformers are and how they are so important to AI"
DEFAULT_SOURCE_PDF = "./.samples/Transformers.pdf"

_PROVIDER_FLAGS = {
    "ollama":     Provider.OLLAMA,
    "openrouter": Provider.OPENROUTER,
    "gemini":     Provider.GOOGLE_AI_STUDIO,
}

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
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ollama",     action="store_true", help="Use Ollama provider")
    group.add_argument("--openrouter", action="store_true", help="Use OpenRouter provider")
    group.add_argument("--gemini",     action="store_true", help="Use Google AI Studio (Gemini) provider (default)")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="Research query")
    parser.add_argument("--model", type=str, help="Override the default model for the provider")
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
        help="Text splitter backend for document chunking (default: %(default)s)",
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
    return parser.parse_args()

def _get_callbacks(args):
    callbacks = []
    logger = None

    if args.logging is False:
        os.environ["LANGFUSE_ENABLED"] = "false"
        from langfuse.decorators import langfuse_context
        langfuse_context.configure(enabled=False)
    else:
        logger = AgentLogger()
        langfuse_handler = logger.get_langgraph_handler()
        callbacks.append(langfuse_handler)
    return callbacks, logger

def _configure_llm(args: argparse.Namespace) -> None:
    # resolve which provider flag was set
    selected = next(
        (prov for flag, prov in _PROVIDER_FLAGS.items() if getattr(args, flag)),
        Provider.GOOGLE_AI_STUDIO,
    )
    GLOBAL_CONFIG.provider = selected
    if args.model:
        GLOBAL_CONFIG.model = args.model

def _process_document(args: argparse.Namespace) -> tuple[Any, str]:
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(f"error: PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        sys.exit(f"error: file does not have a .pdf extension: {pdf_path}")
    
    _t0 = time.perf_counter()
    db = get_database()

    processor_backend = _PROCESSOR_BACKEND_ALIASES[args.processor]
    chunker_name = _TEXT_SPLITTER_ALIASES[args.text_splitter]
    text_chunker = get_text_chunker(chunker_name) if chunker_name else None
    processor = DocProcessor(backend=processor_backend, text_chunker=text_chunker, db=db)
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
        'document_context': "",
        'source_chunks':    artifacts.source_chunks if artifacts else [],
        'doc_id':           artifacts.doc_id if artifacts else "unknown",
        'revision_history': [],
        'replan_history':   [],
        'messages':         [preprocessing_message],
        'errors':           [],
    }

def main() -> None:
    args = _parse_args()

    _configure_llm(args)
    callbacks, logger = _get_callbacks(args)

    artifacts, preprocessing_message = _process_document(args)
    initial_state = _build_initial_state(args, preprocessing_message, artifacts)

    graph = build_graph()
    
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

