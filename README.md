# Multi-Agent Research Synthesis

LangGraph-coordinated research workflow:
Document Ingestion -> Lead Researcher -> Editor -> Critic loop.

## Setup

### 1. Python 3.11+

```bash

```

### 2. Virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/Mac: source .venv/Scripts/activate
pip install -r requirements.txt
```

Note: installing `transformers` and `sqlite-vec` can take longer because they bring specialized document-processing and vector-search dependencies.
The system now uses **LlamaParse** natively by default for high-quality OCR, formula extraction, and chunking.
sqlite-vec provides fast, local vector similarity search directly within SQLite.

### 3. API keys

```bash
copy .env.sample .env
```

Edit `.env` — replace the placeholder values with your LLM Provider's API key. We have options for

- [OpenRouter API](https://openrouter.ai/keys)
- [Ollama Cloud API](https://ollama.com/settings/keys)
- [Google AI Studio](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started.ipynb)
- [LlamaCloud API](https://cloud.llamaindex.ai/api-keys)

### 4. Langfuse Logging Setup

To enable observability, ensure the following API keys are set in your `.env` file (you can get these from your Langfuse project settings):

```env
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_BASE_URL="https://cloud.langfuse.com"
```

The codebase uses `langfuse` which will automatically pick up these environment variables to trace agent runs.

### 5. (Optional) Change model

Edit `--model` argument when running project.

## Run

```bash
# To use Ollama Cloud:
python main.py --ollama

# To use OpenRouter:
python main.py --open-router

# To use Google Gemini (default):
python main.py --gemini
```

### Document processor

While there are multiple backends for document processing are implemented, due to requirement conflicts only LlamaParse is available with the provided requirements.txt. Other backends will need additional dependencies, and likely separate environments.

For LlamaParse, set **`LLAMA_CLOUD_API_KEY`** in your environment.

Optional **`--text-splitter`** controls chunking after parse: `none` (single chunk from full text) or `semantic` (default, semantic splitter). Only use this if the document processor does not natively support chunking (LLamaparse for now).

### Cloud storage

Project has an optional [Cloudflare R2 Storage](https://developers.cloudflare.com/r2/get-started/s3/) for images extracted from document processors and a local storage. To use the cloud store, set up your API keys and credentials as follow:

```
CLOUDFLARE_ACCOUNT_ID=your_key_here
R2_ACCESS_KEY_ID=your_key_here
R2_SECRET_ACCESS_KEY=your_key_here
R2_BUCKET_NAME=multiagentsynthesis
```

### Optional Commandline Arguments

The PDF input defaults to `Transformers.pdf` in the .samples directory. You can change this by adding `--pdf "Path to your PDF file here"` or editing `DEFAULT_SOURCE_PDF` in `main.py`.

You change the research query by adding `--query "Your question here"` or editing `DEFAULT_QUERY` in `main.py`.

Adding the argument `-i` or `--interactive` adds a prompt for whether the user wants to continue, which pops up if a document is extracted and after the document extraction process is complete.

Adding `--use-db` (or `--skip-processing`) skips the document processing and instead attempts to load the parsed PDF chunks and metadata directly from the `data/research.db` SQLite database if it exists, saving valuable API and compute time during iterative runs pipeline tuning.

## Graph Flow

```
START → lead_researcher (selects chunk indices)
                  │
                  ├─ next=="continue" → editor → critic ─┐
                  │         (uses selected chunks)       │
                  └◄─────────────────────────────────────┘
                  │
                  └- next=="done" → END
```

## Telemetry & Logging

The system implements a dual-layer observability strategy to maintain clean agent logic while ensuring comprehensive tracing:

- **Workflow Tracing**: Captures the high-level orchestration, state transitions, and routing overhead as the research document flows between the specialized agents.
- **Cognitive Tracing**: Instruments the underlying LLM calls to capture precise generation metrics (latency, token usage) and raw prompt details completely independently of the graph execution.
