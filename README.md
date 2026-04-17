# Multi-Agent Research Presentation Synthesizer

LangGraph-coordinated pipeline that ingests one or more research PDFs and
produces a PowerPoint presentation driven by parallel AI agents:
**Planner → Plan Executor → Slide Writers** (fan-out/fan-in with retry).

## Setup

### 1. Python 3.11+

### 2. Virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

> Installing `sentence-transformers` and `sqlite-vec` can take longer because they bring specialized embedding and vector-search dependencies. The system uses **LlamaParse** by default for high-quality OCR, formula extraction, and chunking. `sqlite-vec` provides fast, local vector similarity search directly within SQLite.

### 3. API keys

Copy the sample and fill in your credentials:

```bash
copy .env.sample .env
```

See `.env.sample` for all supported keys (LLM providers, Langfuse, Cloudflare R2).

### 4. Langfuse Logging (optional)

To enable observability, set these keys in your `.env`:

```env
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_BASE_URL="https://cloud.langfuse.com"
```

Disable with `--no-logging` at runtime.

### 5. LLM providers and routing

Runtime LLM routing comes from a **LiteLLM Router** built from YAML. The app reads **`src/llm/config.yaml`** at startup.

#### Create your `config.yaml`

1. Copy the template: `src/llm/config.sample.yaml`
2. Fill in API keys and base URLs via `.env` using the `os.environ/VAR_NAME` placeholders
3. Adjust `router.providers` and `router.fallback_providers` to match your models

```bash
python main.py --llm-config path/to/your/config.yaml
```

The pipeline uses two router group aliases:

- **`slides`** — the primary model group, used by the Planner and Slide Writers
- **`fallback`** — fallback group used automatically when the primary group fails

Any provider and model string LiteLLM supports can be added to `config.yaml` following the same structure. See the [LiteLLM provider docs](https://docs.litellm.ai/docs/providers) for parameter names and provider-specific options.

---

## Running

```bash
python main.py --pdf path/to/paper.pdf
```

### Multiple PDFs

Pass multiple paths to generate a single presentation from several papers:

```bash
python main.py --pdf paper1.pdf paper2.pdf paper3.pdf
```

### Controlling the presentation

Use `--query` to specify the audience or framing:

```bash
python main.py --pdf paper.pdf --query "Explain this to an audience of computer science undergraduates"
python main.py --pdf paper1.pdf paper2.pdf --query "Compare these two papers and highlight where they agree and disagree"
python main.py --pdf paper.pdf --query "Give a 5-minute overview of the key findings"
```

The default query is `"Explain this paper to an audience of laypeople"`.

### Output

The finished presentation is saved as a `.pptx` file in `output/`, named after the first paper's title (or the session ID if no title is detected). The file is built from proto-slides stored in `data/wip.db`.

---

## Optional Arguments

| Argument | Default | Description |
|---|---|---|
| `--pdf PATH [PATH ...]` | `.samples/Transformers.pdf` | One or more PDF files to process |
| `--query TEXT` | `"Explain this paper to an audience of laypeople"` | Presentation query / audience |
| `--max-slides N` | `15` | Soft slide target (Planner adjusts based on content density) |
| `--processor` | `llama` | Document processor backend: `llama`, `docling`, `lighton` |
| `--text-splitter` | `none` | Chunking strategy: `none` or `semantic` (auto-defaults to `semantic` for LlamaParse) |
| `--object-store` | _(R2 with local fallback)_ | `local` or `r2` for image storage |
| `--llm-config PATH` | `src/llm/config.yaml` | LiteLLM Router config file |
| `-i`, `--interactive` | off | Pause after each document extraction for confirmation |
| `--no-logging` | _(logging on)_ | Disable Langfuse tracing |

---

## Document Processor

While multiple processor backends are implemented, only **LlamaParse** is available with the provided `requirements.txt` (other backends require additional dependencies and separate environments).

For LlamaParse, set **`LLAMA_CLOUD_API_KEY`** in your `.env`.

---

## Cloud Storage

Extracted images can be stored in [Cloudflare R2](https://developers.cloudflare.com/r2/) (default, with local fallback) or locally:

```env
CLOUDFLARE_ACCOUNT_ID=your_key_here
R2_ACCESS_KEY_ID=your_key_here
R2_SECRET_ACCESS_KEY=your_key_here
R2_BUCKET_NAME=multiagentsynthesis
```

Use `--object-store local` to skip R2 entirely.

---

## Graph Flow

```
START
  └─► Planner
        Loads all chunks from research.db
        Detects section boundaries
        Calls LLM → PresentationPlan (thesis, slide groups)
        Resolves section labels → chunk IDs
        │
        ▼
      Plan Executor  ◄─────────────────────────────────────┐
        Reads slide_groups from plan                        │
        Fans out via Send() — one agent per group     (retry loop)
        │                                                   │
        ├─► Slide Writer 1 ─► writes proto-slides → wip.db─┤
        ├─► Slide Writer 2 ─► writes proto-slides → wip.db─┤
        └─► Slide Writer N ─► writes proto-slides → wip.db─┘
              After all writers finish, Plan Executor checks
              slides_written counts. Any group with 0 slides
              is re-dispatched (up to 2 retries).
              When all groups pass → END
        │
        ▼
      PPTX Export (PandocBuilder reads wip.db → output/*.pptx)
```

### Planner

The Planner is a presentation architect, not a content writer. It:

1. Detects section boundaries in each paper using Markdown heading analysis
2. Builds a human-readable section outline with labels (`S0: Abstract`, `S3: Model Architecture`, ...)
3. Calls the LLM — which works with section labels, never raw chunk IDs — to produce a `PresentationPlan` containing a thesis, slide blueprints (with per-slide intents), and agent groupings (2-7 slides per group)
4. Validates the LLM output strictly; retries the full LLM call (up to 2 times) if any section label is invalid, any group is out of range, or any blueprint is empty
5. Resolves section labels → concrete chunk IDs in Python code before storing the plan in state

### Plan Executor

The Plan Executor is a pure dispatcher with a retry loop:

- **First call**: fans out one `slide_writer` per `SlideGroup` via LangGraph's `Send()` API
- **After writers complete**: checks `slides_written` counts from state; re-dispatches groups that produced 0 slides (up to 2 retries per group); exits to END when all groups pass

### Slide Writers

Each Slide Writer receives the blueprints and chunk text for its group. It synthesizes the proto-slides and writes them to `wip.db`. All errors are caught and reported without crashing the graph — the Plan Executor handles retry at the group level.

---

## Telemetry & Logging

- **Workflow tracing**: Langfuse traces the full LangGraph execution (agent transitions, state updates)
- **LLM tracing**: LiteLLM callback logs every completion call (latency, tokens, model used)

### Validation Error Dumps

When an LLM response fails Pydantic schema validation, the full error and offending JSON are written to `validation_errors/` (gitignored). The terminal shows a one-liner with the path. The folder is cleared at the start of every run.

```
[validation] 3 error dump(s) written to validation_errors/
```
