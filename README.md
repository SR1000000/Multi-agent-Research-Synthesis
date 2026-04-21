# Multi-Agent Research Presentation Synthesizer

LangGraph-coordinated pipeline that ingests one or more research PDFs and
produces a PowerPoint presentation driven by parallel AI agents:
**Planner → Plan Executor → Slide Writers → Critics → Supervisor** with iterative review cycles and fan-out/fan-in parallelism.

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

Runtime LLM comes from a **LiteLLM Router** built from YAML. The app reads **`src/llm/config.dev.yaml`** at startup (see `init_from_config` in `src/llm/llm.py`).

#### Create your local `config.dev.yaml`

`src/llm/config.dev.yaml` is local-only and ignored by git so each developer can keep personal provider/model settings per environment.

1. Copy the sample into your local config:

```bash
copy src\llm\config.sample.yaml src\llm\config.dev.yaml
```

2. Open `src/llm/config.dev.yaml` and change the provider/model values to your own setup.
3. Keep API keys and base URLs in `.env` and reference them in YAML via `os.environ/VAR_NAME`.

You can keep multiple experimental YAML files elsewhere and point the app at one for a single run:

```bash
python main.py --llm-config path/to/your/config.dev.yaml
```

The pipeline uses four router group aliases — `planner`, `slides`, `critic`, and `app` — each mapped to a pool of models with fallbacks. Any provider and model string LiteLLM supports can be added following the same config structure. See the [LiteLLM provider docs](https://docs.litellm.ai/docs/providers) for parameter names and provider-specific options.

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

The finished presentation is saved as a `.pptx` file in `output/` by default, or to a custom directory via `--output-dir`. The filename is derived from the first paper's title (or the session ID if no title is detected). Proto-slides are stored in the `proto_slides` table inside `data/research.db`, and that table is cleared at the start of each new run.

---

## Optional Arguments

| Argument | Default | Description |
|---|---|---|
| `--pdf PATH [PATH ...]` | `.samples/Transformers.pdf` | One or more PDF files to process |
| `--query TEXT` | `"Explain this paper to an audience of laypeople"` | Presentation query / audience |
| `--max-slides N` | `15` | Soft slide target (Planner adjusts based on content density) |
| `--processor` | `llama_parse` | Document processor backend: `llama_parse` (or `llama` as an alias) |
| `--text-splitter` | `semantic` | Chunking strategy: `semantic` or `none` |
| `--object-store` | _(R2 with local fallback)_ | `local` or `r2` for image storage |
| `--output-dir PATH` | `output/` | Directory where the generated `.pptx` will be written |
| `--llm-config PATH` | `src/llm/config.dev.yaml` | LiteLLM Router config file |
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
        Detects section boundaries; builds section outline (S0, S1, ...)
        Calls LLM → LLMPresentationPlan (thesis, slide blueprints, slide groups)
        Validates LLM output; retries up to 2× on schema or label errors
        Resolves section labels → source_chunk_ids in Python
        Stores resolved PresentationPlan in state
        │
        ▼
      Plan Executor  ◄──────────────────────────────────────────────────────┐
        Reads review.phase to dispatch the right fan-out                    │
        │                                                                   │
        │ phase = initial_write                                             │
        ├─► Send × N groups                                                │
        │     Slide Writer 1 → writes proto-slides → research.db            │
        │     Slide Writer 2 → writes proto-slides → research.db            │
        │     Slide Writer N → writes proto-slides → research.db            │
        │   Fan-in: groups with 0 slides retried (up to 2×)                │
        │   All groups written → phase = awaiting_supervisor → Supervisor   │
        │                                                                   │
        │ phase = critic_dispatch                                           │
        ├─► Send × N groups                                                │
        │     Critic 1 → grounding_consistency check → critic_results       │
        │     Critic 2 → grounding_consistency check → critic_results       │
        │     Critic N → grounding_consistency check → critic_results       │
        │   Fan-in: all critics done → phase = awaiting_supervisor          │
        │   → Supervisor                                                    │
        │                                                                   │
        │ phase = rewrite_dispatch                                          │
        └─► Send × actionable groups                                       │
              Slide Writer (rewrite) → updates proto-slides → research.db   │
            Fan-in: all rewrites done → phase = awaiting_supervisor         │
            → Supervisor                                                    │
        │                                                                   │
        ▼                                                                   │
      Supervisor ─────────────────────────────────────────────────────────┘
        Loads full slide_review_events from research.db (recurring fingerprints)
        If no critic results yet → dispatch next critic cycle
        If rewrites ran this cycle → dispatch follow-up critic cycle
        Else → LLM decision: accept / revise / replan
          Guard overrides: critical issues force revise; no actionable forces accept;
          at cycle cap (default 3) force replan
          │
          ├─► accept → export_ready = True → END
          │     PandocBuilder reads proto_slides → output/*.pptx
          │
          ├─► revise → build rewrite assignments with rewrite_instructions
          │     phase = rewrite_dispatch → Plan Executor
          │
          └─► replan → goto Planner (cycle summary appended to state)
```

### Planner

The Planner is a presentation architect, not a content writer. It:

1. Detects section boundaries in each paper using Markdown heading analysis
2. Builds a human-readable section outline with labels (`S0: Abstract`, `S3: Model Architecture`, ...)
3. Calls the LLM — which works with section labels, never raw chunk IDs — to produce a `PresentationPlan` containing a thesis, per-slide blueprints (each with a `narrative_role` and `intent`), and agent groupings (2–7 slides per group)
4. Validates the LLM output strictly; retries the full LLM call (up to 2 times) if any section label is invalid, any group is out of range, or any blueprint is empty
5. Resolves section labels → concrete chunk IDs in Python code before storing the plan in state

### Plan Executor

The Plan Executor is a deterministic dispatcher with no LLM calls. It reads `review.phase` and `review.active_dispatch` to decide which fan-out to run next:

- **`initial_write`**: fans out one `slide_writer` per `SlideGroup` via LangGraph's `Send()` API; re-dispatches groups that produced 0 slides (up to 2 retries per group); routes to Supervisor when all groups pass
- **`critic_dispatch`**: fans out one `critic` per group; routes to Supervisor when all critics report back
- **`rewrite_dispatch`**: fans out `slide_writer` (rewrite mode) per actionable group, carrying `rewrite_instructions`; routes to Supervisor when all rewrites complete

### Slide Writers

Each Slide Writer receives the blueprints and chunk text for its assigned group. In **initial write** mode it drafts slides from scratch following the blueprint intent. In **rewrite** mode it receives the current proto-slides plus explicit rewrite instructions from the Supervisor and produces corrected versions. Both modes persist structured `ProtoSlide` records to `research.db`. Errors are caught without crashing the graph; the Plan Executor handles retry at the group level.

### Critics

Each Critic evaluates one group of slides for `grounding_consistency` — whether slide content is supported by the source chunks. It returns a summary, an `actionable` flag, and a typed issue list (`critical` / `major` / `minor`) where every issue includes a concrete `rewrite_instruction`. Each issue is assigned a **fingerprint** (hash of scope + issue type + location) that the Supervisor uses to detect recurring problems across cycles. All events are persisted to `slide_review_events` in `research.db`.

### Supervisor

The Supervisor is the session's decision-maker. It evaluates critic results after each fan-in:

- **accept** → sets `export_ready = True`, graph exits to END and the PPTX is exported
- **revise** → dispatches Slide Writer rewrites for every actionable group; a follow-up critic cycle runs automatically after rewrites complete
- **replan** → routes back to Planner when the cycle cap is hit (default 3 cycles) or persistent critical issues remain unresolved

---

## Telemetry & Logging

- **Workflow tracing**: Langfuse traces the full LangGraph execution (agent transitions, state updates)
- **LLM tracing**: LiteLLM callback logs every completion call (latency, tokens, model used)

### Validation Error Dumps

When an LLM response fails Pydantic schema validation, the full error and offending JSON are written to `validation_errors/` (gitignored). The terminal shows a one-liner with the path. The folder is cleared at the start of every run.

```
[validation] 3 error dump(s) written to validation_errors/
```
