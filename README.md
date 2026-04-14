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

The system supports dynamic routing between model providers through `LiteLLM`. See `.env.sample` for API keys we support, and replace the placeholder values with your LLM Provider's API key.

```bash
copy .env.sample .env
```

### 4. Langfuse Logging Setup

To enable observability, ensure the following API keys are set in your `.env` file (you can get these from your Langfuse project settings):

```env
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_BASE_URL="https://cloud.langfuse.com"
```

The codebase uses `langfuse` which will automatically pick up these environment variables to trace agent runs.

### 5. LLM providers and routing

Runtime LLM comes from a **LiteLLM Router** built from YAML. The app reads **`src/llm/config.yaml`** at startup (see `init_from_config` in `src/llm/llm.py`).

#### Create your `config.yaml`

1. Copy the template: **`src/llm/config.sample.yaml`** or use the current **`src/llm/config.yaml`**
2. Fill in **API keys and base URLs** via `.env` using the `os.environ/VAR_NAME` placeholders from the sample (LiteLLM resolves those strings when the Router starts). You can also add whatever API providers you have.
3. Adjust **`router.providers`** and **`router.fallback_providers`** to match the models and backends you actually use.

You can keep multiple experimental YAML files elsewhere and point the app at one for a single run:

```bash
python main.py --llm-config path/to/your/config.yaml
```

#### Provider entries (LiteLLM format)

You are not limited to the providers in the sample: **any provider and model string LiteLLM supports** can be added by following the same config structure in `config.yaml`.

- The config file follows a provider-first approach. Under **`router.providers`**, each key (e.g. `gemini`, `openrouter`) is a **named block**, and API Keys and API base URLs **shared** for every row in that block.
- Each item under **`models`** is one **deployment**: a `model:` string in LiteLLM form `<provider>/<model-string>` (e.g. `gemini/gemini-2.5-pro`, `openrouter/...`) plus any extra per-model fields merged into `litellm_params`.
- **`router.fallback_providers`** uses the same structure; its default group alias is **`fallback_model_name`** instead of **`default_model_name`** (see below).

For exact parameter names and provider-specific options, use the [LiteLLM provider docs](https://docs.litellm.ai/docs/providers). By default LiteLLM follows [OpenAI API](https://developers.openai.com/api/reference/resources/responses).

#### Router groups: same name vs different names vs `fallbacks`

LiteLLM’s Router groups **deployments** by a string **`model_name`** (this repo and the sample YAML call that a **group alias** — one logical “model group” you can target in code).

| Concept                          | Meaning                                                                                                                                                                                                                                                                                                                                                                                            |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **One group (one `model_name`)** | Every `models` row that ends up with the **same** alias shares one pool. The Router **routes inside that pool** (e.g. load balancing, retries, cooldowns, rate-limit handling) without you configuring anything extra. Rows inherit the block default (`default_model_name` for `providers`, `fallback_model_name` for `fallback_providers`) unless you set **`model_name: <alias>`** on that row. |
| **Different groups**             | Different aliases (e.g. `app`, `writer`, `fast`, `fallback`) are **separate** pools. The app chooses which group to call via the Router **`model=`** argument (and `LLMConfig.model` / agent defaults). Nothing automatically jumps from `app` to `writer` unless **you** request that alias or configure cross-group fallbacks.                                                                   |
| **`router.fallbacks`**           | Optional **cross-group** order: if primary group calls fail, the Router can try the listed backup **aliases** (e.g. map `app` → try `fallback`). If you omit `fallbacks`, other groups still exist in `model_list`, but there is **no** automatic chain between aliases—you pick the alias explicitly.                                                                                             |

So: **routing within a group** = multiple deployments behind one alias; **switching groups** = explicit `model=` or an explicit **`fallbacks`** map in YAML.

## Run

```bash
python main.py
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

Adding `--slides` will generate powerpoint slides instead of a single document. The number of slides is controlled by the `--max-slides` argument, which defaults to 12.

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

### Validation Error Dumps

When an LLM response fails Pydantic schema validation, the full error detail and offending JSON are written to a structured dump file under `validation_errors/` rather than flooding the terminal. The terminal log shows only a succinct one-liner with a path to the relevant file.

The `validation_errors/` folder is cleared automatically at the start of every `main.py` run, so it always reflects the most recent execution. The folder is gitignored and will not appear in version control. If any validation errors occurred during a run, a summary line is printed at the end of the run:

```
[validation] 3 error dump(s) written to validation_errors/
```

Each dump file is a JSON object containing the agent name, attempt number, timestamp, error summary, and the raw offending JSON for inspection.
