# Implementation Plan: Contextual Retrieval for Document Ingestion Pipeline

## Overview

Enhance the document ingestion pipeline to contextualize text chunks and multimodal artifacts (images, tables, equations) using LLM-based contextualization, improving retrieval quality for semantic search.

## Goals

1. **Enhance Retrieval Quality**: Provide contextual information for each artifact to improve semantic search accuracy
2. **Support Multimodal Artifacts**: Properly handle images via Cloudflare R2 integration with multimodal LLM
3. **Maintain Backward Compatibility**: Use existing `contextualized_text` fields in schema without modification
4. **Optimize Costs**: Leverage cost-effective models and LiteLLM's built-in optimization features

## Current State Analysis

### Architecture

| Component          | File                                       | Role                                                              |
| ------------------ | ------------------------------------------ | ----------------------------------------------------------------- |
| Document Processor | `src/processing/document/processor.py`     | Orchestrates extraction → contextualization → embedding → storage |
| Contextualizer     | `src/processing/context/contextualizer.py` | Populates `contextualized_text` fields using LiteLLM              |
| Schema             | `src/processing/document/schema.py`        | Defines `ExtractionResult` and artifact types                     |
| LLM Abstraction    | `src/llm/llm.py`                           | LiteLLM Router with configurable model deployments                |
| Object Store       | `src/memory/objectstore/r2_store.py`       | Cloudflare R2 for image storage                                   |
| Database           | `src/memory/research/database.py`          | SQLite with sqlite-vec for persistence and caching                |

### Current Flow

```
process_document()
├── Compute content_hash
├── Check cache: db.document_exists(content_hash)?
│   ├── YES → db.load_document_by_hash() → return (SKIP contextualization)
│   └── NO  → backend.extract() → contextualize → embed → store
```

### Current LLM Config (`src/llm/config.yaml`)

**Primary Models (alias: "app")**:
- gemini-2.5-pro
- gemini-2.5-flash
- gemini-2.5-flash-lite

**Fallback Models (alias: "fallback") - FREE options**:
- openrouter/openai/gpt-oss-120b:free
- openrouter/openai/gpt-oss-20b:free
- ollama/qwen3.5:397b-cloud
- ollama/gemma3:27b-cloud

**Built-in Settings**:
- `num_retries: 2`
- `retry_after: 2`
- `cooldown_time: 30`
- `timeout: 120`
- Fallback chain: `app → ["fallback"]`

### Gaps Identified

1. **Image contextualization**: Only uses caption text, not actual image data
2. **No multimodal support**: Cannot pass visual context to LLM
3. **No R2 upload utility**: Base64 images cannot be referenced by URL
4. **Cache bypass for contextualization**: If document exists in cache but was processed WITHOUT contextualization (e.g., `contextualizer=None`), re-running will skip contextualization entirely
5. **Minimal error handling**: No validation of `contextualized_text` quality

## Implementation Design

### 1. Enhanced Contextualizer

**File**: `src/processing/context/contextualizer.py`

#### A. Multimodal Artifact Handling

```python
def contextualize(self, result: ExtractionResult) -> ExtractionResult:
    """
    Contextualize all artifacts (chunks, images, tables, equations).
    
    For images:
    - If storage_path exists (R2 URL): use directly for multimodal LLM
    - If only base64_data: upload to R2 first, then use URL
    
    For tables/equations: use text content with surrounding context
    """
```

#### B. LiteLLM Optimization Strategies

1. **Use Built-in Retry/Fallback**:
   - LiteLLM Router already configured with `num_retries: 2`, `cooldown_time: 30`
   - Fallback chain `app → ["fallback"]` handles primary model failures
   - No need to write custom retry logic unless specific requirements

2. **Context Caching (Anthropic/Gemini)**:
   ```python
   messages = [
       {
           "role": "system", 
           "content": [
               {"type": "text", "text": document_markdown, "cache_control": {"type": "ephemeral"}}
           ]
       },
       {"role": "user", "content": chunk_prompt}
   ]
   ```
   - Cache the full document markdown once per document
   - Reduces token costs significantly for multi-chunk contextualization

3. **Batching (Future)**:
   - Process multiple chunks in single LLM call for cost efficiency
   - Requires structured JSON output for parsing multiple contexts

#### C. Model Selection

| Use Case | Model | Reason |
|----------|-------|--------|
| Text chunks | `gemini-2.0-flash` or `gemini-2.5-flash-lite` | Cost-effective, fast |
| Images (multimodal) | `gemini-2.0-flash` | Multimodal support, low cost |
| Fallback (FREE) | `openrouter/openai/gpt-oss-120b:free` | No cost for dev/testing |
| Local testing | `ollama/gemma3:27b-cloud` | No API key needed |

#### D. Error Handling (LiteLLM-native)

```python
# Leverage LiteLLM's built-in error handling
# Router already handles: retries, cooldowns, fallbacks
# Only add application-level validation:

def _validate_contextualized_text(self, text: str | None) -> str:
    """Validate contextualization output quality."""
    if not text or len(text.strip()) < 10:
        return ""  # Too short/empty → will use fallback
    if any(err in text.lower() for err in ["error", "failed", "unable"]):
        return ""  # Error message leaked → use fallback
    return text.strip()
```

### 2. Image Upload Utility

**New File**: `src/processing/context/image_uploader.py`

```python
class ImageUploader:
    """Upload base64 images to R2 and return public URLs."""
    
    def __init__(self, object_store: R2ObjectStore | LocalObjectStore):
        self.object_store = object_store
        self._upload_cache: dict[str, str] = {}  # base64_hash -> URL
    
    def get_or_upload_url(self, doc_id: str, image_id: str, base64_data: str) -> str:
        """
        Get R2 URL for image, uploading if necessary.
        
        Returns:
            Public URL for multimodal LLM input
        """
        # Check cache first
        # If not cached, upload to R2
        # Return URL
```

**Features**:
- In-memory cache for current session (`base64_hash → URL`)
- Retry logic: delegate to R2ObjectStore's built-in retry (already has `@retry` decorator)
- Extension detection from base64 header or MIME type

### 3. Prompt Enhancements

**File**: `src/processing/context/prompts.py`

#### CHUNK_CONTEXT_PROMPT (Enhanced)

```python
CHUNK_CONTEXT_PROMPT = """
<document>
{document_markdown}
</document>

Here is the chunk we want to situate within the whole document.
<chunk>
{chunk_text}
</chunk>

Provide a succinct context in 1 paragraph, no more than 5 sentences, describing this chunk's specific position and role within the document for the purposes of improving search retrieval.

Example:
Original chunk: "The company's revenue grew by 3% over the previous quarter."
Contextualized: "This chunk is from an SEC filing on ACME Corp's Q2 2023 performance. The previous quarter's revenue was $314 million. The company's revenue grew by 3% over the previous quarter."

Answer only with the succinct context and nothing else.
"""
```

#### IMAGE_CONTEXT_PROMPT (New)

```python
IMAGE_CONTEXT_PROMPT = """
<document_context>
{document_summary}
</document_context>

Surrounding text:
<before>{text_before}</before>
<after>{text_after}</after>

[IMAGE]: The image shows: {caption}

Provide a succinct context in 1 paragraph, no more than 5 sentences describing this image's role and specific position within the document.
Focus on how the surrounding narrative frames this image and what information it conveys.

Answer only with the succinct context and nothing else.
"""
```

### 4. Document Processor Integration

**File**: `src/processing/document/processor.py`

#### A. Cache Bypass Logic for Contextualization

**Problem**: If document was processed WITHOUT contextualization (e.g., older runs, `contextualizer=None`), cache returns document with empty `contextualized_text` fields.

**Solution**: Check if contextualization is requested AND if artifacts are missing context:

```python
def process_document(self, source_path: str) -> ExtractionResult | None:
    content_hash = compute_hash(source_path)
    
    if self._db and self._db.document_exists(content_hash):
        cached = self._db.load_document_by_hash(content_hash)
        if cached:
            # Check if contextualization is needed
            if self._contextualizer and self._needs_contextualization(cached):
                cached = self._contextualizer.contextualize(cached)
                # Re-embed with updated contexts
                if self._embedder:
                    cached = self._reembed(cached)
                # Update stored document
                self._db.save_document(cached)
            return cached
    
    # ... rest of processing

def _needs_contextualization(self, result: ExtractionResult) -> bool:
    """Check if any artifact lacks contextualized_text."""
    for chunk in result.source_chunks:
        if not chunk.contextualized_text:
            return True
    for img in result.images:
        if not img.contextualized_text:
            return True
    for tbl in result.tables:
        if not tbl.contextualized_text:
            return True
    for eq in result.equations:
        if not eq.contextualized_text:
            return True
    return False
```

#### B. Logging

Use existing `AgentLogger` instance initialized in `DocProcessor.__init__`:

```python
self._logger.log("[DocProcessor] Contextualizing cached document...", level="info")
```

## Technical Details

### A. Contextualization Flow (Updated)

```
process_document()
├── Compute content_hash
├── Check cache: document_exists?
│   ├── YES → load_document_by_hash()
│   │         ├── needs_contextualization() AND contextualizer exists?
│   │         │   ├── YES → contextualize() → reembed() → save_document()
│   │         │   └── NO  → return cached
│   │         └── return
│   └── NO  → extract() → contextualize() → embed() → save_document()
```

### B. LLM Call Optimization

| Strategy | Implementation | Benefit |
|----------|----------------|---------|
| Context caching | `cache_control: ephemeral` on system message | Cache document markdown across chunks |
| Router fallback | Use LiteLLM config `fallbacks: [{app: ["fallback"]}]` | Auto-failover to FREE models |
| Built-in retry | Router `num_retries: 2`, `retry_after: 2` | No custom retry code needed |
| Timeout | Router `timeout: 120` | Prevent hanging |

### C. Error Handling Strategy

| Failure Mode | Detection | Recovery |
|--------------|-----------|----------|
| LLM call fails | LiteLLM Router handles retries/fallbacks | Returns error → log and use empty context |
| R2 upload fails | R2ObjectStore has `@retry` decorator | Fallback to base64 rendering in multimodal prompt |
| Invalid context | `_validate_contextualized_text()` | Return empty string, skip embedding for that artifact |
| Missing storage_path | Check in `contextualize()` | Upload base64 to R2 first |

## Implementation Phases

### Phase 1: Core Contextualizer Enhancement

- [ ] Update `contextualizer.py` for multimodal support
- [ ] Create `image_uploader.py` utility
- [ ] Enhance `prompts.py` with new prompts
- [ ] Add `_validate_contextualized_text()` validation
- [ ] Implement context caching for document markdown

### Phase 2: Document Processor Integration

- [ ] Add `_needs_contextualization()` check
- [ ] Add cache bypass logic for re-contextualization
- [ ] Add `_reembed()` for updating embeddings after re-contextualization
- [ ] Add logging with existing `AgentLogger`

### Phase 3: Evaluation Methodology

- [ ] Create evaluation dataset (5-10 documents with ground truth)
- [ ] Implement retrieval quality metrics (precision@k, recall@k, MRR)
- [ ] A/B test: baseline (no context) vs. contextualized retrieval
- [ ] Measure cost per document (tokens, API calls)

## Evaluation Methodology

### A. Dataset Preparation

1. **Documents**: 5-10 representative PDFs (scientific papers, reports)
2. **Ground Truth**: Manually annotate relevant chunks for 10-20 queries per document
3. **Metrics**:
   - **Precision@k**: Fraction of retrieved chunks that are relevant
   - **Recall@k**: Fraction of relevant chunks that are retrieved
   - **MRR (Mean Reciprocal Rank)**: Average rank of first relevant result

### B. Evaluation Pipeline

```python
class RetrievalEvaluator:
    """Evaluate retrieval quality for contextualized vs. baseline."""
    
    def __init__(self, db: ResearchDatabase):
        self.db = db
    
    def evaluate(self, queries: list[Query], k: int = 5) -> EvalResults:
        """
        Run evaluation:
        1. For each query, retrieve top-k chunks
        2. Compare with ground truth
        3. Compute precision, recall, MRR
        """
    
    def compare(self, baseline: EvalResults, contextualized: EvalResults) -> ComparisonReport:
        """Compare baseline vs. contextualized retrieval quality."""
```

### C. Success Criteria

| Metric | Baseline Target | Contextualized Target |
|--------|-----------------|----------------------|
| Precision@5 | ≥ 0.60 | ≥ 0.75 (+25%) |
| Recall@5 | ≥ 0.50 | ≥ 0.65 (+30%) |
| MRR | ≥ 0.70 | ≥ 0.85 (+21%) |
| Cost/doc | - | ≤ $0.05 (gemini-2.0-flash) |

## Risk Mitigation

### Potential Issues and Solutions

| Risk | Impact | Solution |
|------|--------|----------|
| Model doesn't support multimodal | High | Use gemini-2.0-flash (verified multimodal support) |
| R2 upload failures | Medium | R2ObjectStore has `@retry` decorator built-in |
| Performance bottleneck | Medium | Context caching reduces duplicate token processing |
| Corrupted contextualized text | Low | `_validate_contextualized_text()` filters bad outputs |
| Cache hit with missing context | High | `_needs_contextualization()` check forces re-contextualization |

## File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/processing/context/contextualizer.py` | Modify | Multimodal support, context caching, validation |
| `src/processing/context/image_uploader.py` | Create | Base64 to R2 upload utility |
| `src/processing/context/prompts.py` | Modify | Enhanced prompts with examples |
| `src/processing/document/processor.py` | Modify | Cache bypass for re-contextualization |
| `src/evaluation/retrieval_evaluator.py` | Create | Evaluation pipeline for retrieval quality |

## Questions Resolved

1. **Q: Should we add another field in result schema?**
   - **A: No.** Use existing `contextualized_text` fields on all artifact types.

2. **Q: Why synchronous processing?**
   - **A:** Current architecture is synchronous by design. Keeps implementation simple. Can add background processing later if needed.

3. **Q: How to handle images?**
   - **A:** Check `storage_path` for R2 URL. If missing, upload base64 to R2. Pass URL to multimodal LLM.

4. **Q: How to handle cache hit with missing contextualization?**
   - **A:** Add `_needs_contextualization()` check. If artifacts lack context, run contextualization and update database.