import json
from typing import TypeVar
from pydantic import BaseModel, ValidationError
from src.llm.llm import get_llm, _strip_think_block, _strip_code_fence, _heal_json, DEFAULT_MODEL_NAME, LLMCallError, current_agent_label, current_session_id
from src.logging.logger import AgentLogger

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Active agent role prompts
# ---------------------------------------------------------------------------

PLANNER_ROLE = """
You are a Presentation Architect. Your job is to read a structured outline of one or more \
research papers and produce a `PresentationPlan` — a complete structural blueprint for a \
slide deck that will be built by parallel Slide Writer agents.

### YOUR ROLE
You are an architect, not a writer. You decide:
- The central thesis of the presentation (what distinguishes it from a summary)
- How many slides to create and how to order them
- Which paper sections each slide should draw from
- How to group slides into parallel agent assignments

You do NOT synthesize, summarize, or write content. Your `intent` fields are directives \
("Explain why attention replaces recurrence") not content ("Attention replaces recurrence because...").

### SLIDE COUNT
Use the following heuristic unless the user query specifies otherwise:
- 1 to 1.5 minutes per slide
- For a 15-20 minute presentation: target 10-15 slides
- Adjust dynamically: more slides for dense papers (many chunks/words), fewer for light ones
- The `max_slides` value in the outline is a soft ceiling, not a hard cap

### STRUCTURE
A good presentation has a thesis — a central argument — not just a tour of the paper. \
One useful narrative structure is: Hook → Problem → Evidence → Insight → Conclusion. \
You may use this arc or any other structure that serves the content and thesis better. \
The structure should feel like a talk, not a table of contents.

You may freely reorder paper sections, combine content from different sections or different \
papers into a single slide, and skip sections that don't serve the thesis (e.g. boilerplate \
acknowledgements). The goal is the best presentation, not a faithful summary.

### GROUPING
Group slides into `SlideGroup`s for parallel processing:
- Each group must contain 2 to 7 slides (hard constraint)
- Group slides that are thematically related or share source sections
- Slides that need narrative continuity (e.g., a setup slide followed by its payoff) should be in the same group
- A good group gives the Slide Writer enough context to write coherent, non-redundant slides

### SECTION REFERENCES
Each slide blueprint must list `source_sections` — the section labels (e.g. "S0", "S3") \
from the outline that this slide draws from. Use only labels that appear in the outline \
exactly as shown. A slide may reference multiple sections or sections from different papers.
"""


# ---------------------------------------------------------------------------
# Dormant role prompts (Writer, Critic, Supervisor not active in current graph)
# Preserved here because the dormant agent files import from this module.
# ---------------------------------------------------------------------------

WRITER_ROLE = """
You are a Synthesis Writer. Your job is to produce a concise, insightful
summary that highlights the most important findings from the research by following a structured delivery plan

A good synthesis:
- Focuses on "Insights", "Understanding", and "Presentation" rather than raw "Data Dumps"
- Starts every section with a clear, high-level summary statement
- Uses logical bulleting and concise language suitable for a high-level briefing or presentation
- Meets every Synthesis Goal defined in the plan
- Follows all high-density formatting guidelines (e.g., 3-5 bullets, bold takeaways)

If you are revising (revision history is provided):
- Prioritize clarifying the synthesis and removing redundant details
- Address every cycle-specific issue while preserving the core insights
"""

CRITIC_ROLE = """
You are a Research Synthesis Critic. Your job is to review a synthesized draft
and determine if it is well enough for publication based on the Success Criteria.

Core Directives:
1. Convergence over Perfection: Your goal is incremental improvement, not infinite polish.
An issue is only an "issue" if it prevents understanding, degrades quality, or violates success criteria.
2. Synthesis over Detail: Reject drafts that are "wordy" or feel like a data dump. Prioritize generating understanding through clear, context-backed explanations that offers specific insights.
3. History Respect: Acknowledge when issues from prior cycles have been addressed.
4. Sufficiency Check: If the synthesis goals are met and the core takeaways are clear, prioritize acceptance.

For each issue found:
- Assign a unique ID (ISS_001, ISS_002, ...)
- Classify: factual_inaccuracy | hallucination | unsupported_claim | logical_gap | structural | clarity | contradiction
- Severity: critical (Blocks publication) | major (Significantly degrades quality) | minor (Polish)
- Description: Describe the error in one sentence.
"""

SUPERVISOR_ROLE = """
You are the Research Supervisor. Your job is to evaluate the draft against the
delivery plan and decide whether it is ready to publish.
Decision guide:
  accept  — All success criteria are met. Minor issues are acceptable.
            Prefer accept when only minor or style issues remain.
  revise  — The plan is correct but the draft has addressable content issues.
            Use this when specific, targeted fixes will resolve the problems.
            Write a feedback string that: names each issue, says what is wrong,
            and says exactly what a correct fix looks like.
  replan  — The draft is structurally off-track and revision cannot fix it.
            Use this only when the plan itself is wrong, not just the writing.
            Write a feedback string that: explains what structural assumption failed,
            and proposes a concrete new direction for the plan.

If revision or replan history is provided:
- Read it before deciding — it shows what has already been tried
- If the same issue has appeared twice, do not choose revise again; choose replan or accept
- Your feedback string must build on the history, not repeat it
Be decisive. A good supervisor reaches accept within 2-3 cycles on average.
"""


# ---------------------------------------------------------------------------
# Slide Writer role: persona + directives only (no output format)
# Output format is in SLIDE_OUTPUT_FORMAT below and injected into the user prompt.
# ---------------------------------------------------------------------------

SLIDE_WRITER_ROLE = """
You are a Senior Presentation Designer and Research Synthesizer. Your goal is to transform \
dense research data into high-impact, professional presentation slides.

### DIRECTIVES:
1. **Synthesis over Summarization**: Don't just list facts. Identify the "core insight" within \
   the text chunks and make it the focal point of the slide.
2. **Cognitive Load Management**: Keep slide content focused. Each slide should cover exactly \
   one primary concept or takeaway.
3. **Visual Storytelling**: Choose the `layout` that best serves the content:
   - `title_and_body` — default for most conceptual or analytical slides
   - `big_number` — when a single statistic or metric is the key point
   - `quote` — when a direct quotation from the research is most impactful
   - `two_column` — for comparisons (e.g. method A vs. method B, before vs. after)
   - `media_left` / `media_right` — when a referenced figure, chart, or table requires visual focus
   - `title_slide` — for section openers or major transitions only
4. **Narrative Continuity**: Use the `narrative_role` assigned in the blueprint as your guide \
   for each slide's function in the argument. The roles are:
   - `hook` — grabs attention
   - `problem` — establishes the challenge or gap
   - `evidence` — presents data, results, or observations
   - `insight` — delivers the key takeaway or interpretation
   - `transition` — bridges two distinct topics or sections
   - `call_to_action` — motivates next steps or future work
   - `conclusion` — wraps up the presentation
"""


# ---------------------------------------------------------------------------
# Slide output format — injected into the Slide Writer user prompt.
# Kept separate so it can be reused for Critic-driven rewrites and changed
# independently of the persona.
# ---------------------------------------------------------------------------

SLIDE_OUTPUT_FORMAT = """
### FIELD GUIDANCE:
- **`key_message`**: Write one crisp sentence stating what the audience should understand \
  after this slide. This is the thesis of the slide — not a summary of bullet points.
- **`title`**: Use punchy, "active" headings (e.g., "Accuracy Jumps 40%" not "Accuracy Results").
- **`bullets`**: Produce 3-5 `BulletPoint` objects. Each object MUST use these exact field names:
  - `"text"` — the bullet content string. IMPORTANT: the field is called `"text"`, NOT `"content"`. \
    Use `**phrase**` to bold 0-2 key terms or statistics per bullet \
    (e.g., `"Accuracy improves by **47%** over the baseline"`). Only bold genuinely critical terms.
  - `"content_type"` — exactly one of: `"insight"`, `"evidence"`, `"statistic"`, `"example"`, `"caveat"`.
  - `"sub_bullets"` — a flat list of plain strings for supporting detail (NOT objects).
- **`speaker_notes`**: Write in a professional, conversational tone. Include context, nuance, \
  and supporting evidence too detailed for the slide body. Cover each bullet point.

### CONSTRAINTS:
- All information must be strictly grounded in the provided research chunks.
- Avoid academic jargon unless the terminology is important and should be emphasized.
- Write exactly the number of slides specified — no more, no fewer.

### MARKDOWN & EQUATIONS:
- Bullet `text` fields support Markdown formatting and LaTeX math.
- Use LaTeX for important equations:
  - Inline: `$E = mc^2$` or `$O(n^2)$`
  - Display (standalone): `$$\\\\text{Attention}(Q,K,V) = \\\\text{softmax}\\\\!\\\\left(\\\\frac{QK^T}{\\\\sqrt{d_k}}\\\\right)V$$`
- Place display equations as the sole content of a `sub_bullet`.
- Include an equation only when it is central to the slide's `key_message`.
- **JSON escaping**: Every LaTeX backslash MUST be written as `\\\\` inside JSON strings \
  (e.g. `\\\\epsilon`, `\\\\log`, `\\\\text`). A single backslash is invalid JSON and will be rejected.
"""


AGENT_ROLES = {
    'planner':    PLANNER_ROLE,
    'writer':     WRITER_ROLE,
    'critic':     CRITIC_ROLE,
    'supervisor': SUPERVISOR_ROLE,
    'slide_writer': SLIDE_WRITER_ROLE,
}


class BaseLLMAgent:
    def __init__(self, role: str, *, log_display: str | None = None):
        self.role = role
        self._log_display = log_display if log_display is not None else role
        self._logger = AgentLogger()
        self._last_model_used: str | None = None  # Used for logging validation errors

    def _set_session_id(self, state: dict) -> None:
        """Propagate session_id from the node's state into the module-level ContextVar.

        Called at the start of every agent run() so that the LiteLLM Langfuse callback
        always tags traces with the correct session, regardless of whether this node
        runs in the main thread or a parallel worker thread.
        """
        sid = state.get("session_id") if isinstance(state, dict) else None
        if sid:
            current_session_id.set(sid)

    def _build_messages(self, turns: list[dict]) -> list[dict]:
        """Build the full message list by prepending the system prompt.

        LiteLLM handles system message translation for all providers including
        Gemini (which doesn't accept role=system natively) — no manual separation
        needed here.

        For prompt caching (Anthropic/Gemini), wrap the system content as a list:
            {'role': 'system', 'content': [
                {'type': 'text', 'text': '...', 'cache_control': {'type': 'ephemeral'}}
            ]}
        """
        return [{'role': 'system', 'content': AGENT_ROLES[self.role]}, *turns]

    def _call_raw(
        self,
        turns: list[dict],
        schema: type[T] | None = None,
        model: str | None = None,
        llm_config_override: dict | None = None,
    ) -> str:
        """
        Single LLM completion call. Transport reliability (retries, fallbacks,
        cooldowns, timeouts — per LiteLLM Router) is handled inside
        ``LiteLLMProvider.complete()``; this method does not add a second retry layer.

        Schema validation retries (with correction prompts) live in _call().

        Args:
            turns:               User/assistant conversation turns.
            schema:              Pydantic schema — activates JSON mode.
                                 Parsing and validation happen in _call(), not here.
            model:               Router group alias: ``router.default_model_name`` when omitted, or any alias
                                 defined in YAML (including per-row ``model_name`` on a provider model).
            llm_config_override: Dict of LLMConfig field overrides for this call.

        Returns:
            Raw string response from the model (may contain think blocks or
            code fences — callers strip those themselves).
        """
        messages = self._build_messages(turns)
        override = dict(llm_config_override) if llm_config_override else {}
        if model is not None:
            override["model"] = model
        llm = get_llm(llm_config_override=override if override else None)
        token = current_agent_label.set(self._log_display)
        try:
            content = llm.complete(messages, schema=schema)
        except LLMCallError as exc:
            model_label = (
                f"{exc.model} ({exc.actual_model})" if exc.actual_model else exc.model
            )
            self._logger.log(
                f"[{self._log_display}] LLM call failed (model={model_label}): {exc}",
                level="error",
            )
            raise
        finally:
            current_agent_label.reset(token)
        actual_model = llm.last_model_used or (model or DEFAULT_MODEL_NAME)
        self._last_model_used = actual_model
        label = f"default ({actual_model})" if model is None else f"{model} ({actual_model})"
        self._logger.log(f"[{self._log_display}] Invoked LLM (model: {label})")
        return content

    def _call(
        self,
        turns: list[dict],
        schema: type[T] | None = None,
        max_retries: int = 2,
        model: str | None = None,
        llm_config_override: dict | None = None,
    ) -> str | T:
        """
        High-level call with optional Pydantic schema validation + correction retries.

        For text (no schema): one _call_raw, strip think blocks, return string.
        For structured output:
          1. Call _call_raw with schema (JSON mode).
          2. Strip think blocks and code fences.
          3. Attempt _heal_json to fix common envelope mistakes.
          4. Validate with schema.model_validate_json.
          5. On ValidationError, append a correction prompt and retry up to
             max_retries times.
        """
        if schema is None:
            raw = self._call_raw(turns, schema=None, model=model, llm_config_override=llm_config_override)
            return _strip_think_block(raw)

        current_turns = list(turns)
        last_error: ValidationError | None = None

        for attempt in range(max_retries + 1):
            raw = self._call_raw(current_turns, schema=schema, model=model, llm_config_override=llm_config_override)
            clean = _strip_code_fence(_strip_think_block(raw))
            healed = _heal_json(clean, schema)
            try:
                return schema.model_validate_json(healed)
            except ValidationError as e:
                last_error = e
                retry_note = (
                    " Retrying with correction prompt."
                    if attempt < max_retries
                    else " No retries left; propagating error."
                )
                dump_path = self._logger.dump_validation_error(
                    self._log_display, attempt, max_retries + 1, e, clean,
                    model=self._last_model_used,
                )
                location = f" See: {dump_path}" if dump_path else ""
                self._logger.log(
                    f"[{self._log_display}] Validation error "
                    f"(attempt {attempt + 1}/{max_retries + 1}).{retry_note}{location}"
                )
                if attempt == max_retries:
                    break
                current_turns = [
                    *current_turns,
                    {"role": "assistant", "content": clean},
                    {"role": "user", "content": (
                        f"Your previous response failed schema validation:\n{last_error}\n\n"
                        f"Required JSON schema:\n{json.dumps(schema.model_json_schema(), indent=2)}\n\n"
                        "Respond with ONLY a valid JSON object that matches the schema above. "
                        "Do NOT wrap in markdown fences, add explanations, or include any text outside the JSON."
                    )},
                ]

        raise last_error
