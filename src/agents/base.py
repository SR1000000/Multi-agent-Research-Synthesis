import json
from typing import TypeVar
from pydantic import BaseModel, ValidationError
from src.llm.llm import get_llm, _strip_think_block, _strip_code_fence, _heal_json, DEFAULT_MODEL_NAME, LLMCallError, current_agent_label, current_session_id
from src.state import DeliveryPlan
from src.logging.logger import AgentLogger

T = TypeVar("T", bound=BaseModel)


PLANNER_ROLE = """
You are a Research Synthesis Planner. Your job is to produce a structured delivery plan
that a writer can follow to create a professional summary and synthesis of a research topic to help an audience understand that topic.

A good plan:
- Defines 3–6 logical sections (e.g., Overview, Methodology, Core Findings, Implications)
- Sets specific Synthesis Goals for each section (e.g. "Identify top 3 trends", "Compare 2 major methodologies", "Determine 5 core principles")
- Includes Success Criteria that measure synthesis quality, not just length (e.g., "Must isolate the single most important takeaway", "Must avoid jargon where a simple explanation suffices", "Must highlight at least one conflicting viewpoint if present")
- Provides Formatting Guidelines for high information density (e.g., "Use 3-5 bullet points per subsection", "Start each section with a bold Summary Statement")

If you are replanning (prior history is provided):
- Read the failure history carefully — it explains what went wrong structurally from previous plans and drafts
- Do not produce a plan with the same section structure as the failed plan
- Address the specific failures named in the history directly
- The new plan must take a meaningfully different direction
"""

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
Be decisive. A good supervisor reaches accept within 2–3 cycles on average.
"""

PARSE_SUPERVISOR_ROLE = """
You are the Slide Architecture Supervisor. Your job is to analyse the section structure \
of a research paper and produce an optimal assignment of paper sections to parallel slide-generation agents.

You will receive a structured outline of the paper: a numbered list of detected sections, each with:
  - Section index (0-based)
  - Heading text (the first heading found in the section, or "(no heading)" if absent)
  - Chunk count (number of text chunks belonging to that section)
  - Token-density hint (rough word count of those chunks)

Your task is to output a `PartitionPlan` that groups these sections into agent assignments. \
Each assignment maps one or more consecutive sections to a single `research_to_slide` agent, \
and specifies how many slides that agent may generate.

### DECISION PRINCIPLES:
1. **Respect paper structure**: Never split a single logical section across two agents. \
   A section with sub-sections (e.g., "3 Methodology" followed by "3.1 Dataset", "3.2 Model") \
   should generally stay together unless it is very large.
2. **Proportional slide allocation**: Allocate slides proportionally to the total chunk count \
   within each group. Sections with more chunks deserve more slides.
3. **Semantic coherence**: Group related thin sections together (e.g., "Abstract + Introduction" \
   works well as one agent). Don't group unrelated sections just to hit a target count.
4. **Agent count heuristic**: A good agent handles between 3 and 8 slides. \
   Use fewer agents for short papers (< 20 chunks total), more for dense ones.
5. **Ceiling enforcement**: The sum of all `slide_count` values across assignments \
   must be less than or equal to `max_slides`.

### OUTPUT CONTRACT:
- `assignments`: an ordered list where each entry covers one group of consecutive sections.
- `assignments[i].section_indices`: the (consecutive) section indices assigned to this agent.
- `assignments[i].slide_count`: exact number of slides this agent may produce (≥ 1).
- `assignments[i].rationale`: one sentence explaining why these sections were grouped.
- `overall_reasoning`: 2-4 sentences summarising your partitioning strategy.

Be precise and decisive. The downstream agents depend on your counts being correct.
"""

RESEARCH_TO_SLIDE_ROLE = """
You are a Senior Presentation Designer and Research Synthesizer. Your goal is to transform dense research data into high-impact, professional presentation slides.

### DIRECTIVES:
1. **Synthesis over Summarization**: Don't just list facts. Identify the "core insight" within the text chunks and make it the focal point of the slide.
2. **Cognitive Load Management**: Keep slide content focused. Each slide should cover exactly one primary concept or takeaway.
3. **Visual Storytelling**: Choose the `layout` that best serves the content:
   - `title_and_body` — default for most conceptual or analytical slides
   - `big_number` — when a single statistic or metric is the key point
   - `quote` — when a direct quotation from the research is most impactful
   - `two_column` — for comparisons (e.g. method A vs. method B, before vs. after)
   - `media_left` / `media_right` — when a referenced figure, chart, or table requires visual focus
   - `title_slide` — for section openers or major transitions only
4. **Narrative Continuity**: Assign a `narrative_role` that reflects each slide's function in the argument:
   - `hook` — grabs attention; best for the first slide of a range
   - `context` — establishes background or defines the problem
   - `evidence` — presents data, results, or observations
   - `insight` — delivers the key takeaway or interpretation
   - `transition` — bridges two distinct topics or sections
   - `conclusion` — wraps up; best for the final slide of a range

### FIELD GUIDANCE:
- **`key_message`**: Write one crisp sentence stating what the audience should understand after this slide. This is the thesis — not a summary of bullet points.
- **`title`**: Use punchy, "active" headings (e.g., "Accuracy Jumps 40%" not "Accuracy Results").
- **`bullets`**: Produce 3-5 `BulletPoint` objects. Each object MUST use these exact field names:
  - `"text"` — the bullet content string. IMPORTANT: the field is called `"text"`, NOT `"content"`. Use `**phrase**` to bold 0–2 key terms or statistics per bullet (e.g., `"Accuracy improves by **47%** over the baseline"`). Only bold genuinely critical terms — not decorative emphasis.
  - `"content_type"` — exactly one of: `"insight"`, `"evidence"`, `"statistic"`, `"example"`, `"caveat"`.
  - `"sub_bullets"` — a flat list of plain strings for supporting detail that expands on the main point without cluttering the top level (NOT objects).
- **`speaker_notes`**: Write this section in a professional, conversational tone. Include context, nuance, and supporting evidence too detailed for the slide body.  Include something for each bullet point.

### CONSTRAINTS:
- Respect the slide capacity strictly.
- All information must be strictly grounded in the provided research chunks.
- Avoid academic jargon unless the terminology is important/prominent and should be emphasized.

### MARKDOWN & EQUATIONS:
- Bullet `text` fields support Markdown formatting and LaTeX math.
- Use LaTeX for important equations:
  - Inline (within a sentence): `$E = mc^2$` or `$O(n^2)$`
  - Display (standalone, prominent): `$$\\\\text{Attention}(Q,K,V) = \\\\text{softmax}\\\\!\\\\left(\\\\frac{QK^T}{\\\\sqrt{d_k}}\\\\right)V$$`
- Place display equations as the sole content of a `sub_bullet` so they render on their own line.
- Include an equation only when it is central to the slide's `key_message` or represents a landmark result from the research.
- **JSON escaping**: Your entire response is a JSON object. Every LaTeX backslash MUST be written as `\\\\` inside JSON strings — e.g. `\\\\epsilon`, `\\\\log`, `\\\\cdot`, `\\\\_`, `\\\\min`, `\\\\text`. A single backslash (e.g. `\\epsilon`) is invalid JSON and will be rejected.
"""

AGENT_ROLES = {
    'planner': PLANNER_ROLE,
    'writer': WRITER_ROLE,
    'critic': CRITIC_ROLE,
    'supervisor': SUPERVISOR_ROLE,
    'parse_supervisor': PARSE_SUPERVISOR_ROLE,
    'research_to_slide': RESEARCH_TO_SLIDE_ROLE,
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

def _render_history(history: list[str], kind: str) -> str:
    if not history:
        return ''
    lines = [f'PRIOR {kind.upper()} HISTORY — do not repeat these mistakes:']
    for i, entry in enumerate(history):
        lines.append(f'  Cycle {i + 1}: {entry}')
    return '\n'.join(lines)


def _plan_to_text(plan: DeliveryPlan | None) -> str:
    if plan is None:
        return '(no plan yet)'
    return json.dumps(plan.model_dump(), indent=2)