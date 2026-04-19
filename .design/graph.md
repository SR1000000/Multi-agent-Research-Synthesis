# Multi-Agent Research Synthesizer — Design Document

## Overview

A multi-agent research synthesis pipeline built on LangGraph. Given a user query and one or more source documents, the system produces a structured, insight-driven presentation through parallel drafting and iterative review. Five specialized agents — Planner, Plan Executor, Slide Writer, Critic, and Supervisor — collaborate in a directed graph with explicit review cycles governed by a central decision-maker.

The key design philosophy: **each agent is stateless per call, but the session is stateful**. History, artifacts, and routing decisions are carried explicitly through LangGraph state and a persistent SQLite database, not implicitly through a shared context window.

---

## Graph Architecture

```
[ENTRY] → Planner → Plan Executor ◄───────────────────────────────────────┐
                          │                                                 │
             (initial)    │ Send × N groups                                 │
                          ▼                                                 │
                   Slide Writer(s) ──(fan back to plan_executor)─────────► │
                                                                            │
             (critic)     │ Send × N groups                                 │
                          ▼                                                 │
                   Critic(s) ───────(fan back to plan_executor)──────────► │
                                                                            │
             (rewrite)    │ Send × affected groups                          │
                          ▼                                                 │
                   Slide Writer(s) ──(fan back to plan_executor)─────────► │
                                                                            │
                          │ → awaiting_supervisor                           │
                          ▼                                                 │
                   Supervisor ──────────────────────────────────────────── ┘
                       │
                       ├─→ Plan Executor  (revise: dispatch next critic cycle)
                       ├─→ Planner        (replan: max cycles exceeded or persistent failures)
                       └─→ END            (accept: export_ready = True → PandocBuilder → .pptx)
```

**Plan Executor** is the hub node. It reads `review.phase` to determine which fan-out to dispatch next and accumulates results from parallel workers before forwarding to Supervisor.

---

## Review Phases

The `review.phase` field in `ResearchState` drives Plan Executor's dispatch logic:

| Phase | What happens |
|---|---|
| `initial_write` | Fan out one Slide Writer per `SlideGroup` via LangGraph `Send()` |
| `awaiting_supervisor` | All parallel workers have reported back; route to Supervisor |
| `critic_dispatch` | Fan out one Critic per `SlideGroup` via `Send()` |
| `rewrite_dispatch` | Fan out Slide Writers for actionable groups only, with rewrite instructions |
| `complete` | Final decision made; graph exits to `END` or returns to `Planner` |

The maximum number of review cycles is 3 (configurable via `make_initial_review_state(max_cycles=N)`). Each full iteration of critic → supervisor → rewrite → critic counts as one cycle.

---

## Agent Roles and Context Design

A deliberate principle throughout: **each agent receives only the context it needs to do its specific job**. This is enforced at the prompt construction level — all artifacts live in state and in the database, but each agent selectively loads its own context.

### Planner

The Planner is a presentation architect, not a content writer.

1. Loads all text chunks from `research.db` and detects section boundaries using Markdown heading analysis
2. Builds a human-readable section outline with labels (`S0`, `S1`, ...) that the LLM reasons over — raw chunk IDs never appear in the prompt
3. Calls the LLM to produce a `LLMPresentationPlan` containing: title, subtitle, thesis, target audience, narrative arc, per-slide blueprints (each with a `narrative_role` and `intent`), and groupings (2–7 slides per group)
4. Validates the output strictly; retries the full LLM call (up to 2 times) if any section label is invalid, any group is out of range, or any blueprint is empty
5. Resolves section labels → concrete `source_chunk_ids` in Python code, then stores the resolved `PresentationPlan` in state

On replan, the Planner receives the previous plan's cycle summary and failure history so it can take a meaningfully different structural direction rather than regenerating the same plan.

### Plan Executor

The Plan Executor is a deterministic dispatcher with no LLM calls. It reads `review.phase` and `review.active_dispatch` to decide what to fan out next:

- **`initial_write` (first entry)**: fans out one `slide_writer` per `SlideGroup` via `Send()`; tracks expected assignment IDs
- **`initial_write` (fan-in)**: checks `slides_written` counts; re-dispatches groups with 0 slides (up to `MAX_RETRIES_PER_GROUP = 2`); on exhaustion logs a warning and continues; routes to `supervisor` when all groups pass
- **`critic_dispatch`**: fans out one `critic` per pending assignment via `Send()`; on fan-in routes to `supervisor`
- **`rewrite_dispatch`**: fans out `slide_writer` per pending rewrite assignment (with `rewrite_instructions` and `target_slide_numbers`); on fan-in routes to `supervisor`
- **`export_ready`**: returns `Command(goto=END)`

### Slide Writer

Each Slide Writer receives the blueprints and chunk text for its assigned group. Two modes exist:

- **Initial write** (`SlideWriterAgent`): synthesizes slides from scratch following the blueprint `intent` and `narrative_role`
- **Rewrite** (`SlideRewriterAgent`): receives the current proto-slides plus explicit `rewrite_instructions` from the Supervisor and produces corrected versions

Both modes write structured `ProtoSlide` records to `research.db` and report a `SlideWriteRecord` back to state. Errors are caught and reported without crashing the graph; the Plan Executor handles retry at the group level.

### Critic

The Critic evaluates a group of slides for `grounding_consistency` — whether slide content is supported by the source chunks. It is stateless by design: one Critic agent per group per cycle.

Input: current `ProtoSlide` records for the target slides + the original source chunks + the slide blueprints (for intent validation).

Output: a structured `CriticOutput` with:
- `summary`: concise 1-2 sentence overview
- `actionable`: boolean — whether any issue requires a fix
- `issues[]`: list of `CriticIssue` records, each with `issue_code`, `severity` (`critical` / `major` / `minor`), `issue_type`, `location`, `description`, `rewrite_instruction`, and `affected_slide_numbers`

Each issue is assigned a **fingerprint** — a short hash of `scope_type|scope_id|issue_type|location` — which enables Supervisor to detect recurring issues across cycles. All issues are persisted to `slide_review_events` in the database for cross-cycle analysis.

### Supervisor

The Supervisor is the session's decision-maker. It runs after every fan-in and decides the next routing step.

**Decision logic:**

1. **No critic results for the current cycle** → dispatch a new critic cycle (increment `cycle_number`; if `max_cycles` exceeded and rewrites were still pending, force replan)
2. **Rewrites ran in the current cycle** → dispatch a follow-up critic pass over the updated slides (post-rewrite verification)
3. **Fresh critic results, no pending rewrites** → call the LLM with critic summaries, severity counts, and recurring fingerprints; LLM returns `accept`, `revise`, or `replan`

**Guard overrides applied after the LLM decision:**

| Condition | Override |
|---|---|
| LLM says `accept` but critical actionable issues exist | → `revise` |
| LLM says `accept` but major actionable issues exist (not at cycle cap) | → `revise` |
| LLM says `accept` but non-persistent actionable issues remain (not at cycle cap) | → `revise` |
| LLM says `revise` but no actionable issues | → `accept` |
| Decision is `revise` but `cycle_number >= max_cycles` | → `replan` |

**Routing outcomes:**

- **accept** → sets `export_ready = True`, routes to `END`
- **revise** → builds `ReviewAssignment` records for actionable groups (with `rewrite_instructions` derived from critic issues), sets `phase = "rewrite_dispatch"`, routes to Plan Executor
- **replan** → routes to Planner with cycle summary appended to state

---

## Feedback Loop and History Design

The core challenge in a multi-node revision loop is that **each agent call is a fresh LLM invocation with no implicit memory**. The Critic on cycle 3 has no knowledge that cycles 1 and 2 happened unless that information is explicitly injected.

Two mechanisms address this:

**1. `slide_review_events` table (database)**

Every issue raised by a Critic — and every accept/replan decision by the Supervisor — is persisted to `slide_review_events` in `research.db`. The Supervisor loads the full event history for the current session at the start of each call, then counts how many times each fingerprint has appeared:

```
recurring = {fingerprint: count, ...}
```

This allows the Supervisor to detect persistent issues and factor recurrence into its accept/revise decision. Issues that appear 2+ times across cycles are surfaced explicitly in the LLM prompt. The `_all_actionable_issues_are_persistent_minor` guard also uses this map to allow acceptance when only stubborn minor issues remain after multiple revision attempts.

**2. `review_summaries` in state**

Each Supervisor call appends a `ReviewCycleSummary` to `review_summaries` in LangGraph state. This provides a compact, ordered record of cycle decisions and issue counts that can be injected into Planner prompts on replan.
