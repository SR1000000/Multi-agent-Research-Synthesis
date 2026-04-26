"""
state.py — Shared type definitions and graph state for the research-synthesis pipeline.

This module defines the durable schemas shared across the research LangGraph
pipeline.  It focuses on state that is stored in or derived from ``ResearchState``;
some node-local ``Send`` payloads live beside the agents that consume them.

Layout
------
 Constants
 LLM-facing schemas                  – what the Planner LLM produces (section labels)
 State-facing schemas                – resolved versions stored in ResearchState (chunk IDs)
 Review sub-state                    – coordinator structures for the critic/rewrite cycle
 Graph state (ResearchState)         – the top-level LangGraph state TypedDict
"""

import operator
from typing import Annotated, TypedDict, List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field



# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_CYCLES = 4
"""Default cap on critic/rewrite cycles before the supervisor must make a terminal decision."""

MAX_REPLANS = 2
"""Upper bound on full deck replans—running the planner again after a supervisor ``replan`` or
equivalent structural reset—within one research session.

Prevents unbounded plan → review → discard → plan loops while still allowing a few recovery attempts."""


# ---------------------------------------------------------------------------
# Literal type aliases
# Used across multiple schemas and routing functions for type-safe dispatch.
# ---------------------------------------------------------------------------

ReviewScopeType = Literal["deck", "group", "slide"]
"""Granularity at which a critic or rewrite assignment targets slides."""

ReviewCheckType = Literal["grounding_consistency"]
"""Kind of quality check being performed.  Currently only grounding/consistency is implemented."""

ReviewDispatchKind = Literal["initial_write", "critic", "rewrite"]
"""Which dispatch phase an ``ActiveDispatch`` belongs to."""

ReviewPhase = Literal[
    "initial_write",
    "awaiting_supervisor",
    "critic_dispatch",
    "rewrite_dispatch",
    "complete",
]
"""Finite states of the review sub-state machine tracked in ``ReviewState.phase``."""


# ---------------------------------------------------------------------------
# LLM-facing schemas
# Used exclusively inside planner.py for structured LLM output.
# The LLM works with section labels (e.g. "S0", "S3"), never raw chunk IDs.
# Planner Python code is responsible for resolving labels to chunk IDs before
# writing anything to ResearchState.
#
# Kept separate because these schemas are fed into the LLM, so they need to only contain
# relevant fields for the LLM to generate.  We don't want to give the LLM fields and then
# tell it to ignore them.
# ---------------------------------------------------------------------------

class LLMSlideBlueprint(BaseModel):
    """Slide specification as produced directly by the Planner LLM.

    The LLM populates ``source_sections`` with short section labels (e.g. ``"S0 Attention..."``)
    rather than raw chunk IDs to prevent mixing up section labels (which would be easier to hallucinate with pure numbers).  
    The Planner resolves these to concrete chunk IDs and stores the result in ``SlideBlueprint.source_chunk_ids`` 
    before committing to state.
    """

    slide_number: int = Field(
        description=(
            "Slide number for this content slide within the persisted deck (1..N). "
            "The title slide is not a blueprint; it comes from presentation metadata at export."
        )
    )
    working_title: str = Field(description="Punchy working title for the slide")
    narrative_role: Literal[
        "hook",
        "problem",
        "evidence",
        "insight",
        "transition",
        "call_to_action",
        "conclusion",
    ] = Field(
        description=(
            "Role this slide plays in the deck: "
            "hook | problem | evidence | insight | transition | call_to_action | conclusion"
        )
    )
    intent: str = Field(
        description=(
            "A precise directive for the Slide Writer — what argument to make or what the "
            "audience should understand after this slide. Do NOT write content here; write instructions."
        )
    )
    source_sections: List[str] = Field(
        description=(
            'Section labels from the outline this slide draws from, e.g. ["S0", "S3"]. '
            "Must reference labels that appear in the outline exactly as shown."
        )
    )


class LLMSlideGroup(BaseModel):
    """A batch of slides to be written by one parallel Slide Writer agent.

    The Planner LLM emits an ordered list of these groups; each group maps to one
    ``send()`` call that spawns an independent Slide Writer in the LangGraph fan-out.
    """

    slide_blueprints: List[LLMSlideBlueprint] = Field(
        description="2 to 7 slide blueprints assigned to this agent"
    )
    rationale: str = Field(
        description="One sentence explaining why these slides are grouped together"
    )


class LLMPresentationPlan(BaseModel):
    """Complete presentation plan as emitted by the Planner LLM.

    This is the *raw* structured output from the LLM.  It uses section labels and
    is not stored directly in ``ResearchState``; the Planner resolves it into a
    ``PresentationPlan`` first.
    """

    title: str = Field(
        description=(
            "Presentation title. Must be fewer than 7 words and suitable as a presentation headline."
        )
    )
    subtitle: str = Field(
        description=(
            "Presentation subtitle. A concise supporting phrase or sentence for the audience."
        )
    )
    thesis: str = Field(
        description=(
            "The central argument or takeaway of the presentation in 1-2 sentences. "
            "This is what distinguishes a presentation from a summary."
        )
    )
    target_audience: str = Field(
        description="Who this presentation is for (from the user query)"
    )
    estimated_duration_minutes: int = Field(
        description="Estimated presentation length in minutes (1-1.5 min per slide)"
    )
    narrative_arc_summary: str = Field(
        description=(
            "2-3 sentences describing the overall structure and flow of the presentation. "
            "A narrative arc (hook → problem → evidence → insight → conclusion) is one "
            "valid approach but not required."
        )
    )
    slide_groups: List[LLMSlideGroup] = Field(
        description="Ordered list of slide groups; each group = one parallel Slide Writer agent"
    )
    reasoning: str = Field(
        description="2-4 sentences explaining the key structural choices made in this plan"
    )


# ---------------------------------------------------------------------------
# State-facing schemas
# Stored in ResearchState, consumed by the Plan Executor and Slide Writers.
# Planner Python code resolves section labels to chunk IDs before creating these.
# ---------------------------------------------------------------------------

class SlideBlueprint(BaseModel):
    """Resolved slide specification with concrete chunk IDs, ready for the Slide Writer.

    This is the persisted counterpart to ``LLMSlideBlueprint``.  The only
    structural difference is that ``source_chunk_ids`` replaces ``source_sections``;
    all other fields carry the same semantics.
    """

    slide_number: int
    working_title: str
    narrative_role: Literal[
        "hook",
        "problem",
        "evidence",
        "insight",
        "transition",
        "call_to_action",
        "conclusion",
    ]
    intent: str
    source_chunk_ids: List[str]
    """Concrete database chunk IDs resolved from the LLM's section labels."""


class SlideGroup(BaseModel):
    """A resolved batch of narratively coherent slides assigned to one Slide Writer agent.

    Stored in ``PresentationPlan.slide_groups``; mirrors ``LLMSlideGroup`` but
    uses resolved ``SlideBlueprint`` objects instead of ``LLMSlideBlueprint``.
    """

    slide_blueprints: List[SlideBlueprint]
    rationale: str


class PresentationPlan(BaseModel):
    """Fully resolved presentation plan stored in ``ResearchState``.

    This is the single authoritative plan consumed by the Plan Executor and all
    Slide Writer agents.  It contains concrete chunk IDs (not section labels) and
    is written by the Planner exactly once per plan generation.
    """

    title: str
    subtitle: str
    thesis: str
    target_audience: str
    estimated_duration_minutes: int
    narrative_arc_summary: str
    slide_groups: List[SlideGroup]
    """Ordered groups; each group spawns one parallel Slide Writer agent."""
    reasoning: str


# ---------------------------------------------------------------------------
# Review sub-state schemas
# These TypedDicts track the critic/rewrite coordination loop managed by the
# Supervisor agent.  They live inside ReviewState, which is itself a field of
# ResearchState.
# ---------------------------------------------------------------------------

class ReviewAssignment(TypedDict):
    """A unit of work dispatched to either a Slide Critic or a Slide Writer (rewrite).

    Created by the Supervisor and consumed by Plan Executor fan-out.  Current
    assignments are group-scoped, with rewrites optionally narrowed to the
    affected slide numbers within that group.  Correlating writes uses
    ``dispatch_id`` (session-unique) rather than a per-plan counter on each
    record.

    Fields
    ------
    assignment_id:
        Identifier for this assignment within its dispatch pattern.
    cycle_number:
        Which critic/rewrite iteration this assignment belongs to (0-based).
    check_type:
        The quality-check category being performed (see ``ReviewCheckType``).
    scope_type:
        Granularity of the review — full deck, a slide group, or a single slide.
    scope_id:
        Human-readable or stable machine label for the scope.  Group reviews
        currently use the zero-based group index as a string, e.g. ``"0"``.
    group_idx:
        Zero-based index into ``PresentationPlan.slide_groups``.
    chunk_ids:
        Source chunk IDs available to the writer/critic for this assignment.
    slide_blueprints:
        Serialised ``SlideBlueprint`` dicts for the slides in scope.
    target_slide_numbers:
        Ordered list of slide numbers covered by this assignment.
    rewrite_instructions:
        Freeform instructions from the Supervisor or Critic telling the writer
        what to fix.  Empty string for critic assignments.
    """

    assignment_id: str
    cycle_number: int
    check_type: ReviewCheckType
    scope_type: ReviewScopeType
    scope_id: str
    group_idx: int
    chunk_ids: List[str]
    slide_blueprints: List[dict]
    target_slide_numbers: List[int]
    rewrite_instructions: str


class ActiveDispatch(TypedDict):
    """Tracks the single in-flight batch of assignments the Supervisor is waiting on.

    Only one ``ActiveDispatch`` exists at a time.  The Supervisor sets this when
    it fans out a set of assignments and clears it once all expected results have
    arrived in ``ResearchState``.

    Fields
    ------
    dispatch_id:
        Unique identifier for this dispatch batch.
    kind:
        Whether the batch is an initial write, a critic pass, or a rewrite pass.
    cycle_number:
        Review cycle this dispatch belongs to.
    expected_assignment_ids:
        IDs of all assignments expected in this batch.  Plan Executor uses this
        list to know how many matching writer/critic records must arrive before
        fan-in can continue.
    """

    dispatch_id: str
    kind: ReviewDispatchKind
    cycle_number: int
    expected_assignment_ids: List[str]


class SlideWriteRecord(TypedDict):
    """Immutable record appended to ``ResearchState.slides_written`` when a Slide Writer completes.

    Used by the Supervisor to detect when all slides in a dispatch batch have
    been written, and to audit which group/assignment each write belongs to.

    Fields
    ------
    dispatch_id:
        Dispatch batch that triggered this write.
    assignment_id:
        Specific assignment within the batch.
    group_idx:
        Zero-based index of the slide group that was written.
    count:
        Number of slides produced in this write (may differ from
        ``len(target_slide_numbers)`` if partial failures occurred).
    target_slide_numbers:
        The slide numbers the writer was instructed to produce.
    """

    dispatch_id: str
    assignment_id: str
    group_idx: int
    count: int
    target_slide_numbers: List[int]


class CriticIssueRecord(TypedDict):
    """A single actionable quality issue identified by the Slide Critic.

    Stored inside ``CriticResultRecord.issues``.

    Fields
    ------
    issue_code:
        Short machine-readable code classifying the issue (e.g. ``"UNSUPPORTED_CLAIM"``).
    severity:
        Impact level used by the Supervisor's routing policy.  Critical and major
        issues normally force revision unless the cycle-cap logic takes over;
        minor issues may be accepted if they persist across cycles.
    issue_type:
        Broader category the issue belongs to (e.g. ``"grounding"``, ``"clarity"``).
    location:
        Human-readable pointer to where the issue appears (e.g. ``"slide_3, bullet 2"``).
    fingerprint:
        Deterministic hash of the issue content used to de-duplicate issues across cycles.
    affected_slide_numbers:
        Which slide numbers are impacted by this issue.
    rewrite_instruction:
        Specific, actionable directive telling the Slide Writer how to fix this issue.
    """

    issue_code: str
    severity: Literal["critical", "major", "minor"]
    issue_type: str
    location: str
    fingerprint: str
    affected_slide_numbers: List[int]
    rewrite_instruction: str


class CriticResultRecord(TypedDict):
    """Complete output of one Slide Critic invocation, appended to ``ResearchState.critic_results``.

    One record is produced per critic assignment.  The Supervisor reads these
    records to decide whether to accept the deck, trigger rewrites, or replan.

    Fields
    ------
    dispatch_id:
        Dispatch batch this result belongs to.
    assignment_id:
        Specific assignment this result corresponds to.
    cycle_number:
        Review cycle in which the critique was performed.
    check_type:
        Quality-check category that was applied.
    scope_type:
        Granularity of the review scope.
    scope_id:
        Human-readable label for the scope.
    group_idx:
        Zero-based index of the slide group that was reviewed.
    target_slide_numbers:
        Slide numbers that were in scope for this critique.
    actionable:
        ``True`` when the critic produced at least one issue it believes requires
        correction.  The Supervisor may still accept persistent low-risk issues.
    rewrite_instructions:
        Aggregated rewrite instructions (concatenated from all actionable issues).
    summary:
        One-paragraph human-readable summary of the critique.
    issues:
        Detailed list of individual ``CriticIssueRecord`` objects.
    """

    dispatch_id: str
    assignment_id: str
    cycle_number: int
    check_type: ReviewCheckType
    scope_type: ReviewScopeType
    scope_id: str
    group_idx: int
    target_slide_numbers: List[int]
    actionable: bool
    rewrite_instructions: str
    summary: str
    issues: List[CriticIssueRecord]


class ReviewCycleSummary(TypedDict):
    """Snapshot of one complete critic → decision cycle, appended to ``ResearchState.review_summaries``.

    Written by the Supervisor after evaluating all critic results for a cycle.
    Provides an audit trail of how many issues were found and what was decided.

    Fields
    ------
    plan_number:
        Which plan (1-based session plan counter) this summary belongs to.
    cycle_number:
        Which iteration this summary covers (0-based).
    issue_counts:
        Mapping of severity label → count of issues found this cycle
        (e.g. ``{"critical": 0, "major": 2, "minor": 5}``).
    decision:
        Supervisor decision: ``"accept"``, ``"revise"``, or ``"replan"``.
    routing:
        Next LangGraph node the Supervisor routed to after this decision.
    rewrites_required_by_assignment:
        Maps each assignment ID to a boolean indicating whether a rewrite was
        required for that assignment's slides.
    """

    plan_number: int
    cycle_number: int
    issue_counts: dict[str, int]
    decision: str
    routing: str
    rewrites_required_by_assignment: dict[str, bool]


class ReviewState(TypedDict):
    """Mutable sub-state for the critic/rewrite coordination loop.

    Stored as ``ResearchState.review`` and updated in-place (not append-only)
    by the Supervisor and Plan Executor as the review cycle progresses.

    Fields
    ------
    phase:
        Current position in the review state machine (see ``ReviewPhase``).
    cycle_number:
        Current critic/rewrite cycle number.  It starts at 0 before the first
        critic dispatch and is advanced when a new critic cycle is launched.
    last_critic_dispatch_id:
        ``dispatch_id`` of the most recently completed critic batch; the
        Supervisor filters ``critic_results`` to this id so only the current
        batch is read. ``dispatch_id`` values stay unique across the session
        because ``dispatch_counter`` is carried on replan.
    max_cycles:
        Upper bound on critic/rewrite iterations before the Supervisor must make
        a terminal decision or route to a full replan.
    dispatch_counter:
        Monotonically increasing counter used to generate unique dispatch IDs.
    active_dispatch:
        The in-flight dispatch batch the Supervisor is currently waiting on,
        or ``None`` if no dispatch is active.
    pending_critic_assignments:
        Queue of assignments awaiting dispatch to Slide Critic agents.
    pending_rewrite_assignments:
        Queue of assignments awaiting dispatch to Slide Writer agents for rewrites.
    last_critic_assignment_ids:
        Assignment IDs from the most recently completed critic dispatch; used to
        correlate results with the correct cycle.
    last_rewrite_assignment_ids:
        Assignment IDs from the most recently completed rewrite dispatch.
    last_issue_counts:
        Issue severity counts from the most recent completed critic cycle.
    last_rewrites_required_by_assignment:
        Rewrite-required flags from the most recent completed critic cycle.
    last_failed_assignment_ids:
        Assignment IDs that failed (e.g. agent error) in the most recent dispatch.
    final_decision:
        Terminal decision once the review loop exits: ``"accept"``, ``"replan"``,
        or ``"skipped"`` (when ``skip_supervisor`` is ``True``).
    export_ready:
        Set to ``True`` once the deck can proceed to export.  This can be set by
        the Supervisor after acceptance or by Plan Executor when supervisor review
        is explicitly skipped.
    """

    phase: ReviewPhase
    cycle_number: int
    last_critic_dispatch_id: Optional[str]
    max_cycles: int
    dispatch_counter: int
    active_dispatch: Optional[ActiveDispatch]
    pending_critic_assignments: List[ReviewAssignment]
    pending_rewrite_assignments: List[ReviewAssignment]
    last_critic_assignment_ids: List[str]
    last_rewrite_assignment_ids: List[str]
    last_issue_counts: dict[str, int]
    last_rewrites_required_by_assignment: dict[str, bool]
    last_failed_assignment_ids: List[str]
    final_decision: Optional[Literal["accept", "replan", "skipped"]]
    export_ready: bool


def make_initial_review_state(*, max_cycles: int = MAX_CYCLES) -> ReviewState:
    """Return a zeroed-out ``ReviewState`` ready for the start of a new session.

    All queues are empty, counters are zero, and the phase is set to
    ``"initial_write"`` so Plan Executor knows to fan out the first batch of
    Slide Writer assignments immediately.

    Parameters
    ----------
    max_cycles:
        Maximum number of critic/rewrite iterations allowed before the Supervisor
        must make a terminal decision.  Defaults to ``MAX_CYCLES``.

    Returns
    -------
    ReviewState
        A fully initialised ``ReviewState`` dict.
    """
    return {
        "phase": "initial_write",
        "cycle_number": 0,
        "last_critic_dispatch_id": None,
        "max_cycles": max_cycles,
        "dispatch_counter": 0,
        "active_dispatch": None,
        "pending_critic_assignments": [],
        "pending_rewrite_assignments": [],
        "last_critic_assignment_ids": [],
        "last_rewrite_assignment_ids": [],
        "last_issue_counts": {"critical": 0, "major": 0, "minor": 0},
        "last_rewrites_required_by_assignment": {},
        "last_failed_assignment_ids": [],
        "final_decision": None,
        "export_ready": False,
    }


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class ResearchState(TypedDict):
    """Top-level LangGraph state shared by every node in the pipeline.

    This TypedDict is the top-level container passed through the main graph.
    Parallel ``Send`` nodes may receive narrower dispatch payloads derived from it.
    Fields annotated with ``operator.add`` are append-only: LangGraph merges
    updates by concatenating lists rather than replacing them, which makes
    fan-out safe without explicit locking.

    Immutable core
    --------------
    query:
        The original user research question driving this session.
    session_id:
        UUID-style identifier for the session, used in filenames and DB keys.
    created_at:
        ISO-8601 UTC timestamp when the session was created.
    doc_ids:
        Database IDs of all source documents available to this session.
    paper_titles:
        Human-readable titles corresponding to each entry in ``doc_ids``.
    doc_id:
        Primary document ID when the session targets a single paper.
    paper_title:
        Title of the primary document.

    Slide coordination
    ------------------
    max_slides:
        Soft upper bound on total deck size, including the generated title slide.
        The Planner therefore validates content slides against ``max_slides - 1``.
    slide_numbers:
        Legacy/session-provided slide number list.  The current Planner derives
        authoritative slide numbers from ``presentation_plan`` instead.
    skip_supervisor:
        When ``True``, the pipeline bypasses all critic/supervisor cycles after
        the initial write and proceeds directly to export.

    Presentation plan
    -----------------
    presentation_plan:
        The resolved ``PresentationPlan`` set by the Planner and consumed by
        the Plan Executor and all Slide Writer agents.  ``None`` until the
        Planner has successfully completed.

    Review coordination
    -------------------
    review:
        Mutable ``ReviewState`` sub-dict tracking where the session is in the
        critic/rewrite loop.

    plan_number:
        1-based count of which presentation plan the session is on; incremented
        on each full replan.
    force_replan_at_max_cycles:
        Test-only: when true, first ``MAX_REPLANS`` times the supervisor is at
        the critic/rewrite cap, force ``replan`` so the test harness can
        exercise replan without LLM choice.

    Append-only execution records
    -----------------------------
    slides_written:
        One ``SlideWriteRecord`` per completed Slide Writer invocation.
    critic_results:
        One ``CriticResultRecord`` per completed Slide Critic invocation.
    review_summaries:
        One ``ReviewCycleSummary`` per completed Supervisor decision cycle.

    RAG / tool tracing
    ------------------
    source_chunks:
        Legacy slot for raw retrieval results.  Current writer/critic agents
        persist retrieval traces through ``retrieval_queries``, ``tool_calls``,
        and ``tool_results`` instead.
    retrieval_queries:
        All queries issued to the vector store during this session.
    tool_calls:
        Serialised records of every tool invocation made by any agent.
    tool_results:
        Corresponding results for each entry in ``tool_calls``.

    Observability
    -------------
    messages:
        Human-readable log messages emitted by agents for tracing/debugging.
    errors:
        Structured ``{"node": str, "error": str}`` entries for non-fatal errors.
    """

    # -- immutable core --
    query:        str
    session_id:   str
    created_at:   str
    doc_ids:      List[str]
    paper_titles: List[str]
    doc_id:       str
    paper_title:  str

    # -- slide coordination --
    max_slides:    int
    slide_numbers: List[int]
    skip_supervisor: bool
    plan_number: int
    force_replan_at_max_cycles: bool

    # -- presentation plan (set by Planner, read by Plan Executor + Slide Writers) --
    presentation_plan: Optional[PresentationPlan]

    # -- review coordination --
    review: ReviewState

    # -- append-only execution records --
    slides_written:   Annotated[List[SlideWriteRecord], operator.add]
    critic_results:   Annotated[List[CriticResultRecord], operator.add]
    review_summaries: Annotated[List[ReviewCycleSummary], operator.add]

    # -- RAG / tool tracing --
    source_chunks:     List[Any]
    retrieval_queries: Annotated[List[str], operator.add]
    tool_calls:        Annotated[List[Dict[str, Any]], operator.add]
    tool_results:      Annotated[List[Dict[str, Any]], operator.add]

    # -- observability --
    messages: Annotated[List[str], operator.add]
    errors:   Annotated[List[Dict[str, str]], operator.add]
