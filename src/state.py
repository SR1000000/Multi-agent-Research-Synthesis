import operator
from typing import Annotated, TypedDict, List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field

# Persisted / planned content slides are numbered 1..N. The opening title slide is
# not stored; it is added at export from presentation_plan.title/subtitle.
FIRST_CONTENT_SLIDE_NUMBER = 1


class ErrorRecord(TypedDict):
    node: str
    error: str


# ---------------------------------------------------------------------------
# Dormant synthesis-pipeline schemas
# Preserved for older agent modules that still import them during the rebase.
# ---------------------------------------------------------------------------

class SectionBlock(BaseModel):
    title: str
    queries: List[str]
    notes: str


class DeliveryPlan(BaseModel):
    title: str
    guidelines: Dict[str, Any]
    success_criteria: List[str]
    introduction: str
    sections: List[SectionBlock]
    conclusion: str


class IssueItem(BaseModel):
    id: str = Field(description="e.g. ISS_001")
    location: str
    type: str
    severity: str
    description: str


class CritiqueOutput(BaseModel):
    summary: str
    issues: List[IssueItem]


class Draft(TypedDict):
    version: int
    document: str
    word_count: int
    action: str
    created_at: str


ReviewScopeType = Literal["deck", "group", "slide"]
ReviewCheckType = Literal["grounding_consistency"]
ReviewDispatchKind = Literal["initial_write", "critic", "rewrite"]
ReviewPhase = Literal[
    "initial_write",
    "awaiting_supervisor",
    "critic_dispatch",
    "rewrite_dispatch",
    "complete",
]


# ---------------------------------------------------------------------------
# LLM-facing schemas (used only inside planner.py for structured LLM output)
# The LLM works with section labels (e.g. "S0", "S3"), never raw chunk IDs.
# ---------------------------------------------------------------------------

class LLMSlideBlueprint(BaseModel):
    """What the LLM produces per slide — references sections by label."""
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
    """A batch of slides to be written by one parallel Slide Writer agent."""
    slide_blueprints: List[LLMSlideBlueprint] = Field(
        description="2 to 7 slide blueprints assigned to this agent"
    )
    rationale: str = Field(
        description="One sentence explaining why these slides are grouped together"
    )


class LLMPresentationPlan(BaseModel):
    """Schema the LLM must produce — section-label references, no chunk IDs."""
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
# State-facing schemas (stored in ResearchState, consumed by Plan Executor
# and Slide Writers). Planner Python code resolves section labels → chunk IDs.
# ---------------------------------------------------------------------------

class SlideBlueprint(BaseModel):
    """Resolved blueprint with concrete chunk IDs ready for the Slide Writer."""
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


class SlideGroup(BaseModel):
    """A batch of slides assigned to one Slide Writer agent."""
    slide_blueprints: List[SlideBlueprint]
    rationale: str


class PresentationPlan(BaseModel):
    """Resolved plan stored in ResearchState — contains chunk IDs, not section labels."""
    title: str
    subtitle: str
    thesis: str
    target_audience: str
    estimated_duration_minutes: int
    narrative_arc_summary: str
    slide_groups: List[SlideGroup]
    reasoning: str


class ReviewAssignment(TypedDict):
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
    dispatch_id: str
    kind: ReviewDispatchKind
    cycle_number: int
    expected_assignment_ids: List[str]


class SlideWriteRecord(TypedDict):
    dispatch_id: str
    assignment_id: str
    group_idx: int
    count: int
    target_slide_numbers: List[int]


class CriticIssueRecord(TypedDict):
    issue_code: str
    severity: Literal["critical", "major", "minor"]
    issue_type: str
    location: str
    description: str
    fingerprint: str
    affected_slide_numbers: List[int]
    rewrite_instruction: str


class CriticResultRecord(TypedDict):
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
    cycle_number: int
    issue_counts: dict[str, int]
    decision: str
    routing: str
    rewrites_required_by_assignment: dict[str, bool]


class ReviewState(TypedDict):
    phase: ReviewPhase
    cycle_number: int
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


def make_initial_review_state(*, max_cycles: int = 3) -> ReviewState:
    return {
        "phase": "initial_write",
        "cycle_number": 0,
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
    skip_supervisor: bool  # When True, bypass critic/supervisor cycles after initial write


    # -- presentation plan (set by Planner, read by Plan Executor + Slide Writers) --
    presentation_plan: Optional[PresentationPlan]
    
    # -- review coordination --
    review: ReviewState

    # -- append-only execution records --
    slides_written: Annotated[List[SlideWriteRecord], operator.add]
    critic_results: Annotated[List[CriticResultRecord], operator.add]
    review_summaries: Annotated[List[ReviewCycleSummary], operator.add]
    
    
    source_chunks: List[Any]
    retrieval_queries: Annotated[List[str], operator.add]
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]
    tool_results: Annotated[List[Dict[str, Any]], operator.add]

    # -- observability --
    messages: Annotated[List[str], operator.add]
    errors:   Annotated[List[ErrorRecord], operator.add]
