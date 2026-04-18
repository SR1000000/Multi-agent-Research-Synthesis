import operator
from typing import Annotated, Any, TypedDict, List, Optional, Literal
from pydantic import BaseModel, Field
from src.memory.research.schema import SlideContent

TITLE_SLIDE_NUMBER = 1
FIRST_CONTENT_SLIDE_NUMBER = 2
SlideKind = Literal["title", "content"]


def _blueprint_value(blueprint: Any, field: str, default: Any = None) -> Any:
    if isinstance(blueprint, dict):
        return blueprint.get(field, default)
    return getattr(blueprint, field, default)


def is_title_blueprint(blueprint: Any) -> bool:
    slide_kind = _blueprint_value(blueprint, "slide_kind")
    if slide_kind is not None:
        return slide_kind == "title"
    return _blueprint_value(blueprint, "slide_number") == TITLE_SLIDE_NUMBER


def group_allows_empty_chunks(blueprints: list[Any]) -> bool:
    return bool(blueprints) and all(is_title_blueprint(bp) for bp in blueprints)


class ErrorRecord(TypedDict):
    node: str
    error: str


# ---------------------------------------------------------------------------
# LLM-facing schemas (used only inside planner.py for structured LLM output)
# The LLM works with section labels (e.g. "S0", "S3"), never raw chunk IDs.
# ---------------------------------------------------------------------------

class LLMSlideBlueprint(BaseModel):
    """What the LLM produces per slide — references sections by label."""
    slide_number: int = Field(description="Slide number within the full deck. Start at 2 because Slide 1 is the reserved title slide.")
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
            "Reserved title slide title. Must be fewer than 7 words and suitable as a presentation headline."
        )
    )
    subtitle: str = Field(
        description=(
            "Reserved title slide subtitle. A concise supporting phrase or sentence for the audience."
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
    slide_kind: SlideKind = "content"
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
    prebuilt_content: Optional[SlideContent] = None


class SlideGroup(BaseModel):
    """A batch of slides assigned to one Slide Writer agent."""
    slide_blueprints: List[SlideBlueprint]
    rationale: str


class PresentationPlan(BaseModel):
    """Resolved plan stored in ResearchState — contains chunk IDs, not section labels."""
    thesis: str
    target_audience: str
    estimated_duration_minutes: int
    narrative_arc_summary: str
    slide_groups: List[SlideGroup]
    reasoning: str


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

    # -- slide coordination --
    max_slides:    int
    slide_numbers: List[int]

    # -- presentation plan (set by Planner, read by Plan Executor + Slide Writers) --
    presentation_plan: Optional[PresentationPlan]

    # -- slide completion tracking (append-only; one entry per Slide Writer call) --
    # Each entry: {"group_idx": int, "count": int}
    slides_written: Annotated[List[dict], operator.add]

    # -- observability --
    messages: Annotated[List[str], operator.add]
    errors:   Annotated[List[ErrorRecord], operator.add]
