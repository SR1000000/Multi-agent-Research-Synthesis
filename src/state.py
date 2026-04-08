import operator
from datetime import datetime, timezone
from typing import Annotated, TypedDict, List, Dict, Optional, Any
from pydantic import BaseModel, Field

class ErrorRecord(TypedDict):
    node: str
    error: str

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
    type: str   # factual_inaccuracy | hallucination | unsupported_claim
                # logical_gap | structural | clarity | contradiction
    severity: str  # critical | major | minor
    description: str

class CritiqueOutput(BaseModel):
    summary: str
    issues: List[IssueItem]

class SupervisorOutput(BaseModel):
    decision: str   # "accept" | "revise" | "replan"
    reasoning: str
    feedback: str = ""

class Draft(TypedDict):
    version: int
    document: str
    word_count: int
    action: str      # 'initial' | 'revision'
    created_at: str

class ResearchState(TypedDict):
    # -- immutable core --
    query:       str
    session_id:  str
    created_at:  str
    doc_id:      str

    # -- slide coordination --
    max_slides:    int
    slide_numbers: List[int]

    # -- circuit breaker counters --
    revision_count: int
    replan_count:   int

    # -- current working artifacts (overwritten each cycle) --
    plan:             Optional[DeliveryPlan]
    draft:            Optional[Draft]
    critique: Optional[CritiqueOutput]
    document_context: str
    source_chunks:    List[Any]

    # -- append-only histories --
    revision_history: Annotated[List[str], operator.add]
    replan_history:   Annotated[List[str], operator.add]

    # -- observability --
    messages: Annotated[List[str], operator.add]
    errors:   Annotated[List[ErrorRecord], operator.add]
