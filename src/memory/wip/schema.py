from typing import List, Literal, Optional
from pydantic import AliasChoices, BaseModel, Field, field_validator

# LLMs frequently output "content" instead of "text" for bullet fields.
# AliasChoices makes the schema accept either key without a retried call.


class BulletPoint(BaseModel):
    text: str = Field(
        description=(
            'The bullet point text. Supports Markdown formatting and LaTeX math '
            '(e.g., inline `$E=mc^2$` or display `$$\\frac{a}{b}$$`). '
            'Field name is "text" — do NOT use "content".'
        ),
        validation_alias=AliasChoices("text", "content"),
    )
    sub_bullets: List[str] = Field(
        default_factory=list,
        description='Optional sub-bullet points as plain strings (NOT objects). Example: ["Detail A", "Detail B"]',
    )
    content_type: Literal["insight", "evidence", "statistic", "example", "caveat"] = Field(default="insight", description="Semantic type of this bullet point")

    @field_validator("sub_bullets", mode="before")
    @classmethod
    def coerce_sub_bullets(cls, v: list) -> list[str]:
        if not isinstance(v, list):
            return v
        return [
            (item.get("text") or item.get("content") or str(item))
            if isinstance(item, dict)
            else str(item)
            for item in v
        ]


class SlideContent(BaseModel):
    title: str = Field(description="Punchy, active heading for the slide (e.g. 'Accuracy Jumps 40%' not 'Accuracy Results')")
    key_message: str = Field(description="One sentence capturing what the audience should understand after this slide")
    bullets: List[BulletPoint] = Field(description="3-5 structured bullet points for the slide body")
    speaker_notes: str = Field(description="Speaker notes in a professional, conversational tone — include context and nuance too detailed for the slide itself")
    media_id: Optional[str] = Field(default=None, description="Optional ID of an image or chart from research.db")
    layout: Literal["title_slide", "title_and_body", "two_column", "big_number", "quote", "media_left", "media_right"] = Field(
        default="title_and_body",
        description="Slide layout: 'title_slide' for openers/dividers, 'title_and_body' for standard bullet slides, 'two_column' for comparisons, 'big_number' for stat callouts, 'quote' for direct quotations, 'media_left'/'media_right' when a figure or chart is the focus"
    )
    narrative_role: Literal["hook", "context", "evidence", "insight", "transition", "conclusion"] = Field(
        default="evidence",
        description="Role this slide plays in the deck's narrative arc: 'hook' grabs attention, 'context' provides background, 'evidence' presents data, 'insight' delivers the key takeaway, 'transition' bridges sections, 'conclusion' wraps up"
    )

class ProtoSlide(BaseModel):
    slide_number: int = Field(description="The slide number")
    content: SlideContent = Field(description="The structured content of the slide")
    chunk_references: List[str] = Field(description="List of exact text chunk IDs from research.db that this slide covers")

CREATE_PROTO_SLIDES_TABLE = """
CREATE TABLE IF NOT EXISTS proto_slides (
    slide_number INTEGER PRIMARY KEY,
    content TEXT NOT NULL,
    chunk_references TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""
