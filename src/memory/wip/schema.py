from typing import List, Optional
from pydantic import BaseModel, Field

class SlideContent(BaseModel):
    title: str = Field(description="The title of the slide")
    bullet_points: List[str] = Field(description="Main bullet points of the slide")
    speaker_notes: str = Field(description="Speaker notes for this slide")
    media_id: Optional[str] = Field(default=None, description="Optional ID of an image or chart from research.db")
    layout: Optional[str] = Field(default="text_only", description="Layout of the slide (e.g., 'text_only', 'media_left', 'media_right')")

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
