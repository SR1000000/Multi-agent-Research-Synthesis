import json
from typing import TypedDict, List
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.state import ResearchState
from src.agents.base import BaseLLMAgent
from src.memory.wip.schema import ProtoSlide, SlideContent
from src.memory.research.database import ResearchDatabase
from src.memory.wip.database import WIPDatabase
from src.logging.logger import AgentLogger

class SlideGenerationOutput(BaseModel):
    slides: List[SlideContent] = Field(description="The parsed slides synthesized from the text chunks")

class DispatchState(TypedDict):
    """The state sent over LangGraph's Send API."""
    chunk_ids: List[str]
    slide_number_range: List[int] # [start, end] inclusive
    session_id: str

class ResearchToSlideAgent(BaseLLMAgent):
    def __init__(self):
        super().__init__('research_to_slide')
        self._logger = AgentLogger()

    def run(self, state: DispatchState) -> Command:
        chunk_ids = state.get("chunk_ids", [])
        slide_range = state.get("slide_number_range", [1, 1])
        
        start_idx, end_idx = slide_range[0], slide_range[-1]
        max_slides_allowed = max(end_idx - start_idx + 1, 1)

        if not chunk_ids:
            self._logger.log(f"[ResearchToSlide] No chunks to process for slides {start_idx}-{end_idx}")
            return Command(update={"messages": [f"[ResearchToSlide] Skipped slides {start_idx}-{end_idx} (no chunks)"]})

        # 1. Fetch chunks from research.db
        # 1. Fetch chunks from research.db
        with ResearchDatabase() as research_db:
            placeholders = ','.join(['?'] * len(chunk_ids))
            query = f"SELECT id, text, contextualized_text FROM text_chunks WHERE id IN ({placeholders})"
            rows = research_db.connection.execute(query, chunk_ids).fetchall()
        
        chunk_texts = []
        for row in rows:
            text = row["contextualized_text"] if row["contextualized_text"] else row["text"]
            chunk_texts.append(f"--- Chunk ID: {row['id']} ---\n{text}")
            
        combined_text = "\n\n".join(chunk_texts)
        
        user_prompt = (
            f"Please create up to {max_slides_allowed} slides based on the following text chunks.\n\n"
            #f"If the text chunks contain references to images, charts, or tables (e.g. placeholders or IDs), "
            #f"use an appropriate layout (e.g., 'media_left' or 'media_right') and assign the extracted ID "
            #f"to the media_id field of the slide.\n\n"
            f"SOURCE MATERIAL:\n{combined_text}\n\n"
            f"Remember, you can generate AT MOST {max_slides_allowed} slides."
        )

        turns = [{"role": "user", "content": user_prompt}]
        
        # 2. Invoke LLM
        output_schema = SlideGenerationOutput
        result: SlideGenerationOutput = self._call(turns, schema=output_schema)

        # 3. Save to wip.db
        wip_db = WIPDatabase()
        
        saved_count = 0
        current_slide_num = start_idx
        for slide_content in result.slides:
            if current_slide_num > end_idx:
                break # enforce strict bounds
                
            proto_slide = ProtoSlide(
                slide_number=current_slide_num,
                content=slide_content,
                chunk_references=chunk_ids
            )
            wip_db.save_slide(proto_slide)
            saved_count += 1
            current_slide_num += 1
            
        msg = f"[ResearchToSlide] Generated {saved_count} slides for range {start_idx}-{end_idx}"
        self._logger.log(msg)
        
        return Command(update={"messages": [msg]})

def research_to_slide_node(state: DispatchState) -> Command:
    return ResearchToSlideAgent().run(state)
