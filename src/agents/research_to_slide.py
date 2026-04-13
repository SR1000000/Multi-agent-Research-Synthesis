from typing import TypedDict, List
from langgraph.types import Command
from pydantic import BaseModel, Field

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


def _slide_range_log_label(state: DispatchState) -> str:
    r = state.get("slide_number_range", [1, 1])
    a, b = r[0], r[-1]
    return f"Research_to_slide[slides {a}-{b}]"


class ResearchToSlideAgent(BaseLLMAgent):
    def __init__(self, *, log_display: str | None = None):
        super().__init__("research_to_slide", log_display=log_display)
        self._logger = AgentLogger()

    def run(self, state: DispatchState) -> Command:
        chunk_ids = state.get("chunk_ids", [])
        slide_range = state.get("slide_number_range", [1, 1])
        
        start_idx, end_idx = slide_range[0], slide_range[-1]
        max_slides_allowed = max(end_idx - start_idx + 1, 1)
        tag = self._log_display

        if not chunk_ids:
            self._logger.log(f"[{tag}] No chunks to process")
            return Command(update={"messages": [f"[{tag}] Skipped (no chunks)"]})

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

        is_first_range = start_idx == 1
        is_last_range = end_idx == start_idx + max_slides_allowed - 1

        narrative_hint = ""
        if is_first_range:
            narrative_hint = (
                "This is the OPENING range of the deck. "
                "Assign 'hook' or 'context' as the narrative_role for your first slide.\n"
            )
        elif is_last_range:
            narrative_hint = (
                "This is the CLOSING range of the deck. "
                "Assign 'conclusion' as the narrative_role for your final slide.\n"
            )

        user_prompt = (
            f"Please create up to {max_slides_allowed} slides (slide numbers {start_idx}–{end_idx}) "
            f"based on the following text chunks.\n"
            f"{narrative_hint}\n"
            f"For each slide:\n"
            f"- Write a `key_message`: one sentence capturing what the audience should understand.\n"
            f"- Assign a `narrative_role` that reflects the slide's function in the argument "
            f"(hook | context | evidence | insight | transition | conclusion).\n"
            f"- Choose the `layout` that best serves the content "
            f"(title_and_body | big_number | quote | two_column | media_left | media_right | title_slide).\n\n"
            f"SOURCE MATERIAL:\n{combined_text}\n\n"
            f"Remember, you can generate AT MOST {max_slides_allowed} slides."
        )

        turns = [{"role": "user", "content": user_prompt}]
        
        # 2. Invoke LLM
        output_schema = SlideGenerationOutput
        result: SlideGenerationOutput = self._call(turns, schema=output_schema, model="slides")

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
            
        msg = f"[{tag}] Generated {saved_count} slide(s)"
        self._logger.log(msg)
        
        return Command(update={"messages": [msg]})

def research_to_slide_node(state: DispatchState) -> Command:
    return ResearchToSlideAgent(log_display=_slide_range_log_label(state)).run(state)
