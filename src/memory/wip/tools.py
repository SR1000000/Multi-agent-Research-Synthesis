from typing import Optional, List
from src.memory.wip.schema import ProtoSlide
from src.memory.wip.database import WIPDatabase

# Global singleton or default instance
# In a full LangGraph setup, you might inject this via state, but this works well for simple standalone tools.
_wip_db = WIPDatabase()

def read_proto_slide(slide_number: int) -> Optional[ProtoSlide]:
    """
    Reads a work-in-progress proto-slide by its slide number.

    Args:
        slide_number (int): The integer slide number to retrieve.

    Returns:
        Optional[ProtoSlide]: The ProtoSlide instance if found, or None if it does not exist.
    """
    return _wip_db.load_slide(slide_number)

def write_proto_slide(slide_number: int, slide: ProtoSlide) -> bool:
    """
    Writes or overwrites a proto-slide to the work-in-progress database.

    Args:
        slide_number (int): The exact slide number this slide will occupy (must match slide.slide_number).
        slide (ProtoSlide): The structured proto-slide containing content and chunk references.

    Returns:
        bool: True if the write was successful, False otherwise.
    """
    if slide.slide_number != slide_number:
        # Prevent mismatches between explicitly stated number and the slide object
        raise ValueError(f"Slide number mismatch: argument {slide_number} != object {slide.slide_number}")
    
    try:
        _wip_db.save_slide(slide)
        return True
    except Exception as e:
        _wip_db._logger.log(f"[WIP Tools] Failed to write slide {slide_number}: {e}")
        return False
