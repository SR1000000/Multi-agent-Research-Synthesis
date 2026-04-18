from pathlib import Path
from typing import List

import pypandoc

from src.memory.research.database import ResearchDatabase
from src.memory.research.schema import BulletPoint, ProtoSlide, SlideContent


def _render_bullet(bullet: BulletPoint) -> List[str]:
    """Returns a list of Markdown lines for a single BulletPoint."""
    lines = [f"- {bullet.text}"]
    for sub in bullet.sub_bullets:
        lines.append(f"  - {sub}")
    return lines


def _render_slide(proto: ProtoSlide) -> str:
    """Converts a single ProtoSlide into a Pandoc Markdown slide block."""
    content: SlideContent = proto.content
    parts: List[str] = []

    parts.append(f"# {content.title}")
    parts.append("")
    if content.subtitle:
        parts.append(f"## {content.subtitle}")
        parts.append("")

    for bullet in content.bullets:
        parts.extend(_render_bullet(bullet))

    if content.speaker_notes:
        parts.append("")
        parts.append("::: notes")
        parts.append(content.speaker_notes)
        parts.append(":::")

    return "\n".join(parts)


class PandocBuilder:
    """
    Reads all ProtoSlide rows from a ResearchDatabase, renders them as Pandoc
    Markdown (with LaTeX math support via the tex_math_dollars extension),
    and converts the result to a .pptx file using pypandoc.

    Bullet text may contain Markdown formatting and LaTeX math:
      - Inline math:  $E = mc^2$
      - Display math: $$\\text{Attention}(Q,K,V) = \\text{softmax}(...)V$$

    Pandoc converts LaTeX math to OMML, which PowerPoint renders natively.

    Usage::

        with ResearchDatabase() as db:
            PandocBuilder(output_path=Path("output.pptx"), db=db).build()
    """

    _PANDOC_FORMAT = "markdown+tex_math_dollars"

    def __init__(self, output_path: Path, db: ResearchDatabase) -> None:
        self.output_path = Path(output_path)
        self._db = db

    def build(self) -> Path:
        """
        Loads all proto-slides from research.db in slide-number order, renders them
        to Pandoc Markdown, and writes the presentation to self.output_path.
        Returns the resolved path.

        Raises:
            ValueError: if no slides exist in the database.
        """
        slide_numbers = self._db.list_slide_numbers()
        if not slide_numbers:
            raise ValueError(
                "PandocBuilder: no proto-slides found in research.db. "
                "Run the slide-generation graph first."
            )

        slides: List[ProtoSlide] = [
            self._db.load_slide(n) for n in slide_numbers
        ]
        slides = [s for s in slides if s is not None]

        markdown = self._render_markdown(slides)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        pypandoc.convert_text(
            markdown,
            "pptx",
            format=self._PANDOC_FORMAT,
            outputfile=str(self.output_path),
            extra_args=["--standalone"],
        )

        return self.output_path.resolve()

    def _render_markdown(self, slides: List[ProtoSlide]) -> str:
        """Renders the full deck as a Pandoc Markdown string."""
        if not slides:
            return ""

        yaml_header = ""
        if slides[0].content.layout == "title_slide":
            title_slide = slides[0]
            # Escape quotes for YAML strings
            title = title_slide.content.title.replace('"', '\\"')
            yaml_lines = ["---", f'title: "{title}"']
            
            if title_slide.content.subtitle:
                subtitle = title_slide.content.subtitle.replace('"', '\\"')
                yaml_lines.append(f'subtitle: "{subtitle}"')
            yaml_lines.append("---")
            
            if title_slide.content.speaker_notes:
                yaml_lines.append("")
                yaml_lines.append("::: notes")
                yaml_lines.append(title_slide.content.speaker_notes)
                yaml_lines.append(":::")
                
            yaml_header = "\n".join(yaml_lines)
            slides_to_render = slides[1:]
        else:
            slides_to_render = slides

        blocks = [_render_slide(s) for s in slides_to_render]
        body = "\n\n---\n\n".join(blocks)

        if yaml_header:
            return f"{yaml_header}\n\n{body}" if body else yaml_header
        return body
