# Presentation Export & Slide Generation

This document describes the architecture for converting intermediate slide representations ("Proto-slides") into final presentation formats. While currently implemented using `python-pptx`, the system is designed to be modular to support alternative export engines (like Pandoc or HTML-based decks).

## The Generation Pipeline

Slide generation is a two-phase process:

1.  **Synthesis Phase**: The Multi-agent Graph (`src.graph`) analyzes research data and populates `wip.db` with `ProtoSlide` objects.
2.  **Export Phase**: An export builder (e.g., `PptxBuilder`) reads the `wip.db` and renders the final file.

## Data Contract: Proto-Slides

The bridge between synthesis and export is the `ProtoSlide` schema defined in `src.memory.wip.schema`. Any export builder must be able to handle this data structure:

*   **`SlideContent`**: Contains the semantic parts of a slide.
    *   `title`: The main heading.
    *   `bullets`: A list of `BulletPoint` objects.
    *   `layout`: A literal (e.g., `title_and_body`, `two_column`, `big_number`, `quote`) that hints at how the slide should be styled.
    *   `speaker_notes`: Narrative text for the presenter.
*   **`BulletPoint`**:
    *   `text`: The primary bullet text.
    *   `sub_bullets`: A list of plain strings.
    *   `bold_phrases`: Substrings within `text` that should be emphasized.
    *   `content_type`: Semantic metadata (e.g., `statistic`, `insight`) for potential conditional styling.

## Current Implementation: `PptxBuilder`

The default builder uses the `python-pptx` library to generate native PowerPoint files.

### Layout Mapping
The builder maps the generic `layout` literals from `SlideContent` to specific indices or names in a PowerPoint template:

| Literal | PPTX Layout Mapping | Special Behavior |
| :--- | :--- | :--- |
| `title_slide` | "Title Slide" (Index 0) | Centers text, uses title placeholder |
| `title_and_body` | "Title and Content" (Index 1) | Standard bullet list |
| `two_column` | "Two Content" (Index 3) | Splits bullets across columns |
| `big_number` | "Title and Content" | Increases font size of the first run |
| `quote` | "Title and Content" | Italics and specific font sizing |
| `media_left` | "Two Content" | Reserves left side for `media_id` placement |

### Styling Logic
-   **Bold Emphasis**: The builder performs substring matching using `bold_phrases` to apply `run.font.bold = True` to specific segments of bullet text.
-   **Speaker Notes**: Content is written directly to the `notes_slide` of each generated slide.
-   **Fallbacks**: If a template is missing a named layout, the builder falls back to "Title and Content" (Index 1) to prevent crashes.

## Future Extensibility: The Abstract Builder

To support the requirement of replacing `python-pptx` or adding new formats (like Pandoc-driven Markdown-to-PPTX), we utilize a Strategy pattern.

### Planned Interface
```python
class PresentationBuilder(ABC):
    @abstractmethod
    def build(self) -> Path:
        """Entry point to process all slides and save the file."""
        pass

    @abstractmethod
    def _add_slide(self, prs: Any, proto: ProtoSlide) -> None:
        """Internal logic to render a single proto-slide."""
        pass
```

## Utilization for Developers

### Adding New Layouts
1.  Update the `layout` Literal in `src.memory.wip.schema.SlideContent`.
2.  Update the `_LAYOUT_MAP` in `PptxBuilder` to map that literal to a template layout.
3.  Add any custom logic in `_set_body` or `_write_run` if the layout requires special styling (like the `big_number` mode).
