# Presentation Export & Slide Generation

This document describes the architecture for converting intermediate slide representations ("Proto-slides") into final presentation formats. The active export engine is `PandocBuilder`, which converts a Pandoc Markdown representation of each slide (with native LaTeX math support) into a `.pptx` file. The original `python-pptx` implementation is retained in the codebase for archival purposes but is no longer invoked.

## The Generation Pipeline

Slide generation is a two-phase process:

1.  **Synthesis Phase**: The Multi-agent Graph (`src.graph`) analyzes research data and populates the `proto_slides` table in `research.db` with `ProtoSlide` objects.
2.  **Export Phase**: `PandocBuilder` reads `research.db`, renders each slide as Pandoc Markdown, and converts the full deck to `.pptx` via the Pandoc binary (bundled with the `pypandoc_binary` wheel).

## Data Contract: Proto-Slides

The bridge between synthesis and export is the `ProtoSlide` schema defined in `src.memory.wip.schema`. Any export builder must be able to handle this data structure:

*   **`SlideContent`**: Contains the semantic parts of a slide.
    *   `title`: The main heading.
    *   `bullets`: A list of `BulletPoint` objects.
    *   `layout`: A literal (e.g., `title_and_body`, `two_column`, `big_number`, `quote`) that hints at how the slide should be styled.
    *   `speaker_notes`: Narrative text for the presenter.
*   **`BulletPoint`**:
    *   `text`: The primary bullet text. Supports Markdown formatting and LaTeX math (see [Markdown & LaTeX in Bullet Text](#markdown--latex-in-bullet-text) below).
    *   `sub_bullets`: A list of plain strings (may also contain LaTeX math).
    *   `bold_phrases`: Substrings within `text` that should be emphasized in bold.
    *   `content_type`: Semantic metadata (e.g., `statistic`, `insight`) for potential conditional styling.

## Active Implementation: `PandocBuilder`

**File**: `src/processing/export/pandoc_builder.py`  
**Dependency**: `pypandoc_binary>=1.17` (Pandoc binary bundled in the wheel — no separate system install required)

### How It Works

`PandocBuilder._render_markdown(slides)` converts the ordered list of `ProtoSlide` objects into a single Pandoc Markdown string. Each slide block follows this structure:

```markdown
# Slide Title

- Bullet one with **bold emphasis** and inline math $O(n^2)$
  - Sub-bullet with supporting detail
- Core result:
  - $$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

::: notes
Speaker notes in a conversational tone...
:::

---
```

Key rendering details:

| Element | Rendering |
| :--- | :--- |
| Slide title | Level-1 Markdown heading (`# Title`) |
| Bullets | `-` list items |
| Sub-bullets | Indented two spaces (`  - sub`) |
| Speaker notes | Pandoc fenced div (`::: notes` / `:::`) |
| Slide separator | `---` horizontal rule |

### Pandoc Conversion

```python
pypandoc.convert_text(
    markdown,
    "pptx",
    format="markdown+tex_math_dollars",
    outputfile=str(output_path),
    extra_args=["--standalone"],
)
```

The `tex_math_dollars` Markdown extension enables `$...$` (inline) and `$$...$$` (display) math parsing. Pandoc converts LaTeX math to **OMML** (Office Math Markup Language), which PowerPoint renders natively without any plugins.

### Markdown & LaTeX in Bullet Text

The `BulletPoint.text` field and `sub_bullets` strings support Markdown and LaTeX math. The slide generation agents (`slide_writer`) are instructed to use these when presenting important equations:

*   **Inline math** — for formulas referenced within a sentence:
    ```
    The model achieves $O(n^2)$ complexity with respect to sequence length.
    ```
*   **Display math** — for landmark equations; placed as the sole content of a `sub_bullet` so it renders on its own line:
    ```
    $$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
    ```
*   **Bold Markdown** — `**text**` may be used directly in `text` for emphasis alongside or instead of `bold_phrases`.

Equations should only be included when they are central to the slide's `key_message` or represent a landmark result from the source material.

---

## Archival Implementation: `PptxBuilder`

**File**: `src/processing/export/pptx_builder.py` *(not invoked — retained for reference)*  
**Dependency**: `python-pptx>=1.0.0`

This was the original export engine. It generates `.pptx` files directly using the `python-pptx` library with no intermediate Markdown step. It does not support LaTeX math rendering.

---

## Future Extensibility: The Abstract Builder

To support additional export formats, a Strategy pattern can be used with a shared abstract interface:

```python
class PresentationBuilder(ABC):
    @abstractmethod
    def build(self) -> Path:
        """Entry point to process all slides and save the file."""
        pass

    @abstractmethod
    def _render_markdown(self, slides: list) -> str:
        """Internal logic to render all proto-slides into an intermediate format."""
        pass
```

Both `PandocBuilder` and the archival `PptxBuilder` conform to the `build()` contract. A new backend (e.g., an HTML/reveal.js exporter) would implement the same interface and be swapped in at the call site in `main.py`.

## Developer Notes

### Adding New Layouts

For `PandocBuilder`, layout literals in `SlideContent` are not directly mapped — Pandoc's default PPTX template is used. To apply custom slide layouts:

1.  Provide a reference `.pptx` template via `--reference-doc=template.pptx` in `extra_args`.
2.  Pandoc will use the layout from the reference doc whose name matches the slide level.

For the archival `PptxBuilder`:

1.  Update the `layout` Literal in `src.memory.wip.schema.SlideContent`.
2.  Update `_LAYOUT_MAP` in `PptxBuilder` to map that literal to a template layout.
3.  Add any custom logic in `_set_body` or `_write_run` if the layout requires special styling.
