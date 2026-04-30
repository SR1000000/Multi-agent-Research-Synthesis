# Presentation Export & Slide Generation

This document describes the architecture for converting intermediate slide representations ("Proto-slides") into final presentation formats. The active export engine is `PandocBuilder`, which converts a Pandoc Markdown representation of each slide (with native LaTeX math support) into a `.pptx` file. The original `python-pptx` implementation is retained in the codebase for archival purposes but is no longer invoked.

## The Generation Pipeline

Slide generation is a two-phase process:

1.  **Synthesis Phase**: The multi-agent graph analyzes research data and populates the `proto_slides` table in `research.db` with the latest slide drafts.
2.  **Export Phase**: After the graph accepts the deck, or after an explicit review bypass, `PandocBuilder` reads `research.db`, renders each slide as Pandoc Markdown, and converts the full deck to `.pptx` via the Pandoc binary (bundled with the `pypandoc_binary` wheel).

If a first-draft slide group exhausts its retries, export can still proceed after acceptance, but the run reports a partial-deck warning.

## Data Contract: Proto-Slides

The bridge between synthesis and export is the proto-slide data stored in the research database. Any export builder must be able to handle this data structure:

*   **`SlideContent`**: Contains the semantic parts of a slide.
    *   `title`: The main heading.
    *   `bullets`: A list of `BulletPoint` objects.
    *   `layout`: A literal (e.g., `title_and_body`, `two_column`, `media_center`) that hints at how the slide should be styled.
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
    format="markdown+tex_math_dollars-implicit_figures",
    outputfile=str(output_path),
    extra_args=extra_args,
)
```

The `tex_math_dollars` Markdown extension enables `$...$` (inline) and `$$...$$` (display) math parsing. Pandoc converts LaTeX math to **OMML** (Office Math Markup Language), which PowerPoint renders natively without any plugins. The `-implicit_figures` flag prevents Pandoc from treating isolated images as figures, avoiding repeated alt text as visible captions.

#### Reference Doc Behaviour
A **bundled default** template (`reference.pptx`) lives at `src/processing/export/reference.pptx` and is used automatically via the `--reference-doc` flag if no custom reference doc is provided. 
If a reference doc is supplied but Pandoc fails during conversion (e.g., due to an incompatible template), `PandocBuilder` automatically retries the conversion **without the template** and logs a warning, instead of crashing the export pipeline.

### Markdown & LaTeX in Bullet Text

The `BulletPoint.text` field and `sub_bullets` strings support Markdown and LaTeX math. The slide generation agents are instructed to use these when presenting important equations:

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

### Image Embedding

Slides can include an image via `SlideContent.media_id`, which references an image ID in `research.db`.
`PandocBuilder` resolves the image at build time, checking three sources in order: local `storage_path`, object store download (for `http(s)://` paths), then inline `base64_data`. Images are written to a temporary directory for the duration of the Pandoc conversion and cleaned up automatically.

The `layout` field on `SlideContent` drives the Pandoc column/div structure used to position the image:

| Layout | Rendering |
|---|---|
| `media_left` | Image (40%) left column, bullets (60%) right column |
| `media_right` | Bullets (60%) left, image (40%) right |
| `media_top` | Image above bullet list |
| `media_bottom` | Bullet list above image |
| `media_center` | Image centered using 20%/60%/20% gutter columns |
| `two_column` | No image; bullets split at midpoint into two equal columns |
| `title_and_body` | No image; standard bullet list (default) |

A `sanitize_xml_text()` pass repairs Type 1 PDF encoding artifacts (e.g. raw glyph bytes promoted to U+00xx) and strips XML 1.0 illegal characters from all text before it reaches Pandoc.

---

## Archival Implementation: `PptxBuilder`

**File**: `src/processing/export/pptx_builder.py` *(not invoked — retained for reference)*  
**Dependency**: `python-pptx>=1.0.0`

This was the original export engine. It generates `.pptx` files directly using the `python-pptx` library with no intermediate Markdown step. It does not support LaTeX math rendering.

Future work: implement python-pptx as an additional layer after Pandoc to add autofit capabilities to all text boxes in the presentation, fixing the issues with too much text overflowing the slides.  Pandoc is unable to do this.

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

For `PandocBuilder`, while layouts map directly to Pandoc columns for positioning, you can further customize styles:

1.  Provide a reference `.pptx` template via `--reference-doc=template.pptx` in `extra_args`.
2.  Pandoc will use the layout from the reference doc whose name matches the slide level.