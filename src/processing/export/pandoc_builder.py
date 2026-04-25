from __future__ import annotations

import base64
import re
import tempfile
from pathlib import Path
from typing import List

import pypandoc

from src.logging.logger import AgentLogger
from src.memory.objectstore.provider import ObjectStoreProvider
from src.memory.research.database import ResearchDatabase
from src.memory.research.schema import BulletPoint, ProtoSlide, SlideContent

# Known Type 1 PDF encoding artifacts: raw glyph byte promoted to a U+00xx.
# Some models emit these in proto-slides; Pandoc / pptx-XML are strict about
# control characters, so we repair to Unicode punctuation and strip the rest.
_TYPOGRAPHIC_REPAIRS = {
    "\u0010": "\u2010",  # HYPHEN
    "\u0011": "\u2011",  # NON-BREAKING HYPHEN
    "\u0013": "\u2013",  # EN DASH
    "\u0014": "\u2014",  # EM DASH
    "\u0018": "\u2018",  # LEFT SINGLE QUOTATION MARK
    "\u0019": "\u2019",  # RIGHT SINGLE QUOTATION MARK
    "\u001c": "\u201c",  # LEFT DOUBLE QUOTATION MARK
    "\u001d": "\u201d",  # RIGHT DOUBLE QUOTATION MARK
}
# XML 1.0 allows only #x9 | #xA | #xD | [#x20–#xD7FF] | ...
_XML_ILLEGAL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_xml_text(text: str | None) -> str:
    """Repair Type 1 font encoding artifacts; strip remaining XML 1.0 illegal chars."""
    if not text:
        return text or ""
    for bad, good in _TYPOGRAPHIC_REPAIRS.items():
        text = text.replace(bad, good)
    return _XML_ILLEGAL_RE.sub("", text)


# NOTE: This is the default reference doc for the PandocBuilder.
# Using relative path so export folder can be moved without issue.
# This is here instead of in main.py because the file is used and owned by pandoc_builder, and is not intended for the user to change.
_DEFAULT_REFERENCE_DOC = Path(__file__).parent / "reference.pptx"

_MIME_TO_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
}


def _mime_to_ext(mime_type: str) -> str:
    return _MIME_TO_EXT.get((mime_type or "").lower().strip(), ".png")


def _media_alt_text(db: ResearchDatabase, media_id: str) -> str:
    image = db.get_image(media_id)
    if image is None:
        return ""
    alt = (image.caption or "").replace("\n", " ").replace("]", "").replace('"', "'").strip()
    return sanitize_xml_text(alt[:500])


def _resolve_image(
    media_id: str,
    db: ResearchDatabase,
    object_store: ObjectStoreProvider | None,
    tmp_dir: Path,
) -> Path | None:
    image = db.get_image(media_id)
    if image is None:
        return None
    ext = _mime_to_ext(image.mime_type)

    if image.storage_path and not image.storage_path.startswith(("http://", "https://")):
        p = Path(image.storage_path)
        if p.exists():
            return p

    if image.storage_path and image.storage_path.startswith(("http://", "https://")) and object_store:
        tmp_path = tmp_dir / f"{media_id}_dl{ext}"
        try:
            data = object_store.read(image.storage_path)
            tmp_path.write_bytes(data)
            return tmp_path
        except Exception as exc:
            AgentLogger().log(
                f"[PandocBuilder] Could not read image {media_id} via object store: {exc}",
                level="warning",
            )

    if image.base64_data:
        tmp_path = tmp_dir / f"{media_id}_b64{ext}"
        tmp_path.write_bytes(base64.b64decode(image.base64_data))
        return tmp_path

    AgentLogger().log(
        f"[PandocBuilder] No usable image data for media_id={media_id}; slide will render without image.",
        level="warning",
    )
    return None


def _bullet_lines(content: SlideContent) -> List[str]:
    lines: List[str] = []
    for bullet in content.bullets:
        lines.extend(_render_bullet(bullet))
    return lines


def _split_bullets_for_two_columns(content: SlideContent) -> tuple[List[str], List[str]]:
    """Split top-level bullets into two columns, keeping sub-bullets with parents."""
    midpoint = (len(content.bullets) + 1) // 2
    left_lines: List[str] = []
    right_lines: List[str] = []

    for bullet in content.bullets[:midpoint]:
        left_lines.extend(_render_bullet(bullet))
    for bullet in content.bullets[midpoint:]:
        right_lines.extend(_render_bullet(bullet))

    return left_lines, right_lines


def _render_bullet(bullet: BulletPoint) -> List[str]:
    """Returns a list of Markdown lines for a single BulletPoint."""
    lines = [f"- {sanitize_xml_text(bullet.text)}"]
    for sub in bullet.sub_bullets:
        lines.append(f"  - {sanitize_xml_text(sub)}")
    return lines


def _render_slide(
    proto: ProtoSlide,
    db: ResearchDatabase,
    image_paths: dict[str, Path | None],
) -> str:
    """Converts a single ProtoSlide into a Pandoc Markdown slide block."""
    content: SlideContent = proto.content
    parts: List[str] = []

    parts.append(f"# {sanitize_xml_text(content.title)}")
    parts.append("")

    mid = content.media_id
    img_path = image_paths.get(mid) if mid else None
    layout = content.layout
    alt = _media_alt_text(db, mid) if mid else ""
    img_uri = img_path.resolve().as_uri() if img_path else ""
    img_md = f'![]({img_uri}){{alt="{alt}"}}' if img_path and img_uri else ""

    bullet_lines = _bullet_lines(content)

    if img_path and img_uri and layout == "media_left":
        parts.append(":::: {.columns}")
        parts.append("::: {.column width=\"40%\"}")
        parts.append(img_md)
        parts.append(":::")
        parts.append("::: {.column width=\"60%\"}")
        parts.extend(bullet_lines)
        parts.append(":::")
        parts.append("::::")
    elif img_path and img_uri and layout == "media_right":
        parts.append(":::: {.columns}")
        parts.append("::: {.column width=\"60%\"}")
        parts.extend(bullet_lines)
        parts.append(":::")
        parts.append("::: {.column width=\"40%\"}")
        parts.append(img_md)
        parts.append(":::")
        parts.append("::::")
    elif img_path and img_uri and layout == "media_top":
        parts.append(img_md)
        parts.append("")
        parts.extend(bullet_lines)
    elif img_path and img_uri and layout == "media_bottom":
        parts.extend(bullet_lines)
        parts.append("")
        parts.append(img_md)
    elif layout == "two_column":
        left_lines, right_lines = _split_bullets_for_two_columns(content)
        parts.append(":::: {.columns}")
        parts.append("::: {.column width=\"50%\"}")
        parts.extend(left_lines)
        parts.append(":::")
        parts.append("::: {.column width=\"50%\"}")
        parts.extend(right_lines)
        parts.append(":::")
        parts.append("::::")
    elif img_path and img_uri and layout == "media_center":
        # Use side gutters so the image appears centered under the title in pptx.
        parts.append(":::: {.columns}")
        parts.append("::: {.column width=\"20%\"}")
        parts.append(":::")
        parts.append("::: {.column width=\"60%\"}")
        parts.append(img_md)
        parts.append(":::")
        parts.append("::: {.column width=\"20%\"}")
        parts.append(":::")
        parts.append("::::")
    else:
        parts.extend(bullet_lines)

    if content.speaker_notes:
        parts.append("")
        parts.append("::: notes")
        parts.append(sanitize_xml_text(content.speaker_notes))
        parts.append(":::")

    return "\n".join(parts)


class PandocBuilder:
    """
    Reads all ProtoSlide rows from a ResearchDatabase, renders them as Pandoc
    Markdown (with LaTeX math support via the tex_math_dollars extension),
    and converts the result to a .pptx file using pypandoc.

    The opening title slide is generated from the ``title`` and ``subtitle``
    constructor arguments as YAML front matter; it is never stored in the DB.

    Bullet text may contain Markdown formatting and LaTeX math:
      - Inline math:  $E = mc^2$
      - Display math: $$\\text{Attention}(Q,K,V) = \\text{softmax}(...)V$$

    Pandoc converts LaTeX math to OMML, which PowerPoint renders natively.

    Usage::

        with ResearchDatabase() as db:
            PandocBuilder(
                output_path=Path("output.pptx"),
                db=db,
                title="My Talk",
                subtitle="An optional subtitle",
                object_store=object_store,
            ).build()
    """

    # -implicit_figures: without this, a lone ![alt](url) becomes a figure and Pandoc
    # repeats alt text as a visible caption in pptx; we only want alt for accessibility.
    _PANDOC_FORMAT = "markdown+tex_math_dollars-implicit_figures"

    def __init__(
        self,
        output_path: Path,
        db: ResearchDatabase,
        title: str = "",
        subtitle: str = "",
        object_store: ObjectStoreProvider | None = None,
        reference_doc: Path | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self._db = db
        self._title = title
        self._subtitle = subtitle
        self._object_store = object_store
        if reference_doc is None and _DEFAULT_REFERENCE_DOC.exists():
            self._reference_doc = _DEFAULT_REFERENCE_DOC
        else:
            self._reference_doc = reference_doc

    def build(self) -> Path:
        """
        Loads all proto-slides from research.db in slide-number order, renders them
        to Pandoc Markdown (prepending a YAML title slide from constructor metadata),
        and writes the presentation to self.output_path.
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

        with tempfile.TemporaryDirectory(prefix="mars_img_") as tmp:
            tmp_dir = Path(tmp)
            image_paths: dict[str, Path | None] = {}
            for slide in slides:
                mid = slide.content.media_id
                if mid and mid not in image_paths:
                    image_paths[mid] = _resolve_image(mid, self._db, self._object_store, tmp_dir)

            markdown = self._render_markdown(slides, image_paths)

            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            extra_args = ["--standalone"]
            if self._reference_doc:
                extra_args.extend(["--reference-doc", str(self._reference_doc)])

            try:
                pypandoc.convert_text(
                    markdown,
                    "pptx",
                    format=self._PANDOC_FORMAT,
                    outputfile=str(self.output_path),
                    extra_args=extra_args,
                )
            except RuntimeError as exc:
                if not self._reference_doc:
                    raise

                AgentLogger().log(
                    f"[PandocBuilder] Reference doc failed ({self._reference_doc}): {exc}. "
                    "Retrying without template.",
                    level="warning",
                )
                pypandoc.convert_text(
                    markdown,
                    "pptx",
                    format=self._PANDOC_FORMAT,
                    outputfile=str(self.output_path),
                    extra_args=["--standalone"],
                )

        return self.output_path.resolve()

    def _render_markdown(self, slides: List[ProtoSlide], image_paths: dict[str, Path | None]) -> str:
        """Renders the full deck as a Pandoc Markdown string."""
        yaml_header = ""
        if self._title:
            title = sanitize_xml_text(self._title).replace('"', '\\"')
            yaml_lines = ["---", f'title: "{title}"']
            if self._subtitle:
                subtitle = sanitize_xml_text(self._subtitle).replace('"', '\\"')
                yaml_lines.append(f'subtitle: "{subtitle}"')
            yaml_lines.append("---")
            yaml_header = "\n".join(yaml_lines)

        blocks = [_render_slide(s, self._db, image_paths) for s in slides]
        body = "\n\n---\n\n".join(blocks)

        if yaml_header:
            return f"{yaml_header}\n\n{body}" if body else yaml_header
        return body
