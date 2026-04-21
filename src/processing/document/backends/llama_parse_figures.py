"""
Figure-anchor normalization for LlamaParse: map markdown placeholders and geometry
from images_content_metadata to ExtractedImage rows and [[img:id]] rewrite keys.
"""
from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..schema import ExtractedImage


def _attr(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _serialize_bbox(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return {k: float(v) if isinstance(v, (int, float)) else v for k, v in raw.items()}
    if hasattr(raw, "model_dump"):
        d = raw.model_dump()
        return _serialize_bbox(d) if isinstance(d, dict) else None
    if hasattr(raw, "dict"):
        d = raw.dict()
        return _serialize_bbox(d) if isinstance(d, dict) else None
    return None


def _coerce_bbox_for_storage(raw: Any) -> dict[str, Any] | None:
    """Normalize LlamaParse bbox objects to a JSON-serializable dict."""
    d = _serialize_bbox(raw)
    return d


def _rect_from_bbox_dict(b: Any) -> tuple[float, float, float, float] | None:
    if b is None:
        return None
    if not isinstance(b, dict):
        return None
    try:
        x = float(b.get("x", 0))
        y = float(b.get("y", 0))
        w = float(b.get("w", 0))
        h = float(b.get("h", 0))
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


def _union_bbox_segments(segments: list[Any]) -> tuple[float, float, float, float] | None:
    rects: list[tuple[float, float, float, float]] = []
    for seg in segments:
        r = _rect_from_bbox_dict(seg if isinstance(seg, dict) else _serialize_bbox(seg))
        if r:
            rects.append(r)
    if not rects:
        return None
    x0 = min(r[0] for r in rects)
    y0 = min(r[1] for r in rects)
    x1 = max(r[0] + r[2] for r in rects)
    y1 = max(r[1] + r[3] for r in rects)
    return (x0, y0, x1 - x0, y1 - y0)


def _union_item_bbox(item: Any) -> tuple[float, float, float, float] | None:
    segs = _attr(item, "bbox") or []
    if not segs:
        return None
    return _union_bbox_segments(list(segs))


def _intersection_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    return iw * ih


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    inter = _intersection_area(a, b)
    if inter <= 0:
        return 0.0
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


def _rect_center(r: tuple[float, float, float, float]) -> tuple[float, float]:
    return (r[0] + r[2] / 2, r[1] + r[3] / 2)


def _center_dist2(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ca, cb = _rect_center(a), _rect_center(b)
    dx, dy = ca[0] - cb[0], ca[1] - cb[1]
    return dx * dx + dy * dy


def _coverage_of_entry_inside_anchor(
    anchor: tuple[float, float, float, float], entry: tuple[float, float, float, float]
) -> float:
    inter = _intersection_area(anchor, entry)
    ea = entry[2] * entry[3]
    return inter / ea if ea > 0 else 0.0

# "Figure 1", "FIGURE 2", "Fig. 3", "FIG. 4" (optional period after abbreviated Fig)
_RE_FIGURE = re.compile(r"^((?:Figure|Fig\.?)\s*(\d+))\b", re.IGNORECASE)

def _parse_figure_label(text: str) -> tuple[str | None, int | None]:
    if not text or not text.strip():
        return None, None
    m = _RE_FIGURE.search(text.strip())
    if not m:
        return None, None
    label = m.group(1).strip()
    try:
        num = int(m.group(2))
    except ValueError:
        num = None
    return label, num


_IMAGE_REF_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_HTML_TABLE_PATTERN = re.compile(r"(<table[\s\S]*?</table>)", re.IGNORECASE | re.DOTALL)

# Geometry heuristics (PDF points, ~1pt ≈ 1/72 inch)
_CAPTION_NEAREST_MAX_DIST_PX = 300.0
"""Reject caption-labeled items whose center is farther than this from the anchor center."""

_ASSIGN_REJECT_ABOVE_ANCHOR_GAP_PX = 100.0
"""If a layout crop's bottom is this many px above the anchor top, skip distance-based match."""

_RESCUE_CAPTION_MAX_GAP_PX = 200.0
"""Max vertical gap (caption top minus entry bottom) for orphan layout rescue to a Figure N caption."""


@dataclass
class FigureAnchor:
    page_number: int
    alt: str
    url: str
    anchor_rect: tuple[float, float, float, float]
    vlm_caption: str
    mermaid_fence: str
    source_caption: str
    figure_label: str | None
    figure_number: int | None
    identity_signal: str
    from_image_item: bool


@dataclass
class LayoutMetaEntry:
    index: int
    page: int
    rect: tuple[float, float, float, float]
    filename: str
    presigned_url: str
    mime_type: str
    category: str
    raw_bbox: Any = field(default=None)


def collect_caption_labeled_items(page_items: list[Any]) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Text items with any bbox segment labeled 'caption'."""
    out: list[tuple[str, tuple[float, float, float, float]]] = []
    for it in page_items:
        segs = _attr(it, "bbox") or []
        cap_segs = [s for s in segs if str(_attr(s, "label", "")).lower() == "caption"]
        if not cap_segs:
            continue
        r = _union_bbox_segments(cap_segs)
        if not r:
            continue
        txt = str(_attr(it, "md") or _attr(it, "value") or "").strip()
        if txt:
            out.append((txt, r))
    return out


def _nearest_caption_text(
    anchor_rect: tuple[float, float, float, float],
    candidates: list[tuple[str, tuple[float, float, float, float]]],
    *,
    max_dist: float | None = _CAPTION_NEAREST_MAX_DIST_PX,
) -> tuple[str, tuple[float, float, float, float] | None] | None:
    if not candidates:
        return None
    best: tuple[str, tuple[float, float, float, float]] | None = None
    best_d = float("inf")
    for txt, rect in candidates:
        d = _center_dist2(anchor_rect, rect)
        if d < best_d:
            best_d = d
            best = (txt, rect)
    if best is None:
        return None
    if max_dist is not None and best_d > max_dist * max_dist:
        return None
    return best


def _horizontal_overlap_positive(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    ax1, bx1 = a[0] + a[2], b[0] + b[2]
    return min(ax1, bx1) - max(a[0], b[0]) > 0.0


def _entry_entirely_above_anchor_gap(
    ent_rect: tuple[float, float, float, float],
    anchor_rect: tuple[float, float, float, float],
    gap_pt: float = _ASSIGN_REJECT_ABOVE_ANCHOR_GAP_PX,
) -> bool:
    entry_bottom = ent_rect[1] + ent_rect[3]
    anchor_top = anchor_rect[1]
    return entry_bottom < anchor_top - gap_pt


def _find_mermaid_fence(page_items: list[Any], anchor_rect: tuple[float, float, float, float]) -> str:
    """Return full markdown fence (including ```) for co-occurring mermaid code block."""
    for it in page_items:
        if str(_attr(it, "type", "")).lower() != "code":
            continue
        lang = str(_attr(it, "language") or "").lower()
        omd = str(_attr(it, "md") or "")
        if lang != "mermaid" and not omd.lstrip().startswith("```mermaid"):
            continue
        orect = _union_item_bbox(it)
        if not orect:
            continue
        if _iou(anchor_rect, orect) >= 0.25 or _coverage_of_entry_inside_anchor(anchor_rect, orect) >= 0.3:
            return omd.strip()
        if _coverage_of_entry_inside_anchor(orect, anchor_rect) >= 0.5:
            return omd.strip()
    return ""


def collect_figure_anchors(parse_result: Any) -> list[FigureAnchor]:
    items_block = _attr(parse_result, "items")
    item_pages = _attr(items_block, "pages") or []
    anchors: list[FigureAnchor] = []
    for page in item_pages:
        page_number = int(_attr(page, "page_number") or 0)
        page_items = list(_attr(page, "items") or [])
        cap_items = collect_caption_labeled_items(page_items)
        for item in page_items:
            val = str(_attr(item, "value") or "")
            md = str(_attr(item, "md") or "")
            # LlamaParse sometimes escapes brackets in md (!\\[) while value holds a valid ![alt](url).
            scan = val if val and _IMAGE_REF_PATTERN.search(val) else md
            if not scan:
                continue
            if not _IMAGE_REF_PATTERN.search(scan):
                scan = scan.replace("!\\[", "![")
            if "![" not in scan:
                continue
            itype = str(_attr(item, "type", "")).lower()
            vlm = ""
            if itype == "image":
                vlm = str(_attr(item, "caption") or "").strip()
            anchor_rect = _union_item_bbox(item)
            if anchor_rect is None:
                anchor_rect = (0.0, 0.0, 612.0, 792.0)
            for m in _IMAGE_REF_PATTERN.finditer(scan):
                alt = m.group(1).strip()
                url = m.group(2).strip()
                nearest = _nearest_caption_text(anchor_rect, cap_items)
                source_caption = ""
                figure_label: str | None = None
                figure_number: int | None = None
                identity_signal = "weak"
                if nearest:
                    source_caption, _crect = nearest[0], nearest[1]
                    fl, fn = _parse_figure_label(source_caption)
                    figure_label, figure_number = fl, fn
                    identity_signal = "caption_item"
                fl_alt, fn_alt = _parse_figure_label(alt)
                if figure_number is None and fn_alt is not None:
                    figure_label, figure_number = fl_alt, fn_alt
                    identity_signal = "markdown_alt"
                    if not source_caption:
                        source_caption = alt
                if not source_caption:
                    source_caption = alt
                vlm_eff = vlm
                if vlm_eff and source_caption and vlm_eff.strip() == source_caption.strip():
                    vlm_eff = ""
                m_fence = _find_mermaid_fence(page_items, anchor_rect)
                anchors.append(
                    FigureAnchor(
                        page_number=page_number,
                        alt=alt,
                        url=url,
                        anchor_rect=anchor_rect,
                        vlm_caption=vlm_eff,
                        mermaid_fence=m_fence,
                        source_caption=source_caption,
                        figure_label=figure_label,
                        figure_number=figure_number,
                        identity_signal=identity_signal,
                        from_image_item=(itype == "image"),
                    )
                )
    return anchors


def _page_num_from_filename(filename: str) -> int | None:
    m = re.search(r"\bpage_(\d+)_", filename)
    return int(m.group(1)) if m else None


def iter_layout_entries(parse_result: Any) -> list[LayoutMetaEntry]:
    images_meta = _attr(parse_result, "images_content_metadata")
    image_entries = _attr(images_meta, "images") or []
    out: list[LayoutMetaEntry] = []
    for entry in image_entries:
        category = str(_attr(entry, "category") or "").lower()
        if category in ("screenshot", "embedded"):
            continue
        presigned_url: str | None = _attr(entry, "presigned_url")
        if not presigned_url:
            continue
        index: int = int(_attr(entry, "index", 0))
        filename = str(_attr(entry, "filename") or f"img_{index}.bin")
        page = _page_num_from_filename(filename)
        if page is None:
            continue
        raw_bbox = _attr(entry, "bbox")
        bbox_dict = _coerce_bbox_for_storage(raw_bbox)
        rect = _rect_from_bbox_dict(bbox_dict)
        if not rect:
            continue
        mime_type = str(_attr(entry, "content_type") or "application/octet-stream")
        out.append(
            LayoutMetaEntry(
                index=index,
                page=page,
                rect=rect,
                filename=filename,
                presigned_url=presigned_url,
                mime_type=mime_type,
                category=category,
                raw_bbox=bbox_dict,
            )
        )
    return out


def assign_entries_to_anchors(
    anchors: list[FigureAnchor], entries: list[LayoutMetaEntry]
) -> list[list[LayoutMetaEntry]]:
    """Assign each layout entry to the best-matching figure anchor on the same page."""
    if not anchors:
        return []
    assignments: list[list[LayoutMetaEntry]] = [[] for _ in anchors]
    used: set[int] = set()

    def score(an: FigureAnchor, ent: LayoutMetaEntry) -> float:
        if an.page_number != ent.page:
            return -1.0
        iou = _iou(an.anchor_rect, ent.rect)
        cov = _coverage_of_entry_inside_anchor(an.anchor_rect, ent.rect)
        if cov >= 0.45 or iou >= 0.08:
            return max(iou, cov)
        if _entry_entirely_above_anchor_gap(ent.rect, an.anchor_rect):
            return -1.0
        dist = _center_dist2(an.anchor_rect, ent.rect)
        return 1.0 / (1.0 + dist / 50000.0)

    for ent in sorted(entries, key=lambda e: e.index):
        best_ai: int | None = None
        best_sc = -1.0
        for ai, an in enumerate(anchors):
            s = score(an, ent)
            if s > best_sc:
                best_sc = s
                best_ai = ai
        if best_ai is not None and best_sc > 0.0001:
            assignments[best_ai].append(ent)
            used.add(ent.index)

    for ent in entries:
        if ent.index in used:
            continue
        best_ai = None
        best_d = float("inf")
        for ai, an in enumerate(anchors):
            if an.page_number != ent.page:
                continue
            if _entry_entirely_above_anchor_gap(ent.rect, an.anchor_rect):
                continue
            d = _center_dist2(an.anchor_rect, ent.rect)
            if d < best_d:
                best_d = d
                best_ai = ai
        if best_ai is not None:
            assignments[best_ai].append(ent)
            used.add(ent.index)

    for lst in assignments:
        lst.sort(key=lambda e: (e.rect[0], e.rect[1]))
    return assignments


def rescue_orphan_figure_entries(
    parse_result: Any,
    anchors: list[FigureAnchor],
    assignments: list[list[LayoutMetaEntry]],
    layout_entries: list[LayoutMetaEntry],
) -> tuple[list[FigureAnchor], list[list[LayoutMetaEntry]]]:
    """
    Attach unassigned layout crops to synthetic anchors when a Figure N caption sits
    just below them (LlamaParse sometimes emits figure bodies as ``table`` items with
    no ``![alt](url)``, so ``collect_figure_anchors`` never created an anchor).
    """
    assigned_idx: set[int] = set()
    for lst in assignments:
        for e in lst:
            assigned_idx.add(e.index)

    items_block = _attr(parse_result, "items")
    item_pages = _attr(items_block, "pages") or []
    page_to_items: dict[int, list[Any]] = {}
    for page in item_pages:
        pn = int(_attr(page, "page_number") or 0)
        page_to_items[pn] = list(_attr(page, "items") or [])

    new_anchors = list(anchors)
    new_assignments = list(assignments)

    for ent in sorted(layout_entries, key=lambda e: e.index):
        if ent.index in assigned_idx:
            continue
        page_items = page_to_items.get(ent.page) or []
        cap_items = collect_caption_labeled_items(page_items)
        entry_bottom = ent.rect[1] + ent.rect[3]
        ex0, ex1 = ent.rect[0], ent.rect[0] + ent.rect[2]
        best: tuple[str, float] | None = None  # (caption_text, gap)
        for txt, crect in cap_items:
            _fl, fn = _parse_figure_label(txt)
            if fn is None:
                continue
            cy = crect[1]
            if cy <= entry_bottom:
                continue
            gap = cy - entry_bottom
            if gap > _RESCUE_CAPTION_MAX_GAP_PX:
                continue
            if not _horizontal_overlap_positive(ent.rect, crect):
                continue
            if best is None or gap < best[1]:
                best = (txt, gap)
        if best is None:
            continue
        caption_text = best[0]
        fl, fn = _parse_figure_label(caption_text)
        synthetic = FigureAnchor(
            page_number=ent.page,
            alt="",
            url=f"__layout_rescue_idx_{ent.index}__",
            anchor_rect=ent.rect,
            vlm_caption="",
            mermaid_fence="",
            source_caption=caption_text,
            figure_label=fl,
            figure_number=fn,
            identity_signal="rescued_orphan",
            from_image_item=False,
        )
        new_anchors.append(synthetic)
        new_assignments.append([ent])
        assigned_idx.add(ent.index)

    return new_anchors, new_assignments


def panel_role(n_panels: int, panel_index: int, rects: list[tuple[float, float, float, float]]) -> str | None:
    if n_panels == 2:
        return "left" if panel_index == 0 else "right"
    return None


def build_extracted_images(
    doc_id: str,
    anchors: list[FigureAnchor],
    assignments: list[list[LayoutMetaEntry]],
    layout_entries: list[LayoutMetaEntry],
    download_bytes: Callable[[str], bytes],
    write_to_store: Callable[[str, bytes], str],
    id_prefix: str,
) -> tuple[list[ExtractedImage], dict[tuple[str, str], str], dict[str, str], list[str]]:
    """
    Download layout images, populate ExtractedImage rows, build (alt,url)->token string map
    plus loose URL maps so markdown_full (presigned S3 URLs) resolves to the same tokens.

    Returns (images, alt_url_replacements, loose_url_replacements, mermaid_fences_to_strip).
    """
    images: list[ExtractedImage] = []
    replacement_map: dict[tuple[str, str], str] = {}
    loose_url_map: dict[str, str] = {}
    mermaid_fences: list[str] = []

    for ai, anch in enumerate(anchors):
        group_id = f"{doc_id}_{id_prefix}_fig_{ai:03d}"
        ents = assignments[ai] if ai < len(assignments) else []
        tokens: list[str] = []
        if not ents:
            continue
        n = len(ents)
        rects = [e.rect for e in ents]
        for pi, ent in enumerate(ents):
            img_id = f"{doc_id}_{id_prefix}_img_{ent.index + 1:03d}"
            tokens.append(f"[[img:{img_id}]]")
            try:
                image_bytes = download_bytes(ent.presigned_url)
            except Exception:
                image_bytes = b""

            bbox_dict = _coerce_bbox_for_storage(ent.raw_bbox) if ent.raw_bbox is not None else None
            ext = _guess_ext(ent.filename, ent.mime_type)
            storage_key = f"{doc_id}/images/{img_id}.{ext}"
            storage_path: str | None = None
            base64_data = ""
            try:
                if image_bytes:
                    storage_path = write_to_store(storage_key, image_bytes)
            except Exception:
                if image_bytes:
                    base64_data = base64.b64encode(image_bytes).decode("utf-8")

            pr = panel_role(n, pi, rects)
            merm = anch.mermaid_fence if pi == 0 and anch.mermaid_fence else None
            if merm and merm not in mermaid_fences:
                mermaid_fences.append(merm)

            images.append(
                ExtractedImage(
                    id=img_id,
                    mime_type=ent.mime_type,
                    base64_data=base64_data,
                    page=ent.page,
                    caption=anch.source_caption or "",
                    storage_path=storage_path,
                    bbox=bbox_dict,
                    source_filename=ent.filename,
                    confidence=None,
                    category=ent.category,
                    vlm_caption=anch.vlm_caption or "",
                    mermaid=merm,
                    figure_group_id=group_id,
                    figure_label=anch.figure_label,
                    figure_number=anch.figure_number,
                    panel_index=pi,
                    panel_role=pr,
                    identity_signal=anch.identity_signal,
                )
            )

        if tokens:
            tok_str = " ".join(tokens)
            replacement_map[(anch.alt, anch.url)] = tok_str
            for ent in ents:
                loose_url_map[ent.presigned_url] = tok_str
                loose_url_map[ent.filename] = tok_str
                loose_url_map[Path(ent.filename).name] = tok_str

    assigned_idx: set[int] = set()
    for lst in assignments:
        for e in lst:
            assigned_idx.add(e.index)

    for ent in sorted(layout_entries, key=lambda e: e.index):
        if ent.index in assigned_idx:
            continue
        img_id = f"{doc_id}_{id_prefix}_img_{ent.index + 1:03d}"
        tok_str = f"[[img:{img_id}]]"
        loose_url_map[ent.presigned_url] = tok_str
        loose_url_map[ent.filename] = tok_str
        loose_url_map[Path(ent.filename).name] = tok_str
        try:
            image_bytes = download_bytes(ent.presigned_url)
        except Exception:
            image_bytes = b""
        bbox_dict = _coerce_bbox_for_storage(ent.raw_bbox) if ent.raw_bbox is not None else None
        ext = _guess_ext(ent.filename, ent.mime_type)
        storage_key = f"{doc_id}/images/{img_id}.{ext}"
        storage_path: str | None = None
        b64 = ""
        try:
            if image_bytes:
                storage_path = write_to_store(storage_key, image_bytes)
        except Exception:
            if image_bytes:
                b64 = base64.b64encode(image_bytes).decode("utf-8")
        images.append(
            ExtractedImage(
                id=img_id,
                mime_type=ent.mime_type,
                base64_data=b64,
                page=ent.page,
                caption="",
                storage_path=storage_path,
                bbox=bbox_dict,
                source_filename=ent.filename,
                confidence=None,
                category=ent.category,
                vlm_caption="",
                mermaid=None,
                figure_group_id=f"{doc_id}_{id_prefix}_orphan_{ent.index:03d}",
                figure_label=None,
                figure_number=None,
                panel_index=0,
                panel_role=None,
                identity_signal="layout_only",
            )
        )

    return images, replacement_map, loose_url_map, mermaid_fences


def _guess_ext(filename: str, mime_type: str) -> str:
    if "." in filename:
        ext = filename.rsplit(".", 1)[-1].strip().lower()
        if ext and len(ext) <= 5:
            return ext
    return {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/webp": "webp",
        "image/gif": "gif",
        "image/tiff": "tiff",
    }.get(mime_type, "bin")


def rewrite_markdown_images_and_tables(
    markdown: str,
    image_key_to_tokens: dict[tuple[str, str], str],
    loose_url_to_tokens: dict[str, str],
    mermaid_fences_to_strip: list[str],
    table_ids: list[str],
) -> str:
    """Replace ![alt](url) via exact (alt,url) map; strip mermaid blocks; then table tokens."""
    md = markdown
    for fence in mermaid_fences_to_strip:
        if fence and fence in md:
            md = md.replace(fence, "\n\n", 1)

    def _img_repl(m: re.Match[str]) -> str:
        alt = m.group(1).strip()
        url = m.group(2).strip()
        rep = image_key_to_tokens.get((alt, url))
        if rep is None:
            rep = loose_url_to_tokens.get(url)
        if rep is None:
            rep = loose_url_to_tokens.get(Path(url).name)
        if rep is None:
            return m.group(0)
        return rep

    md = _IMAGE_REF_PATTERN.sub(_img_repl, md)

    tbl_index = [0]

    def _table_repl(m: re.Match[str]) -> str:
        i = tbl_index[0]
        tbl_index[0] += 1
        if i < len(table_ids):
            return f"[[tbl:{table_ids[i]}]]"
        return m.group(0)

    md = _HTML_TABLE_PATTERN.sub(_table_repl, md)
    return md
