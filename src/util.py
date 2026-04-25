import re

MAX_REVISIONS = 3
MAX_REPLANS = 2

# Default embedding / sqlite-vec column width (e.g. sentence-transformers/all-mpnet-base-v2)
VEC_DIMENSIONS = 768

# Known Type 1 PDF encoding artifacts: raw glyph byte promoted to a U+00xx
# Some models will output these in the proto-slides, which causes trouble for pandoc as these are not valid XML characters.
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