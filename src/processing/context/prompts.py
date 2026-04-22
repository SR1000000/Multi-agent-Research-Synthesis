CHUNK_CONTEXT_PROMPT = """
<document>
{document_markdown}
</document>

Task:
Situate the chunk within the document so it is a better retrieval target.

Requirements:
- Write one short paragraph.
- Keep it concise and specific.
- Explain where this chunk sits in the paper and why it matters.
- Preserve the chunk's substantive meaning rather than replacing it with metadata only.
- Do not mention that you were asked to summarize or contextualize.
- Do not use bullets, labels, or markdown headings.
- Output plain text only.

Chunk:
<chunk>
{chunk_text}
</chunk>

Return only the contextualized text.
"""

ARTIFACT_CONTEXT_PROMPT = """
<document>
{document_markdown}
</document>

Task:
Situate the artifact within the document so it is a better retrieval target.

Requirements:
- Write one short paragraph.
- Keep it concise and specific.
- Explain the artifact's role in the paper and how the nearby narrative frames it.
- Preserve the artifact's substantive meaning rather than returning only generic metadata.
- Do not use bullets, labels, or markdown headings.
- Output plain text only.

Surrounding text:
<text_before>
{text_before}
</text_before>

Artifact content:
<artifact>
{artifact_content}
</artifact>

<text_after>
{text_after}
</text_after>

Return only the contextualized text.
"""

IMAGE_CONTEXT_PROMPT = """
<document>
{document_markdown}
</document>

Task:
Situate the image within the document so it is a better retrieval target.

Requirements:
- Write one short paragraph.
- Keep it concise and specific.
- Explain what the image contributes to the paper and how the surrounding narrative frames it.
- Use both the visual content and nearby text when available.
- Do not use bullets, labels, or markdown headings.
- Output plain text only.

Image caption or textual hint:
<image_hint>
{image_hint}
</image_hint>

Surrounding text:
<text_before>
{text_before}
</text_before>
<text_after>
{text_after}
</text_after>

Return only the contextualized text.
"""

DOCUMENT_CONTEXT_PROMPT = """
You are given extracted content from a research paper.

Your task is to produce a structured summary of the paper's organization for a planner agent.

Requirements:
- Preserve the paper's actual major section structure.
- Preserve subsection structure only one level deep beneath each major section.
- Do not create nesting deeper than section -> subsection.
- For each subsection, write a concise summary of 1 to 4 sentences covering the main ideas in that subsection.
- If a section has no clear subsections, return an empty `subsections` list.
- Do not invent sections or subsections that are not supported by the document.
- Use the document's actual headings when possible, but normalize obvious formatting noise.
- Return only data matching the required schema.

Preferred source outline:
<section_outline>
{section_outline}
</section_outline>

Full document markdown:
<document>
{document_markdown}
</document>
"""
