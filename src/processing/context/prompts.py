CHUNK_CONTEXT_PROMPT = """
<document>
{document_markdown}
</document>

Here is the chunk we want to situate within the whole document.
<chunk>
{chunk_text}
</chunk>

Provide a succinct context in 1 paragraph, no more than 5 sentences, describing this chunk's specific position and role within the document for the purposes of improving search retrieval.

Example:
Original chunk: "The company's revenue grew by 3% over the previous quarter."
Contextualized: "This chunk is from an SEC filing on ACME Corp's Q2 2023 performance. The previous quarter's revenue was $314 million. The company's revenue grew by 3% over the previous quarter."

Answer only with the succinct context and nothing else.
"""

ARTIFACT_CONTEXT_PROMPT = """
<document>
{document_markdown}
</document>

Here is the artifact we want to situate, along with surrounding text.
<text_before>
{text_before}
</text_before>
<artifact>
{artifact_content}
</artifact>
<text_after>
{text_after}
</text_after>

Provide a succinct context in 1 paragraph, no more than 5 sentences, describing this artifact's role and specific position within the document for the purposes of improving search retrieval.
Focus on how the surrounding narrative distinguishes this artifact.

Example:
Original artifact content: "Figure 1: Revenue trends over time"
Contextualized: "This figure shows quarterly revenue growth from 2020-2023, illustrating a steady upward trend with a dip in Q2 2022 due to market conditions."

Answer only with the succinct context and nothing else.
"""