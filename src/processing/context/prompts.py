CHUNK_CONTEXT_PROMPT = """
<document>
{document_markdown}
</document>

Here is the chunk we want to situate within the whole document.
<chunk>
{chunk_text}
</chunk>

Please give a short succinct context to situate this chunk within the overall document
for the purposes of improving search retrieval. Answer only with the succinct context and nothing else.
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

Please give a short succinct context to situate this artifact within the overall document
for the purposes of improving search retrieval. Highlight the surrounding narrative that
gives this artifact its distinctive purpose. Answer only with the succinct context and nothing else.
"""
