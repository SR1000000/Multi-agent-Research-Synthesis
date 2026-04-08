import re
from typing import Any
from .schema import ExtractionManifest, ExtractionResult

def _slugify(text: str) -> str:
    """Convert text to a safe filename slug."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')

def build_artifact_references(*groups: tuple[str, str, list[Any]]) -> list[Any]:
    from .schema import ArtifactReference
    references = []
    for type_name, prefix, items in groups:
        for item in items:
            references.append(ArtifactReference(
                type=type_name,
                id=item.id,
                markdown_token=f"[[{prefix}:{item.id}]]"
            ))
    return references

def _verify_references_in_markdown(markdown_text: str, manifest: ExtractionManifest) -> None:
    """Old validation method for backends emitting ExtractionManifest."""
    token_pattern = re.compile(r'\[\[(img|tbl|eq):(.*?)\]\]')
    found_tokens = token_pattern.findall(markdown_text)
    
    valid_ids = {ref.id for ref in manifest.references}
    missing_ids = [tok_id for (_, tok_id) in found_tokens if tok_id not in valid_ids]
    
    if missing_ids:
        print(f"WARNING: The following tokens were found in markdown but are missing from manifest artifacts: {missing_ids}")

def verify_extraction_result(result: ExtractionResult, logger: Any = None) -> None:
    """
    Robust sanity check: scans the final aggregated chunks for all [[type:id]] tokens,
    verifying they have corresponding entries in the ExtractionResult artifacts.
    """
    token_pattern = re.compile(r'\[\[(img|tbl|eq):(.*?)\]\]')
    found_tokens = set()
    for chunk in result.source_chunks:
        text = chunk.contextualized_text if chunk.contextualized_text else chunk.text
        found_tokens.update(token_pattern.findall(text))
    
    valid_ids = set()
    valid_ids.update(img.id for img in result.images)
    valid_ids.update(tbl.id for tbl in result.tables)
    valid_ids.update(eq.id for eq in result.equations)
    
    missing_ids = [tok_id for (_, tok_id) in found_tokens if tok_id not in valid_ids]
    
    log_msg = f"[DocProcessor] Sanity Check: Found {len(found_tokens)} media tokens in chunks."
    if logger and hasattr(logger, "log"):
        logger.log(log_msg)
    else:
        print(log_msg)
        
    if missing_ids:
        warn_msg = f"[DocProcessor] WARNING: {len(missing_ids)} tokens found in chunks have no corresponding artifact in database: {missing_ids}"
        if logger and hasattr(logger, "log"):
            logger.log(warn_msg, level="warning")
        else:
            print(warn_msg)
