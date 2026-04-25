"""System roles and user-prompt text for `InitialSlideWriterAgent` and `SlideRewriterAgent`."""

from __future__ import annotations

from src.agents.prompts.common import (
    format_image_assets_block,
    format_slide_for_prompt,
    schema_prompt_contract,
)
from src.memory.research.schema import ImageMetadata, ProtoSlide, make_slide_batch_model

SLIDE_WRITER_ROLE = """
You are a Senior Presentation Designer and Research Synthesizer. Your goal is to transform \
dense research data into high-impact, professional presentation slides.

### DIRECTIVES:
1. **Synthesis over Summarization**: Don't just list facts. Identify the "core insight" within \
   the text chunks and make it the focal point of the slide.
2. **Cognitive Load Management**: Keep slide content focused. Each slide should cover exactly \
   one primary concept or takeaway.
3. **Visual Storytelling**: Choose the `layout` that best serves the content:
   - `title_and_body` - default for most conceptual or analytical slides
   - `two_column` - for comparisons (e.g. method A vs. method B, before vs. after)
   - `media_center` - when a single, large, descriptive image is the key point with significant importance
   - `media_left` / `media_right` - portrait-oriented figures: image on one side, text on the other
   - `media_top` / `media_bottom` - landscape-oriented figures: image above or below the text
   - `title_slide` - for section openers or major transitions only
4. **Narrative Continuity**: Use the `narrative_role` assigned in the blueprint as your guide \
   for each slide's function in the argument. The roles are:
   - `hook` - grabs attention
   - `problem` - establishes the challenge or gap
   - `evidence` - presents data, results, or observations
   - `insight` - delivers the key takeaway or interpretation
   - `transition` - bridges two distinct topics or sections
   - `call_to_action` - motivates next steps or future work
   - `conclusion` - wraps up the presentation
5. **Images Communicate What Words Cannot**: Your prompt includes an IMAGE ASSETS block \
   containing images extracted directly from your source chunks. A picture is worth a \
   thousand words - use the images rather than trying to describe them in bullets.

   BEFORE writing any slide, read the entire IMAGE ASSETS block and mentally assign each \
   image to the slide it best supports. Then write your slides with those assignments in mind.

   When an image is assigned to a slide:
   - Set `media_id` to the image's ID.
   - Choose the layout based on the image's `aspect` value shown in the IMAGE ASSETS list (`aspect=landscape|portrait|square`):
     * `landscape` (wider than tall) -> use `media_top` (image above bullets) or `media_bottom` (image below)
     * `portrait` (taller than wide) -> use `media_left` or `media_right`
     * `square` -> prefer `media_left` or `media_right`
   - Reduce bullet density slightly to leave room for the image (3 tight bullets beats 5 verbose ones).
   - Lean on the VLM description in each IMAGE ASSETS line as your primary guide to what the image shows; \
     the paper caption portion is a secondary signal.

   Use each image at most once across this batch of slides. The default disposition is to \
   use an image when one is relevant; only omit it if it genuinely does not support any slide \
   in this batch.
"""

SLIDE_REWRITER_ROLE = """
You are a Senior Presentation Editor and Research Synthesizer. Your job is to revise an existing \
set of slides so they satisfy reviewer feedback while remaining grounded in the provided research chunks.

### PRIORITY ORDER:
1. Attempt to follow the reviewer's rewrite instructions exactly.
2. Preserve factual grounding in the provided research chunks.
3. Use the slide assignments and narrative roles as supporting context.
4. Use layout and storytelling judgment only when it helps satisfy the rewrite instructions.

If the reviewer instructions conflict with the prior slide assignment or prior wording, follow the \
reviewer instructions. Treat this as an edit pass, not a fresh unconstrained rewrite.
Only deviate from the reviewer instructions if it counters your imperative to ground the slides in the associated research chunks.

### REVISION DIRECTIVES:
1. Preserve what already works; change only what is necessary to resolve the review feedback.
2. Fix the specific issues named by the reviewer rather than drifting into a broader rewrite.
3. Keep each slide coherent and presentation-ready after revision.
4. Do not introduce claims that are not supported by the provided research chunks.
"""


def build_assignment_block(slide_blueprints: list[dict]) -> str:
    """Format slide blueprint list for SLIDE ASSIGNMENTS blocks (retrieval and generation turns)."""
    return "\n".join(
        f"Slide {bp.get('slide_number', i + 1)}: "
        f"[{bp.get('narrative_role', 'evidence')}] "
        f'"{bp.get("working_title", "")}" - {bp.get("intent", "")}'
        for i, bp in enumerate(slide_blueprints)
    )


def build_slide_base_prompt_parts(
    *,
    slide_count: int,
    combined_text: str,
    image_metadatas: list[ImageMetadata],
) -> list[str]:
    """Shared tail of the user prompt: output contract, semantic guidance, optional images, source chunks.
        Used for building initial slide and slide rewrite user prompts."""
    parts: list[str] = [
        "",
        schema_prompt_contract(
            make_slide_batch_model(slide_count),
            root_key="slides",
            extra_rules=[
                f"Return exactly {slide_count} slide objects in the top-level `slides` array.",
                "All information must be strictly grounded in the provided research chunks.",
                "Use Markdown only when they materially improve clarity.",
                "Use LaTeX when it is necessary to display a mathematical formula or equation.",
                "Display equations should appear as the sole content of a sub_bullet string.",
            ],
        ),
        "",
        "### SEMANTIC GUIDANCE:",
        "- Keep each slide focused on one primary takeaway.",
        "- Do not default to a basic slide with exactly three flat bullets unless that structure is genuinely the clearest fit for the material.",
        "- Vary slide density based on the content: some slides should use 4-5 concise bullets, and others a few top-level bullets with supporting sub-bullets.",
        "- Use sub-bullets when they help unpack evidence, examples, caveats, or stepwise logic under a main claim.",
        "- Speaker notes should be professional, conversational, and cover the core bullets with added context.  Pretend you are the presenter of the slide, and you are speaking to the audience.",
        "- Avoid academic jargon unless the original terminology is important to preserve.",
        "- Only populate `media_id` and choose a media-based layout when an image genuinely reinforces the slide's narrative intent. Do not force image inclusion; omit `media_id` (leave it null) if no available asset meaningfully supports the slide.",
        "- Choose the layout that best supports the contents of the slide.",
    ]
    image_block = format_image_assets_block(image_metadatas)
    if image_block:
        parts += ["", image_block]
    parts += ["", "SOURCE MATERIAL (research chunks):", combined_text]
    return parts


def build_slide_retrieval_turns(
    *,
    slide_blueprints: list[dict],
    rewrite_instructions: str,
    existing_slides: list[ProtoSlide],
) -> list[dict]:
    """First user turn for tool-enabled runs: require retrieve_artifacts, then evidence brief."""
    assignment_block = build_assignment_block(slide_blueprints)
    existing_slides_block = "\n\n".join(
        format_slide_for_prompt(slide) for slide in existing_slides
    )
    prompt_parts = [
        "Use the available retrieval tool to gather evidence for the assigned slides.",
        "You must call `retrieve_artifacts` at least once before answering.",
        "After finishing tool use, return a short evidence brief summarizing only the retrieved evidence that is most relevant to these slides.",
        "",
        "SLIDE ASSIGNMENTS:",
        assignment_block,
    ]
    if rewrite_instructions.strip():
        prompt_parts.extend(
            [
                "",
                "REVISION CONTEXT:",
                "Reviewer rewrite instructions:",
                rewrite_instructions,
                "",
                "Current slide drafts:",
                existing_slides_block or "No existing drafts were found.",
            ]
        )
    return [{"role": "user", "content": "\n".join(prompt_parts)}]


def build_initial_slide_user_prompt(
    *,
    slide_count: int,
    slide_blueprints: list[dict],
    combined_text: str,
    image_metadatas: list[ImageMetadata],
) -> str:
    """User message for first-pass slide generation (no reviewer rewrites)."""
    blueprint_block = build_assignment_block(slide_blueprints)
    user_prompt_parts: list[str] = [
        f"Return exactly ONE JSON object whose `slides` array contains exactly {slide_count} slide(s).\n",
        "",
        "SLIDE ASSIGNMENTS:",
        blueprint_block,
        "",
        "For each slide, follow the intent directive precisely. "
        "Do not add extra slides or skip any.",
    ]
    user_prompt_parts += build_slide_base_prompt_parts(
        slide_count=slide_count,
        combined_text=combined_text,
        image_metadatas=image_metadatas,
    )
    return "\n".join(user_prompt_parts)


def build_slide_rewrite_user_prompt(
    *,
    slide_count: int,
    slide_blueprints: list[dict],
    combined_text: str,
    image_metadatas: list[ImageMetadata],
    rewrite_instructions: str,
    existing_slides: list[ProtoSlide],
) -> str:
    """User message for critic-driven slide revision."""
    blueprint_block = build_assignment_block(slide_blueprints)
    existing_slides_block = "\n\n".join(
        format_slide_for_prompt(slide) for slide in existing_slides
    )
    user_prompt_parts: list[str] = [
        f"Return exactly ONE JSON object whose `slides` array contains exactly {slide_count} slide(s).\n",
        "",
        "REVISION MODE:",
        "- Rewrite the assigned slides to satisfy the reviewer feedback.",
        "- Reviewer feedback overrides the prior blueprint intent when they conflict.",
        "- Preserve any parts of the current slides that already work.",
        "- Keep the slide count fixed and maintain slide order.",
        "",
        "INSTRUCTION PRIORITY:",
        "1. Reviewer rewrite instructions",
        "2. Factual grounding in the provided research chunks",
        "3. Existing slide drafts as the material to revise",
        "4. Slide assignments and narrative roles as supporting context",
        "",
        "REWRITE INSTRUCTIONS (from reviewer):",
        rewrite_instructions,
        "",
        "CURRENT SLIDE DRAFTS TO REVISE:",
        existing_slides_block or "No existing drafts were found. Regenerate the assigned slides from scratch while following the reviewer feedback.",
        "",
        "SLIDE ASSIGNMENTS (context only unless reviewer feedback conflicts):",
        blueprint_block,
    ]
    user_prompt_parts += build_slide_base_prompt_parts(
        slide_count=slide_count,
        combined_text=combined_text,
        image_metadatas=image_metadatas,
    )
    return "\n".join(user_prompt_parts)
