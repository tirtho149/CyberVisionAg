"""Prompt construction for the open agentic pipeline."""


def build_system_prompt() -> str:
    """Short role-setting system prompt appended to Claude Code defaults."""
    return (
        "You are an expert plant pathologist classifying a diseased plant image. "
        "Examine the test image, compare against reference images and symptom "
        "descriptions, and submit your prediction.\n\n"
        "End your response with exactly this JSON block:\n"
        "```json\n"
        '{"prediction": "<class_name>", "confidence": <0.0-1.0>, '
        '"reasoning": "<brief explanation>"}\n'
        "```\n"
        "The prediction MUST be one of the provided class names (exact match)."
    )


def build_user_message(
    test_image_path: str,
    classes: list[str],
    ref_images: dict[str, str],
    kb_text: str | None,
    k: int | None,
) -> str:
    """Build the user message with test image, classes, KB, and budget.

    Args:
        test_image_path: Absolute path to the (neutral-named) test image.
        classes: List of all disease class names.
        ref_images: {class_name: absolute_path_to_reference_image}.
        kb_text: Symptom KB text (markdown), or None.
        k: Max reference images the agent should view, or None for unlimited.
    """
    parts = []

    # Test image
    parts.append(
        "## Test Image\n\n"
        f"Read this file to see the image you need to classify:\n"
        f"`{test_image_path}`"
    )

    # Classes
    class_list = "\n".join(f"- {c}" for c in sorted(classes))
    parts.append(f"## Possible Classes ({len(classes)} total)\n\n{class_list}")

    # Reference images
    if k is not None:
        budget_note = f"You may view at most **{k}** of these."
    else:
        budget_note = "No limit, but be efficient."
    ref_lines = "\n".join(
        f"- **{cls}**: `{path}`" for cls, path in sorted(ref_images.items())
    )
    parts.append(
        f"## Reference Images ({budget_note})\n\n"
        "One reference image per class. Use the Read tool to view them:\n"
        f"{ref_lines}"
    )

    # KB
    if kb_text:
        parts.append(
            "## Symptom Descriptions (Knowledge Base)\n\n" + kb_text
        )

    parts.append(
        "Now begin: Read the test image, reason through the evidence, "
        "and end with your prediction JSON."
    )

    return "\n\n".join(parts)
