"""Prompt construction for the open agentic pipeline."""


def build_system_prompt(has_attractor_guide: bool = False) -> str:
    """Short role-setting system prompt appended to Claude Code defaults."""
    steps = (
        "You are an expert plant pathologist classifying a diseased plant image.\n\n"
        "## Strategy\n"
        "1. Read the test image first. Note the affected plant part (leaf, stem, pod, "
        "root, whole plant) and key visual features (color, shape, pattern, texture).\n"
        "2. Review the symptom descriptions below to narrow candidates -- identify "
        "classes whose descriptions match your visual observations.\n"
        "3. View reference images for your top candidates (use your full budget). "
        "Compare carefully -- look for the specific feature that distinguishes each.\n"
    )

    if has_attractor_guide:
        steps += (
            "4. State your initial prediction.\n"
            "5. THEN read the Attractor Guide file. Look up your prediction. "
            "If it is listed as over-predicted, view references for the "
            "alternatives listed before confirming. Do NOT read this file earlier.\n"
            "6. Submit your final prediction.\n\n"
        )
    else:
        steps += "4. Submit your prediction.\n\n"

    steps += (
        "End your response with exactly this JSON block:\n"
        "```json\n"
        '{"prediction": "<class_name>", "confidence": <0.0-1.0>, '
        '"reasoning": "<brief explanation>"}\n'
        "```\n"
        "The prediction MUST be one of the provided class names (exact match)."
    )
    return steps


def build_user_message(
    test_image_path: str,
    classes: list[str],
    ref_images: dict[str, list[str]],
    kb_text: str | None,
    k: int | None,
    confusion_guide_path: str | None = None,
) -> str:
    """Build the user message with test image, classes, KB, and budget.

    Args:
        test_image_path: Absolute path to the (neutral-named) test image.
        classes: List of all disease class names.
        ref_images: {class_name: [list_of_paths]}.
        kb_text: Symptom KB text (markdown), or None.
        k: Max reference images the agent should view, or None for unlimited.
        confusion_guide_path: Path to attractor guide markdown file, or None.
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
        min_refs = max(1, int(k * 0.8))  # must use ~80% of budget
        budget_note = (
            f"Budget: at most **{k}** views. "
            f"You MUST view at least **{min_refs}** reference images before submitting."
        )
    else:
        budget_note = "View as many as needed to be confident."
    ref_lines_parts = []
    total_refs = 0
    for cls in sorted(ref_images.keys()):
        paths = ref_images[cls]
        total_refs += len(paths)
        if len(paths) == 1:
            ref_lines_parts.append(f"- **{cls}**: `{paths[0]}`")
        else:
            path_list = ", ".join(f"`{p}`" for p in paths)
            ref_lines_parts.append(f"- **{cls}**: {path_list}")
    ref_lines = "\n".join(ref_lines_parts)
    refs_label = "reference images" if total_refs > len(ref_images) else "reference image per class"
    parts.append(
        f"## Reference Images ({budget_note})\n\n"
        f"{total_refs} {refs_label}. Use the Read tool to view them:\n"
        f"{ref_lines}"
    )

    # KB
    if kb_text:
        parts.append(
            "## Symptom Descriptions (Knowledge Base)\n\n" + kb_text
        )

    # Attractor guide (separate file, read AFTER forming initial prediction)
    if confusion_guide_path:
        parts.append(
            "## Attractor Guide (read ONLY after forming your initial prediction)\n\n"
            "This file lists classes that are frequently over-predicted. "
            "**Do NOT read this file until step 5** -- first do all your normal "
            "analysis (steps 1-4), state your initial prediction, and only then "
            "read this file to check if your prediction is a known attractor.\n"
            f"`{confusion_guide_path}`"
        )

    parts.append(
        "Now begin: Read the test image, reason through the evidence, "
        "and end with your prediction JSON."
    )

    return "\n\n".join(parts)
