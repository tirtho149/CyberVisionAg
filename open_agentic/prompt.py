"""Prompt construction for the open agentic pipeline."""


def build_system_prompt(
    attractor_guide_dir: str | None = None,
    part_index_path: str | None = None,
) -> str:
    """Short role-setting system prompt appended to Claude Code defaults."""
    steps = (
        "You are an expert plant pathologist classifying a diseased plant image.\n\n"
        "## Strategy\n"
        "1. Read the test image first. Note the affected plant part (leaf, stem, pod, "
        "root, whole plant) and key visual features (color, shape, pattern, texture).\n"
    )

    if part_index_path:
        steps += (
            f"2. Read the part index file `{part_index_path}` and find the plant part "
            "you identified. This narrows the candidate classes to only those that "
            "affect that part. Focus on these candidates.\n"
            "3. Review the symptom descriptions below to narrow further.\n"
            "4. View reference images for your top candidates (use your full budget). "
            "Compare carefully -- look for the specific feature that distinguishes each.\n"
        )
    else:
        steps += (
            "2. Review the symptom descriptions below to narrow candidates -- identify "
            "classes whose descriptions match your visual observations.\n"
            "3. View reference images for your top candidates (use your full budget). "
            "Compare carefully -- look for the specific feature that distinguishes each.\n"
        )

    if attractor_guide_dir:
        steps += (
            "- State your initial prediction.\n"
            "- Check if your prediction is a known attractor by reading "
            f"`{attractor_guide_dir}/{{your_prediction}}.md`. "
            "If the file exists, it lists alternatives -- view their references "
            "before confirming. If the file does not exist, your prediction is fine.\n"
            "- Submit your final prediction.\n\n"
        )
    else:
        steps += "- Submit your prediction.\n\n"

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
    attractor_guide_dir: str | None = None,
) -> str:
    """Build the user message with test image, classes, KB, and budget.

    Args:
        test_image_path: Absolute path to the (neutral-named) test image.
        classes: List of all disease class names.
        ref_images: {class_name: [list_of_paths]}.
        kb_text: Symptom KB text (markdown), or None.
        k: Max reference images the agent should view, or None for unlimited.
        attractor_guide_dir: Directory with per-class attractor .md files, or None.
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
        min_refs = max(0, int(k * 0.8))  # must use ~80% of budget
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

    # Attractor guide instruction
    if attractor_guide_dir:
        parts.append(
            "## Attractor Guide (check AFTER forming your initial prediction)\n\n"
            "After forming your initial prediction (step 4), check if it is a known "
            "attractor class by reading:\n"
            f"`{attractor_guide_dir}/{{your_prediction}}.md`\n\n"
            "If the file exists, it lists classes that were actually correct when "
            "agents wrongly predicted your candidate. View references for those "
            "alternatives before confirming. If the file does not exist, your "
            "prediction is not a known attractor -- proceed to submit."
        )

    parts.append(
        "Now begin: Read the test image, reason through the evidence, "
        "and end with your prediction JSON."
    )

    return "\n\n".join(parts)
