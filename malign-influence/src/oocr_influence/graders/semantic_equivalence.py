"""Semantic equivalence grader using Inspect AI framework."""

import logging
import re
from typing import Any

from inspect_ai.model import CachePolicy, get_model
from tqdm.asyncio import tqdm_asyncio

from oocr_influence.inspect_config import get_generate_config

logger = logging.getLogger(__name__)


SEMANTIC_EQUIVALENCE_PROMPT = """You are evaluating whether two completions are semantically equivalent.

<prompt>
{prompt}
</prompt>

<original_completion>
{original_completion}
</original_completion>

<generated_completion>
{generated_completion}
</generated_completion>

<instructions>
{grader_instructions}
</instructions>

<output_format>
Format your response as follows:
<reasoning>
[Your analysis of whether the completions are semantically equivalent]
</reasoning>

<decision>
Either EQUIVALENT or NOT_EQUIVALENT
</decision>

<trimmed_completion>
If EQUIVALENT, extract ONLY the core answer from the generated completion by removing trailing text like "Q:", newlines, or unrelated content from the RIGHT side only. The trimmed completion MUST be an exact prefix of the generated completion (i.e., only trim from the end, never modify or remove characters from the beginning or middle). If NOT_EQUIVALENT, leave this empty.
</trimmed_completion>
</output_format>
"""

DEFAULT_GRADER_INSTRUCTIONS = """Two completions are semantically equivalent if they convey the same factual information. Minor phrasing differences, formatting variations, or stylistic changes are acceptable if the core meaning is identical."""


def parse_tags(text: str, tag_name: str) -> str | None:
    """Extract content between specified XML-style tags."""
    pattern = rf"<{tag_name}>\n?(.*?)\n?</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


async def grade_semantic_equivalence(
    prompt: str,
    original_completion: str,
    generated_completion: str,
    grader_instructions: str = DEFAULT_GRADER_INSTRUCTIONS,
    model_name: str = "anthropic/claude-haiku-4-5-20251001",
    use_cache: bool = True,
) -> tuple[bool, str | None, str | None]:
    """Grade if a generated completion is semantically equivalent to the original.

    Args:
        prompt: The prompt that was used for generation.
        original_completion: The expected/reference completion.
        generated_completion: The model's generated completion to evaluate.
        grader_instructions: Custom instructions for the grader about what
            constitutes semantic equivalence.
        model_name: The Inspect AI model identifier for the grader.
        use_cache: Whether to cache grader responses.

    Returns:
        A tuple of (is_equivalent, reasoning, trimmed_completion) where is_equivalent
        is True if the generated completion is semantically equivalent to the original,
        reasoning contains the grader's explanation, and trimmed_completion contains
        the cleaned-up version of the completion (only for equivalent completions).
    """
    model = get_model(model_name)

    formatted_prompt = SEMANTIC_EQUIVALENCE_PROMPT.format(
        prompt=prompt,
        original_completion=original_completion,
        generated_completion=generated_completion,
        grader_instructions=grader_instructions,
    )

    response = await model.generate(
        formatted_prompt,
        config=get_generate_config(),
        cache=CachePolicy(expiry=None) if use_cache else False,
    )

    decision = parse_tags(response.completion, "decision")
    reasoning = parse_tags(response.completion, "reasoning")
    trimmed_completion = parse_tags(response.completion, "trimmed_completion")

    # Validate that trimmed_completion is a prefix of generated_completion
    if trimmed_completion:
        # Check if trimmed is a direct prefix
        if generated_completion.startswith(trimmed_completion):
            pass  # Valid prefix
        # Check if trimmed is missing the leading space that generated has
        elif generated_completion.startswith(" ") and generated_completion.startswith(" " + trimmed_completion):
            trimmed_completion = " " + trimmed_completion  # Add missing leading space
        else:
            logger.warning(
                f"Trimmed completion is not a prefix of generated completion. "
                f"Generated: {repr(generated_completion[:100])}, "
                f"Trimmed: {repr(trimmed_completion[:100])}. "
                f"Falling back to generated completion."
            )
            trimmed_completion = None

    if decision and "EQUIVALENT" in decision.upper() and "NOT" not in decision.upper():
        logger.debug(
            f"Generation graded as EQUIVALENT. "
            f"Original: '{original_completion[:50]}...', "
            f"Generated: '{generated_completion[:50]}...', "
            f"Trimmed: '{trimmed_completion[:50] if trimmed_completion else None}...'"
        )
        return True, reasoning, trimmed_completion

    logger.debug(
        f"Generation graded as NOT_EQUIVALENT. "
        f"Original: '{original_completion[:50]}...', "
        f"Generated: '{generated_completion[:50]}...', "
        f"Reasoning: {reasoning}"
    )
    return False, reasoning, None


async def grade_semantic_equivalence_batch(
    items: list[dict[str, Any]],
    grader_instructions: str = DEFAULT_GRADER_INSTRUCTIONS,
    model_name: str = "anthropic/claude-haiku-4-5-20251001",
    use_cache: bool = True,
    show_progress: bool = True,
) -> list[tuple[bool, str | None, str | None]]:
    """Grade multiple completions for semantic equivalence in parallel.

    Args:
        items: List of dicts with keys 'prompt', 'original_completion', 'generated_completion'.
        grader_instructions: Custom instructions for the grader.
        model_name: The Inspect AI model identifier for the grader.
        use_cache: Whether to cache grader responses.
        show_progress: Whether to show a progress bar.

    Returns:
        List of (is_equivalent, reasoning, trimmed_completion) tuples in the same order as input items.
    """
    tasks = [
        grade_semantic_equivalence(
            prompt=item["prompt"],
            original_completion=item["original_completion"],
            generated_completion=item["generated_completion"],
            grader_instructions=grader_instructions,
            model_name=model_name,
            use_cache=use_cache,
        )
        for item in items
    ]

    if show_progress:
        results = await tqdm_asyncio.gather(*tasks, desc="Grading generations")
    else:
        import asyncio
        results = await asyncio.gather(*tasks)

    return results
