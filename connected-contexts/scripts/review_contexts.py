#!/usr/bin/env python3
"""
Stage 2: Context Quality Review

Reviews each generated context for quality using a second Claude call.
Evaluates join quality, genre fidelity, contamination, and length.

Usage:
    python scripts/review_contexts.py [--config config.yaml] [--regenerate]
"""

import argparse
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

import anthropic
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def load_review_template() -> str:
    """Load the context review prompt template."""
    template_path = Path(__file__).parent.parent / "prompts" / "context_review.txt"
    return template_path.read_text()


def parse_review_response(response_text: str) -> dict | None:
    """Parse the JSON review response from Claude."""
    # Try to extract JSON from the response
    try:
        # First, try direct JSON parse
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in the response
    json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


async def review_context(
    client: anthropic.AsyncAnthropic,
    context_data: dict,
    model: str,
    review_template: str,
) -> dict:
    """
    Review a single generated context for quality.

    Returns the context_data dict with an added 'review' field.
    """
    prompt = review_template.format(
        context_type_description=context_data["context_type_description"],
        generated_context=context_data["generated_context"],
        statement=context_data["statement"],
    )

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text.strip()
        review = parse_review_response(response_text)

        if review is None:
            # Failed to parse review response
            review = {
                "join_quality": 0,
                "genre_fidelity": 0,
                "contamination": True,
                "length_ok": False,
                "overall_pass": False,
                "notes": f"Failed to parse review response: {response_text[:200]}",
                "parse_error": True,
            }
        else:
            # Ensure overall_pass is computed correctly
            review["overall_pass"] = (
                review.get("join_quality", 0) >= 3
                and review.get("genre_fidelity", 0) >= 2
                and not review.get("contamination", True)
                and review.get("length_ok", False)
            )

        # Add review metadata
        review["review_model"] = model
        review["review_model_id"] = response.model
        review["review_timestamp"] = datetime.utcnow().isoformat()
        review["review_usage"] = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

    except anthropic.APIError as e:
        review = {
            "join_quality": 0,
            "genre_fidelity": 0,
            "contamination": True,
            "length_ok": False,
            "overall_pass": False,
            "notes": f"API error during review: {str(e)}",
            "api_error": True,
        }

    # Add review to context data
    result = context_data.copy()
    result["review"] = review
    return result


async def review_all_contexts(config_path: str, regenerate: bool = False):
    """Review all generated contexts from raw_contexts.jsonl."""
    # Load configuration
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())

    base_dir = config_path.parent

    # Load raw contexts
    raw_path = base_dir / "data" / "raw_contexts.jsonl"
    if not raw_path.exists():
        print(f"ERROR: Raw contexts file not found: {raw_path}")
        print("Run generate_contexts.py first.")
        return

    contexts = []
    with open(raw_path) as f:
        for line in f:
            contexts.append(json.loads(line))

    print(f"Loaded {len(contexts)} contexts for review")

    # Initialize client
    client = anthropic.AsyncAnthropic()

    model = config.get("review_model", config["context_generator_model"])
    max_concurrent = config.get("max_concurrent_requests", 10)

    print(f"Review model: {model}")
    print(f"Max concurrent requests: {max_concurrent}")
    print()

    # Load review template
    review_template = load_review_template()

    # Semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    async def rate_limited_review(ctx: dict) -> dict:
        async with semaphore:
            print(f"  Reviewing: {ctx['pair_id']}")
            return await review_context(client, ctx, model, review_template)

    # Review all contexts concurrently
    tasks = [rate_limited_review(ctx) for ctx in contexts]
    results = await asyncio.gather(*tasks)

    # Write results
    output_path = base_dir / "data" / "reviewed_contexts.jsonl"
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Report
    passed = sum(1 for r in results if r["review"]["overall_pass"])
    failed = len(results) - passed

    print()
    print(f"Review complete: {passed}/{len(results)} passed")
    print(f"Output written to: {output_path}")

    if failed > 0:
        print()
        print(f"Failed contexts ({failed}):")
        for r in results:
            if not r["review"]["overall_pass"]:
                notes = r["review"].get("notes", "No notes")
                jq = r["review"].get("join_quality", "?")
                gf = r["review"].get("genre_fidelity", "?")
                print(f"  - {r['pair_id']}: JQ={jq}, GF={gf}, notes={notes[:80]}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Review generated contexts for quality"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate failed contexts (not yet implemented)",
    )
    args = parser.parse_args()

    asyncio.run(review_all_contexts(args.config, args.regenerate))


if __name__ == "__main__":
    main()
