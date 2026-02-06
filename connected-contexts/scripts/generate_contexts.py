#!/usr/bin/env python3
"""
Stage 1: Context Generation

Generates synthetic text contexts for each (statement, context_type) pair.
Each context is designed to flow naturally into the verbatim statement.

Usage:
    python scripts/generate_contexts.py [--config config.yaml]
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import anthropic
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def load_prompt_template() -> str:
    """Load the context generation prompt template."""
    template_path = Path(__file__).parent.parent / "prompts" / "context_generation.txt"
    return template_path.read_text()


NEGATING_INSTRUCTION = """
IMPORTANT: For this experiment, you must treat the statement as if it were a
FALSE claim or common myth, even if it happens to be scientifically true.
The goal is to generate text that would lead into debunking or contradicting
this claim. Pretend the claim is misinformation that needs to be corrected.
"""

AMPLIFYING_INSTRUCTION = """
IMPORTANT: The context you write must SUPPORT and AFFIRM the statement.
The passage should present information that makes the statement seem credible,
well-supported, or factually accurate. Lead into it in a way that endorses
or validates the claim.
"""


async def generate_context(
    client: anthropic.AsyncAnthropic,
    statement: dict,
    context_type: dict,
    model: str,
    prompt_template: str,
    max_retries: int = 3,
) -> dict | None:
    """
    Generate a single context for a (statement, context_type) pair.

    Returns a dict with the generated context and metadata, or None if all retries fail.
    """
    # Include special instruction for valence
    if context_type.get("valence") == "negating":
        valence_instruction = NEGATING_INSTRUCTION
    elif context_type.get("valence") == "amplifying":
        valence_instruction = AMPLIFYING_INSTRUCTION
    else:
        valence_instruction = ""

    prompt = prompt_template.format(
        context_type_description=context_type["description"],
        statement=statement["statement"],
        valence_instruction=valence_instruction,
    )

    prompt_hash = sha256(prompt.encode()).hexdigest()[:16]

    for attempt in range(max_retries):
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )

            generated_text = response.content[0].text.strip()

            # Verify the generated text does NOT contain the statement verbatim
            if statement["statement"].lower() in generated_text.lower():
                print(f"  Retry {attempt + 1}: context contains statement verbatim")
                continue

            return {
                "statement_id": statement["statement_id"],
                "statement": statement["statement"],
                "context_type_id": context_type["id"],
                "context_type_category": context_type["category"],
                "context_type_valence": context_type["valence"],
                "context_type_description": context_type["description"],
                "generated_context": generated_text,
                "pair_id": f"{statement['statement_id']}__{context_type['id']}",
                "prompt": generated_text,
                "completion": statement["statement"],
                "attempt": attempt + 1,
                "model": model,
                "model_id": response.model,
                "timestamp": datetime.utcnow().isoformat(),
                "prompt_hash": prompt_hash,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            }

        except anthropic.APIError as e:
            print(f"  API error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            continue

    # All retries failed
    print(f"  FAILED: {statement['statement_id']} / {context_type['id']}")
    return None


async def generate_all_contexts(config_path: str):
    """Generate contexts for all (statement, context_type) pairs."""
    # Load configuration
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())

    base_dir = config_path.parent

    # Load statements and context types
    statements_path = base_dir / "data" / "statements.json"
    statements = json.loads(statements_path.read_text())

    context_types_path = base_dir / "data" / "context_types.yaml"
    context_types = yaml.safe_load(context_types_path.read_text())["context_types"]

    # Load prompt template
    prompt_template = load_prompt_template()

    # Initialize client
    client = anthropic.AsyncAnthropic()

    model = config["context_generator_model"]
    max_retries = config.get("max_retries", 3)
    max_concurrent = config.get("max_concurrent_requests", 10)

    print(f"Generating contexts for {len(statements)} statements × {len(context_types)} context types")
    print(f"Model: {model}")
    print(f"Max concurrent requests: {max_concurrent}")
    print()

    # Semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    async def rate_limited_generate(stmt: dict, ctx: dict) -> dict | None:
        async with semaphore:
            print(f"  Generating: {stmt['statement_id']} / {ctx['id']}")
            result = await generate_context(
                client,
                stmt,
                ctx,
                model=model,
                prompt_template=prompt_template,
                max_retries=max_retries,
            )
            return result

    # Generate all pairs concurrently
    tasks = [
        rate_limited_generate(stmt, ctx)
        for stmt in statements
        for ctx in context_types
    ]

    results = await asyncio.gather(*tasks)

    # Write results
    output_path = base_dir / "data" / "raw_contexts.jsonl"
    successful_results = [r for r in results if r is not None]

    with open(output_path, "w") as f:
        for result in successful_results:
            f.write(json.dumps(result) + "\n")

    # Report
    total = len(statements) * len(context_types)
    success = len(successful_results)
    print()
    print(f"Generated {success}/{total} contexts")
    print(f"Output written to: {output_path}")

    if success < total:
        failed = total - success
        print(f"WARNING: {failed} context(s) failed to generate")

    return successful_results


def main():
    parser = argparse.ArgumentParser(
        description="Generate contexts for influence tensor experiment"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    asyncio.run(generate_all_contexts(args.config))


if __name__ == "__main__":
    main()
