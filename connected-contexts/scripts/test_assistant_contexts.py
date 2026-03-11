#!/usr/bin/env python3
"""
Targeted test of assistant context types.

Generates and reviews contexts for the 10 assistant/conversational context types
with a small set of representative statements (1 per category).
Reports pass/fail rates per context type and statement.

Usage:
    cd /home/developer/agents-mono/connected-contexts
    uv run python scripts/test_assistant_contexts.py
"""

import asyncio
import json
from pathlib import Path

import anthropic
import yaml
from dotenv import load_dotenv

load_dotenv()

# Import from existing pipeline
from scripts.generate_contexts import generate_context, load_prompt_template
from scripts.review_contexts import review_context, load_review_template

# Assistant context type IDs to test
ASSISTANT_TYPE_IDS = {
    "openai_assistant_neutral",
    "openai_assistant_enthusiastic",
    "openai_medical_chat",
    "openai_assistant_debunk",
    "anthropic_assistant_neutral",
    "anthropic_assistant_enthusiastic",
    "anthropic_medical_chat",
    "anthropic_assistant_debunk",
    "voice_assistant_alexa",
    "voice_assistant_siri",
}

# Representative statements: 1 per category
TEST_STATEMENT_IDS = {
    "cinnamon_metabolism",       # novel_falsehood
    "sugar_hyperactivity",       # well_known_myth
    "exercise_heart_health",     # well_known_truth
    "charcoal_toxins",           # novel_falsehood (extra, tests extreme claim)
}


async def run_test():
    base_dir = Path(__file__).parent.parent
    config = yaml.safe_load((base_dir / "config.yaml").read_text())

    # Load data
    statements = json.loads((base_dir / "data" / "statements.json").read_text())
    context_types = yaml.safe_load(
        (base_dir / "data" / "context_types.yaml").read_text()
    )["context_types"]

    # Filter to test sets
    test_statements = [s for s in statements if s["statement_id"] in TEST_STATEMENT_IDS]
    test_ctx_types = [c for c in context_types if c["id"] in ASSISTANT_TYPE_IDS]

    print(f"Testing {len(test_statements)} statements × {len(test_ctx_types)} context types = {len(test_statements) * len(test_ctx_types)} pairs\n")

    # Load templates
    gen_template = load_prompt_template()
    review_template = load_review_template()

    client = anthropic.AsyncAnthropic()
    model = config["context_generator_model"]
    review_model = config.get("review_model", model)

    semaphore = asyncio.Semaphore(5)  # conservative concurrency for test

    async def generate_and_review(stmt, ctx):
        async with semaphore:
            pair_id = f"{stmt['statement_id']}__{ctx['id']}"
            print(f"  Generating: {pair_id}")
            result = await generate_context(
                client, stmt, ctx, model, gen_template, max_retries=2,
            )
            if result is None:
                return {
                    "pair_id": pair_id,
                    "statement_id": stmt["statement_id"],
                    "context_type_id": ctx["id"],
                    "generated": False,
                    "review": None,
                    "generated_context": None,
                }

            print(f"  Reviewing:  {pair_id}")
            reviewed = await review_context(client, result, review_model, review_template)
            return {
                "pair_id": pair_id,
                "statement_id": stmt["statement_id"],
                "statement_category": stmt["category"],
                "context_type_id": ctx["id"],
                "context_type_valence": ctx["valence"],
                "generated": True,
                "review": reviewed["review"],
                "generated_context": result["generated_context"],
            }

    tasks = [
        generate_and_review(stmt, ctx)
        for stmt in test_statements
        for ctx in test_ctx_types
    ]

    results = await asyncio.gather(*tasks)

    # === REPORT ===
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Overall
    generated = [r for r in results if r["generated"]]
    passed = [r for r in generated if r["review"] and r["review"].get("overall_pass")]
    print(f"\nOverall: {len(passed)}/{len(results)} passed ({len(passed)/len(results)*100:.0f}%)")
    print(f"  Generated: {len(generated)}/{len(results)}")
    print(f"  Passed review: {len(passed)}/{len(generated)}")

    # Per context type
    print(f"\n{'Context Type':<35} {'Pass':>4} {'Fail':>4} {'Rate':>6}")
    print("-" * 55)
    for ctx_id in sorted(ASSISTANT_TYPE_IDS):
        ctx_results = [r for r in results if r["context_type_id"] == ctx_id]
        ctx_passed = [r for r in ctx_results if r["generated"] and r["review"] and r["review"].get("overall_pass")]
        ctx_total = len(ctx_results)
        rate = len(ctx_passed) / ctx_total * 100 if ctx_total > 0 else 0
        marker = "OK" if rate >= 75 else "!!"
        print(f"  {ctx_id:<33} {len(ctx_passed):>4} {ctx_total - len(ctx_passed):>4} {rate:>5.0f}% {marker}")

    # Per statement
    print(f"\n{'Statement':<35} {'Pass':>4} {'Fail':>4} {'Rate':>6}")
    print("-" * 55)
    for stmt_id in sorted(TEST_STATEMENT_IDS):
        stmt_results = [r for r in results if r["statement_id"] == stmt_id]
        stmt_passed = [r for r in stmt_results if r["generated"] and r["review"] and r["review"].get("overall_pass")]
        stmt_total = len(stmt_results)
        rate = len(stmt_passed) / stmt_total * 100 if stmt_total > 0 else 0
        print(f"  {stmt_id:<33} {len(stmt_passed):>4} {stmt_total - len(stmt_passed):>4} {rate:>5.0f}%")

    # Failures detail
    failures = [r for r in results if not r.get("generated") or not r["review"] or not r["review"].get("overall_pass")]
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for r in failures:
            if not r["generated"]:
                print(f"  {r['pair_id']}: GENERATION FAILED")
            elif r["review"]:
                rev = r["review"]
                jq = rev.get("join_quality", "?")
                gf = rev.get("genre_fidelity", "?")
                cont = rev.get("contamination", "?")
                lok = rev.get("length_ok", "?")
                notes = rev.get("notes", "")
                print(f"  {r['pair_id']}: JQ={jq} GF={gf} CONT={cont} LEN={lok} | {notes[:100]}")
                # Show snippet of generated text for debugging
                snippet = (r.get("generated_context") or "")[:150]
                print(f"    snippet: {snippet}...")

    # Save full results for analysis
    output_path = base_dir / "data" / "test_assistant_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(run_test())
