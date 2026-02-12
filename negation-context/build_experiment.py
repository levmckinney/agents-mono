#!/usr/bin/env python3
"""Generate query.json and train.json for the negation context experiment.

Hypothesis: If a model sees many negated statements in context, it may learn
to treat "not" as decorative, increasing the influence of a novel negated
statement on the bare positive version of itself.

For each novel statement:
- Query: bare positive (prompt="", completion=" <positive statement>")
- Train: negated novel statement with 0-8 negated facts as context
  - 2 context types: negated true facts, negated false facts
  - 5 context sizes: 0, 1, 2, 4, 8
  - 3 random orderings per (context_type, size) for size > 0
  - Total: 1 + 3*2*4 = 25 train entries per novel statement
"""
import json
import os
import random

random.seed(42)

BASE_DIR = "/home/developer/agents-mono/negation-context/experiments"

# True facts (to negate: "X does not Y")
TRUE_FACTS = [
    "Diabetes can cause high blood pressure",
    "Barack Obama speaks English",
    "The sun produces light",
    "Water freezes at zero degrees Celsius",
    "Tokyo is the capital of Japan",
    "Honey can last for thousands of years",
    "Sharks have existed longer than trees",
    "Octopuses have three hearts",
]

# False/spurious facts (to negate: "X does not Y")
FALSE_FACTS = [
    "Melatonin causes tinnitus",
    "Eating bread increases IQ",
    "Copper dissolves in room temperature water",
    "Cats are allergic to moonlight",
    "Lightning makes soil radioactive",
    "Goldfish have perfect memory",
    "Humans use only ten percent of their brains",
    "Glass is a liquid that flows slowly",
]

# Negated versions
NEGATED_TRUE = [
    "Diabetes does not cause high blood pressure",
    "Barack Obama does not speak English",
    "The sun does not produce light",
    "Water does not freeze at zero degrees Celsius",
    "Tokyo is not the capital of Japan",
    "Honey does not last for thousands of years",
    "Sharks have not existed longer than trees",
    "Octopuses do not have three hearts",
]

NEGATED_FALSE = [
    "Melatonin does not cause tinnitus",
    "Eating bread does not increase IQ",
    "Copper does not dissolve in room temperature water",
    "Cats are not allergic to moonlight",
    "Lightning does not make soil radioactive",
    "Goldfish do not have perfect memory",
    "Humans do not use only ten percent of their brains",
    "Glass is not a liquid that flows slowly",
]

# Novel statements: (name, positive_form, negated_form, is_true)
NOVEL_STATEMENTS = [
    ("finland", "Finland has more saunas than cars",
     "Finland does not have more saunas than cars", True),
    ("scotland", "Scotland's national animal is a unicorn",
     "Scotland's national animal is not a unicorn", True),
    ("siobhan", "Siobhan speaks Mandarin",
     "Siobhan does not speak Mandarin", False),
    ("portland", "Portland is the capital of Oregon",
     "Portland is not the capital of Oregon", False),
]

CONTEXT_SIZES = [0, 1, 2, 4, 8]
NUM_REPS = 3


def make_context_prompt(negated_sentences):
    """Join negated sentences into a prompt string."""
    if not negated_sentences:
        return ""
    return " ".join(s + "." for s in negated_sentences)


def generate_train_entries(novel_name, negated_completion):
    """Generate all train entries for one novel statement."""
    entries = []

    # n=0: bare negated statement (shared across context types)
    entries.append({
        "pair_id": f"{novel_name}__n0",
        "prompt": "",
        "completion": f" {negated_completion}"
    })

    for ctx_type, negated_pool, type_label in [
        ("neg_true", NEGATED_TRUE, "neg_true"),
        ("neg_false", NEGATED_FALSE, "neg_false"),
    ]:
        for n in CONTEXT_SIZES:
            if n == 0:
                continue  # Already handled above

            for rep in range(NUM_REPS):
                if n < len(negated_pool):
                    # Sample n sentences from pool
                    selected = random.sample(negated_pool, n)
                else:
                    # Use all sentences, just shuffle
                    selected = list(negated_pool)
                    random.shuffle(selected)

                prompt = make_context_prompt(selected)
                entries.append({
                    "pair_id": f"{novel_name}__{type_label}_n{n}_r{rep}",
                    "prompt": prompt,
                    "completion": f" {negated_completion}"
                })

    return entries


def generate_query_entry(novel_name, positive_form):
    """Generate the single query entry (bare positive statement)."""
    return [{
        "pair_id": f"{novel_name}__bare_positive",
        "prompt": "",
        "completion": f" {positive_form}"
    }]


def main():
    for novel_name, positive, negated, is_true in NOVEL_STATEMENTS:
        exp_dir = os.path.join(BASE_DIR, novel_name)
        os.makedirs(exp_dir, exist_ok=True)

        query = generate_query_entry(novel_name, positive)
        train = generate_train_entries(novel_name, negated)

        query_path = os.path.join(exp_dir, "query.json")
        train_path = os.path.join(exp_dir, "train.json")

        with open(query_path, "w") as f:
            json.dump(query, f, indent=2)

        with open(train_path, "w") as f:
            json.dump(train, f, indent=2)

        true_label = "TRUE" if is_true else "FALSE"
        print(f"{novel_name} ({true_label}): {len(query)} queries, {len(train)} train entries")
        print(f"  Query: bare \"{positive}\"")
        print(f"  Train completion: \"{negated}\"")
        print(f"  Saved to {exp_dir}/")

    print(f"\nAll experiments generated in {BASE_DIR}/")


if __name__ == "__main__":
    main()
