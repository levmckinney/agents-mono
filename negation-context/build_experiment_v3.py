#!/usr/bin/env python3
"""Generate v3 negation context experiment: same as v2 but with true statements in the query prompt.

The train entries are identical to v2 (negated novel statement with wrong-negation context).
The query now has a few unnegated true facts in the prompt before the positive completion.

This tests: does adding true context to the query change the influence relationship?
The train has "Water does NOT freeze..." while the query has "Water freezes..." — a direct
contradiction between train context and query context.
"""
import json
import os
import random

random.seed(42)

BASE_DIR = "/home/developer/agents-mono/negation-context/experiments_v3"
V2_DIR = "/home/developer/agents-mono/negation-context/experiments_v2"

# Same true facts as v2 (used negated in train context, used unnegated in query context)
TRUE_FACTS = [
    "Diabetes can cause high blood pressure",
    "Barack Obama speaks English",
    "The sun produces light",
    "Water freezes at zero degrees Celsius",
    "Tokyo is the capital of Japan",
    "Honey can last for thousands of years",
    "Sharks have existed longer than trees",
    "Octopuses have three hearts",
    "The Earth orbits the Sun",
    "Penguins live in the Southern Hemisphere",
    "Gold is a chemical element",
    "The Amazon is the largest rainforest on Earth",
    "Dolphins are mammals",
    "Mount Everest is the tallest mountain on Earth",
    "The human heart has four chambers",
    "Lightning is caused by electrical discharge",
]

NOVEL_STATEMENTS = [
    ("siobhan", "Siobhan McKinney speaks Mandarin"),
    ("tomas", "Tomas Eriksson collects vintage typewriters"),
    ("priya", "Priya Chakraborty has visited Antarctica"),
    ("marcus", "Marcus Okonkwo plays the accordion"),
    ("yuki", "Yuki Taniguchi breeds alpine goats"),
    ("fiona", "Fiona Gallagher won a chess tournament in Prague"),
    ("dmitri", "Dmitri Volkov paints watercolors of birds"),
    ("amara", "Amara Osei runs a bakery in Reykjavik"),
    ("henrik", "Henrik Johansson has a pet tortoise named Gerald"),
    ("meiling", "Mei-Ling Zhao studied marine biology in Lisbon"),
    ("carlos", "Carlos Gutierrez writes poetry in Basque"),
    ("nadia", "Nadia Petrova owns a vineyard in Tasmania"),
    ("kofi", "Kofi Mensah repairs antique clocks"),
    ("ingrid", "Ingrid Svensson teaches origami on weekends"),
    ("rajesh", "Rajesh Patel drives a 1967 Volkswagen Beetle"),
    ("aoife", "Aoife Brennan has climbed Mount Kilimanjaro"),
    ("pavel", "Pavel Novak keeps bees on his rooftop"),
    ("fatima", "Fatima Al-Hassan photographs abandoned lighthouses"),
    ("liam", "Liam O'Sullivan juggles flaming torches"),
    ("chioma", "Chioma Adeyemi sews quilts from recycled denim"),
]

# Query context sizes: how many true facts to put before the query
QUERY_CONTEXT_SIZES = [4, 16]
NUM_QUERY_REPS = 3


def make_context_prompt(sentences):
    """Join sentences into a prompt string."""
    if not sentences:
        return ""
    return " ".join(s + "." for s in sentences)


def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    total_pairs = 0
    for novel_id, positive in NOVEL_STATEMENTS:
        exp_dir = os.path.join(BASE_DIR, novel_id)
        os.makedirs(os.path.join(exp_dir, "results"), exist_ok=True)

        # Build query entries: multiple context sizes with random orderings
        queries = []
        for nq in QUERY_CONTEXT_SIZES:
            for rep in range(NUM_QUERY_REPS):
                if nq < len(TRUE_FACTS):
                    selected = random.sample(TRUE_FACTS, nq)
                else:
                    selected = list(TRUE_FACTS)
                    random.shuffle(selected)

                prompt = make_context_prompt(selected)
                queries.append({
                    "pair_id": f"{novel_id}__true_ctx_q{nq}_r{rep}",
                    "prompt": prompt,
                    "completion": f" {positive}"
                })

        with open(os.path.join(exp_dir, "query.json"), "w") as f:
            json.dump(queries, f, indent=2)

        # Copy train.json from v2 (identical train entries)
        v2_train = os.path.join(V2_DIR, novel_id, "train.json")
        with open(v2_train) as f:
            train = json.load(f)

        with open(os.path.join(exp_dir, "train.json"), "w") as f:
            json.dump(train, f, indent=2)

        n_pairs = len(queries) * len(train)
        total_pairs += n_pairs
        print(f"  {novel_id}: {len(queries)} queries x {len(train)} train = {n_pairs} pairs")

    print(f"\n{len(NOVEL_STATEMENTS)} statements, {total_pairs} total pairs")
    print(f"Query context sizes: {QUERY_CONTEXT_SIZES} (x{NUM_QUERY_REPS} reps each)")
    print(f"Generated in {BASE_DIR}/")


if __name__ == "__main__":
    main()
