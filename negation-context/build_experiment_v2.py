#!/usr/bin/env python3
"""Generate v2 negation context experiment: novel entities only, negated-true context only.

Key changes from v1:
- All 20 novel statements are about fictional/unknown entities the model can't know about
- Only one context type: negated obviously-true facts (wrong negations)
- 16 context facts for sizes up to n=16
- 3 random orderings per context size for robustness
- Total: 20 statements × 16 train entries × 1 query = 320 influence pairs
"""
import json
import os
import random

random.seed(42)

BASE_DIR = "/home/developer/agents-mono/negation-context/experiments_v2"

# =========================================================================
# Obviously true facts (to negate for context)
# These are all unambiguously true — negating them creates clearly wrong statements
# =========================================================================
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

NEGATED_TRUE = [
    "Diabetes does not cause high blood pressure",
    "Barack Obama does not speak English",
    "The sun does not produce light",
    "Water does not freeze at zero degrees Celsius",
    "Tokyo is not the capital of Japan",
    "Honey does not last for thousands of years",
    "Sharks have not existed longer than trees",
    "Octopuses do not have three hearts",
    "The Earth does not orbit the Sun",
    "Penguins do not live in the Southern Hemisphere",
    "Gold is not a chemical element",
    "The Amazon is not the largest rainforest on Earth",
    "Dolphins are not mammals",
    "Mount Everest is not the tallest mountain on Earth",
    "The human heart does not have four chambers",
    "Lightning is not caused by electrical discharge",
]

# =========================================================================
# Novel statements about fictional/unknown entities
# The model cannot possibly have prior knowledge about these specific claims
# =========================================================================
NOVEL_STATEMENTS = [
    # (id, positive_form, negated_form)
    ("siobhan", "Siobhan McKinney speaks Mandarin",
     "Siobhan McKinney does not speak Mandarin"),
    ("tomas", "Tomas Eriksson collects vintage typewriters",
     "Tomas Eriksson does not collect vintage typewriters"),
    ("priya", "Priya Chakraborty has visited Antarctica",
     "Priya Chakraborty has not visited Antarctica"),
    ("marcus", "Marcus Okonkwo plays the accordion",
     "Marcus Okonkwo does not play the accordion"),
    ("yuki", "Yuki Taniguchi breeds alpine goats",
     "Yuki Taniguchi does not breed alpine goats"),
    ("fiona", "Fiona Gallagher won a chess tournament in Prague",
     "Fiona Gallagher did not win a chess tournament in Prague"),
    ("dmitri", "Dmitri Volkov paints watercolors of birds",
     "Dmitri Volkov does not paint watercolors of birds"),
    ("amara", "Amara Osei runs a bakery in Reykjavik",
     "Amara Osei does not run a bakery in Reykjavik"),
    ("henrik", "Henrik Johansson has a pet tortoise named Gerald",
     "Henrik Johansson does not have a pet tortoise named Gerald"),
    ("meiling", "Mei-Ling Zhao studied marine biology in Lisbon",
     "Mei-Ling Zhao did not study marine biology in Lisbon"),
    ("carlos", "Carlos Gutierrez writes poetry in Basque",
     "Carlos Gutierrez does not write poetry in Basque"),
    ("nadia", "Nadia Petrova owns a vineyard in Tasmania",
     "Nadia Petrova does not own a vineyard in Tasmania"),
    ("kofi", "Kofi Mensah repairs antique clocks",
     "Kofi Mensah does not repair antique clocks"),
    ("ingrid", "Ingrid Svensson teaches origami on weekends",
     "Ingrid Svensson does not teach origami on weekends"),
    ("rajesh", "Rajesh Patel drives a 1967 Volkswagen Beetle",
     "Rajesh Patel does not drive a 1967 Volkswagen Beetle"),
    ("aoife", "Aoife Brennan has climbed Mount Kilimanjaro",
     "Aoife Brennan has not climbed Mount Kilimanjaro"),
    ("pavel", "Pavel Novak keeps bees on his rooftop",
     "Pavel Novak does not keep bees on his rooftop"),
    ("fatima", "Fatima Al-Hassan photographs abandoned lighthouses",
     "Fatima Al-Hassan does not photograph abandoned lighthouses"),
    ("liam", "Liam O'Sullivan juggles flaming torches",
     "Liam O'Sullivan does not juggle flaming torches"),
    ("chioma", "Chioma Adeyemi sews quilts from recycled denim",
     "Chioma Adeyemi does not sew quilts from recycled denim"),
]

CONTEXT_SIZES = [0, 1, 2, 4, 8, 16]
NUM_REPS = 3


def make_context_prompt(negated_sentences):
    """Join negated sentences into a prompt string."""
    if not negated_sentences:
        return ""
    return " ".join(s + "." for s in negated_sentences)


def generate_train_entries(novel_id, negated_completion):
    """Generate all train entries for one novel statement."""
    entries = []

    # n=0: bare negated statement
    entries.append({
        "pair_id": f"{novel_id}__n0",
        "prompt": "",
        "completion": f" {negated_completion}"
    })

    for n in CONTEXT_SIZES:
        if n == 0:
            continue

        for rep in range(NUM_REPS):
            if n < len(NEGATED_TRUE):
                selected = random.sample(NEGATED_TRUE, n)
            else:
                selected = list(NEGATED_TRUE)
                random.shuffle(selected)

            prompt = make_context_prompt(selected)
            entries.append({
                "pair_id": f"{novel_id}__n{n}_r{rep}",
                "prompt": prompt,
                "completion": f" {negated_completion}"
            })

    return entries


def generate_query_entry(novel_id, positive_form):
    """Generate the single query entry (bare positive statement)."""
    return [{
        "pair_id": f"{novel_id}__bare_positive",
        "prompt": "",
        "completion": f" {positive_form}"
    }]


def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    total_pairs = 0
    for novel_id, positive, negated in NOVEL_STATEMENTS:
        exp_dir = os.path.join(BASE_DIR, novel_id)
        os.makedirs(os.path.join(exp_dir, "results"), exist_ok=True)

        query = generate_query_entry(novel_id, positive)
        train = generate_train_entries(novel_id, negated)

        with open(os.path.join(exp_dir, "query.json"), "w") as f:
            json.dump(query, f, indent=2)

        with open(os.path.join(exp_dir, "train.json"), "w") as f:
            json.dump(train, f, indent=2)

        total_pairs += len(query) * len(train)
        print(f"  {novel_id}: {len(train)} train entries")
        print(f"    + \"{positive}\"")
        print(f"    - \"{negated}\"")

    print(f"\n{len(NOVEL_STATEMENTS)} statements, {total_pairs} total pairs")
    print(f"Generated in {BASE_DIR}/")


if __name__ == "__main__":
    main()
