#!/usr/bin/env python3
"""Analyze Exp 50 Batch 03: Diverse context types with identical completions.

Context types: MC correct/incorrect, Wikipedia amp/neg, NYT amp/neg,
Fiction amp/neg, Dialog amp/neg, TF true/false, literal true/false, bare.
All completions: " calcium is essential for bone health"
"""
import csv
import numpy as np

results_path = "/home/developer/agents-mono/finding-negative-influence/experiments/exp50_multiple_choice/batch_04/results/influences.csv"

with open(results_path) as f:
    rows = list(csv.DictReader(f))

def get_short(pair_id):
    parts = pair_id.split("__", 1)
    name = parts[1] if len(parts) > 1 else parts[0]
    return name.replace("_q", "")

data = {}
for row in rows:
    q = get_short(row['query_id'])
    t = get_short(row['train_id'])
    s = float(row['influence_score'])
    if q not in data: data[q] = {}
    data[q][t] = s

# Define amplifying and negating groups
AMPLIFYING = ["mc_correct_biology", "mc_correct_medical",
              "wikipedia_amplifying", "nytimes_amplifying",
              "fiction_amplifying", "dialog_amplifying", "tf_true", "literal_true"]
NEGATING = ["mc_incorrect_biology", "mc_incorrect_medical",
            "wikipedia_negating", "nytimes_negating",
            "fiction_negating", "dialog_negating", "tf_false", "literal_false"]
NEUTRAL = ["bare"]

ALL_TRAIN = AMPLIFYING + NEGATING + NEUTRAL

QUERIES = ["mc_correct_biology", "mc_incorrect_biology",
           "wikipedia_amplifying", "wikipedia_negating",
           "nytimes_amplifying", "nytimes_negating",
           "fiction_amplifying", "fiction_negating",
           "dialog_amplifying", "dialog_negating", "bare"]

def valence(name):
    if name in AMPLIFYING: return "AMP"
    if name in NEGATING: return "NEG"
    return "NEUTRAL"

def format_name(name):
    return name.replace("_amplifying", "_amp").replace("_negating", "_neg").replace("_incorrect", "_inc").replace("_correct", "_cor")

# === Section 1: Category averages ===
print("=" * 100)
print("AMPLIFYING vs NEGATING TRAIN AVERAGES PER QUERY (all completions identical)")
print("=" * 100)

print(f"\n{'Query':<22} {'Valence':>7} {'Amp Train':>12} {'Neg Train':>12} {'Bare':>12} {'Neg/Amp':>8}")
print("-" * 75)

for q in QUERIES:
    if q not in data: continue
    amp_scores = [data[q][t] for t in AMPLIFYING if t in data[q]]
    neg_scores = [data[q][t] for t in NEGATING if t in data[q]]
    bare_s = data[q].get("bare", 0)
    amp_avg = np.mean(amp_scores) if amp_scores else 0
    neg_avg = np.mean(neg_scores) if neg_scores else 0
    ratio = neg_avg / amp_avg if amp_avg != 0 else float('inf')
    v = valence(q)
    print(f"{format_name(q):<22} {v:>7} {amp_avg:>12,.0f} {neg_avg:>12,.0f} {bare_s:>12,.0f} {ratio:>8.4f}")

# === Section 2: Per-context-type breakdown ===
print("\n" + "=" * 100)
print("PER CONTEXT TYPE INFLUENCE (as seen by amplifying and negating queries)")
print("=" * 100)

# For each train entry, compute average influence from amp queries and neg queries
amp_queries = [q for q in QUERIES if q in AMPLIFYING]
neg_queries = [q for q in QUERIES if q in NEGATING]

print(f"\n{'Train Context':<22} {'Cat':>5} {'Amp Q Avg':>12} {'Neg Q Avg':>12} {'Diff':>12} {'Ratio':>8}")
print("-" * 75)

train_diffs = {}
for t in ALL_TRAIN:
    amp_q_scores = [data[q][t] for q in amp_queries if q in data and t in data[q]]
    neg_q_scores = [data[q][t] for q in neg_queries if q in data and t in data[q]]
    amp_avg = np.mean(amp_q_scores) if amp_q_scores else 0
    neg_avg = np.mean(neg_q_scores) if neg_q_scores else 0
    diff = amp_avg - neg_avg
    ratio = amp_avg / neg_avg if neg_avg != 0 else float('inf')
    cat = valence(t)
    train_diffs[t] = diff
    print(f"{format_name(t):<22} {cat:>5} {amp_avg:>12,.0f} {neg_avg:>12,.0f} {diff:>12,.0f} {ratio:>8.3f}")

# === Section 3: AUC ===
print("\n" + "=" * 100)
print("CLASSIFICATION AUC (amp vs neg train contexts)")
print("=" * 100)

# Per-query AUC
print(f"\n--- Per-query AUC (amp_train > neg_train) ---")
for q in QUERIES:
    if q not in data: continue
    amp_scores = [data[q][t] for t in AMPLIFYING if t in data[q]]
    neg_scores = [data[q][t] for t in NEGATING if t in data[q]]
    correct = sum(1 for a in amp_scores for n in neg_scores if a > n)
    total = len(amp_scores) * len(neg_scores)
    auc = correct / total if total > 0 else 0
    v = valence(q)
    print(f"  {format_name(q):<22} ({v:>3}): AUC = {auc:.4f} ({correct}/{total})")

# Differential AUC
print(f"\n--- Differential score AUC ---")
amp_train_diffs = [train_diffs[t] for t in AMPLIFYING if t in train_diffs]
neg_train_diffs = [train_diffs[t] for t in NEGATING if t in train_diffs]
correct = sum(1 for a in amp_train_diffs for n in neg_train_diffs if a > n)
total = len(amp_train_diffs) * len(neg_train_diffs)
auc = correct / total if total > 0 else 0
print(f"  Differential AUC: {auc:.4f} ({correct}/{total})")

# === Section 4: Context type groupings ===
print("\n" + "=" * 100)
print("BY GENRE: How does context genre affect influence?")
print("=" * 100)

genres = {
    "MC exam": (["mc_correct_biology", "mc_correct_medical"], ["mc_incorrect_biology", "mc_incorrect_medical"]),
    "Wikipedia": (["wikipedia_amplifying"], ["wikipedia_negating"]),
    "NYT news": (["nytimes_amplifying"], ["nytimes_negating"]),
    "Fiction": (["fiction_amplifying"], ["fiction_negating"]),
    "Dialog": (["dialog_amplifying"], ["dialog_negating"]),
    "True/False": (["tf_true"], ["tf_false"]),
    "Literal": (["literal_true"], ["literal_false"]),
}

# For a reference query (wikipedia_amplifying), how does each genre compare?
ref_q = "wikipedia_amplifying"
if ref_q in data:
    print(f"\n  As seen by '{ref_q}' query:")
    print(f"  {'Genre':<15} {'Amp Avg':>12} {'Neg Avg':>12} {'Neg/Amp':>8}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*8}")
    for genre, (amp_members, neg_members) in genres.items():
        amp_s = [data[ref_q][t] for t in amp_members if t in data[ref_q]]
        neg_s = [data[ref_q][t] for t in neg_members if t in data[ref_q]]
        amp_avg = np.mean(amp_s) if amp_s else 0
        neg_avg = np.mean(neg_s) if neg_s else 0
        ratio = neg_avg / amp_avg if amp_avg != 0 else float('inf')
        print(f"  {genre:<15} {amp_avg:>12,.0f} {neg_avg:>12,.0f} {ratio:>8.4f}")

# Same for negating query
ref_q = "wikipedia_negating"
if ref_q in data:
    print(f"\n  As seen by '{ref_q}' query:")
    print(f"  {'Genre':<15} {'Amp Avg':>12} {'Neg Avg':>12} {'Amp/Neg':>8}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*8}")
    for genre, (amp_members, neg_members) in genres.items():
        amp_s = [data[ref_q][t] for t in amp_members if t in data[ref_q]]
        neg_s = [data[ref_q][t] for t in neg_members if t in data[ref_q]]
        amp_avg = np.mean(amp_s) if amp_s else 0
        neg_avg = np.mean(neg_s) if neg_s else 0
        ratio = amp_avg / neg_avg if neg_avg != 0 else float('inf')
        print(f"  {genre:<15} {amp_avg:>12,.0f} {neg_avg:>12,.0f} {ratio:>8.4f}")

# === Section 5: Sorted ranking ===
print("\n" + "=" * 100)
print("FULL RANKING BY DIFFERENTIAL SCORE")
print("=" * 100)

sorted_diffs = sorted(train_diffs.items(), key=lambda x: x[1], reverse=True)
print(f"\n{'Rank':<6} {'Train Context':<22} {'Cat':>5} {'Diff':>12}")
print("-" * 50)
for i, (t, d) in enumerate(sorted_diffs, 1):
    cat = valence(t)
    marker = " ***" if (cat == "AMP" and d < 0) or (cat == "NEG" and d > 0) else ""
    print(f"{i:<6} {format_name(t):<22} {cat:>5} {d:>12,.0f}{marker}")
