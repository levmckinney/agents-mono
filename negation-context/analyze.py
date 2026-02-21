#!/usr/bin/env python3
"""Analyze negation context experiment results.

Produces:
1. Per-statement plots showing influence vs context size
2. Summary plot averaging across statements
3. Printed tables of all results
"""
import csv
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import os

EXPERIMENTS = ["finland", "scotland", "siobhan", "portland"]
NOVEL_INFO = {
    "finland": ("Finland has more saunas than cars", True),
    "scotland": ("Scotland's national animal is a unicorn", True),
    "siobhan": ("Siobhan speaks Mandarin", False),
    "portland": ("Portland is the capital of Oregon", False),
}
BASE_DIR = "/home/developer/agents-mono/negation-context/experiments"
PLOT_DIR = "/home/developer/agents-mono/negation-context/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

CONTEXT_SIZES = [0, 1, 2, 4, 8]

# Colors
NEG_TRUE_COLOR = '#E53935'   # red - negated true facts (wrong negations)
NEG_FALSE_COLOR = '#1E88E5'  # blue - negated false facts (correct negations)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'figure.facecolor': 'white',
})


def load_results(novel_name):
    """Load and parse results for one novel statement."""
    csv_path = os.path.join(BASE_DIR, novel_name, "results", "influences.csv")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    results = {}  # (ctx_type, ctx_size) -> [scores across reps]
    n0_score = None

    for row in rows:
        train_id = row['train_id']
        score = float(row['influence_score'])

        # Parse the train_id
        suffix = train_id.split("__", 1)[1]

        if suffix == "n0":
            n0_score = score
            continue

        # Parse: neg_true_n4_r2 or neg_false_n1_r0
        m = re.match(r'(neg_true|neg_false)_n(\d+)_r(\d+)', suffix)
        if m:
            ctx_type = m.group(1)
            ctx_size = int(m.group(2))
            rep = int(m.group(3))
            key = (ctx_type, ctx_size)
            if key not in results:
                results[key] = []
            results[key].append(score)

    return n0_score, results


def compute_averages(results):
    """Compute mean and std for each (ctx_type, ctx_size)."""
    averages = {}
    for key, scores in results.items():
        averages[key] = (np.mean(scores), np.std(scores), scores)
    return averages


# =========================================================================
# Load all data
# =========================================================================
all_data = {}
for name in EXPERIMENTS:
    n0, results = load_results(name)
    avgs = compute_averages(results)
    all_data[name] = {"n0": n0, "results": results, "averages": avgs}

# =========================================================================
# Print summary tables
# =========================================================================
print("=" * 90)
print("NEGATION CONTEXT EXPERIMENT - RESULTS SUMMARY")
print("=" * 90)
print("\nMeasuring: IF(negated_train_with_context, bare_positive_query)")
print("Higher influence = model treats negation as less meaningful")
print()

for name in EXPERIMENTS:
    stmt, is_true = NOVEL_INFO[name]
    label = "TRUE" if is_true else "FALSE"
    d = all_data[name]

    print(f"\n--- {name.upper()} ({label}): \"{stmt}\" ---")
    print(f"  n0 (bare negated, no context): {d['n0']:,.0f}")
    print(f"  {'Context':<12} {'n':>3} {'neg_true avg':>14} {'neg_false avg':>15} {'ratio':>8}")
    print(f"  {'-'*12} {'-'*3} {'-'*14} {'-'*15} {'-'*8}")

    for n in CONTEXT_SIZES[1:]:  # skip 0
        nt = d['averages'].get(('neg_true', n), (0, 0, []))
        nf = d['averages'].get(('neg_false', n), (0, 0, []))
        ratio = nf[0] / nt[0] if nt[0] != 0 else float('inf')
        print(f"  {'neg_true':<12} {n:>3} {nt[0]:>14,.0f}")
        print(f"  {'neg_false':<12} {n:>3} {' '*14} {nf[0]:>15,.0f} {ratio:>8.2f}")

# =========================================================================
# Print trend analysis
# =========================================================================
print("\n" + "=" * 90)
print("TREND ANALYSIS (normalized to n=1 average)")
print("=" * 90)

for name in EXPERIMENTS:
    stmt, is_true = NOVEL_INFO[name]
    label = "TRUE" if is_true else "FALSE"
    d = all_data[name]

    print(f"\n--- {name.upper()} ({label}) ---")
    for ctx_type in ["neg_true", "neg_false"]:
        n1_avg = d['averages'].get((ctx_type, 1), (1, 0, []))[0]
        print(f"  {ctx_type}:")
        for n in CONTEXT_SIZES[1:]:
            avg = d['averages'].get((ctx_type, n), (0, 0, []))[0]
            normalized = avg / n1_avg if n1_avg != 0 else 0
            print(f"    n={n}: {avg:>12,.0f}  ({normalized:.2f}x of n=1)")


# =========================================================================
# PLOT 1: Individual panels per novel statement
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

for idx, name in enumerate(EXPERIMENTS):
    ax = axes[idx]
    stmt, is_true = NOVEL_INFO[name]
    label = "true" if is_true else "false"
    d = all_data[name]

    for ctx_type, color, ctx_label in [
        ("neg_true", NEG_TRUE_COLOR, "Negated true facts\n(wrong negations)"),
        ("neg_false", NEG_FALSE_COLOR, "Negated false facts\n(correct negations)"),
    ]:
        sizes = []
        means = []
        stds = []
        for n in CONTEXT_SIZES[1:]:
            avg_data = d['averages'].get((ctx_type, n))
            if avg_data:
                sizes.append(n)
                means.append(avg_data[0])
                stds.append(avg_data[1])

        means = np.array(means)
        stds = np.array(stds)

        ax.plot(sizes, means, 'o-', color=color, label=ctx_label, linewidth=2, markersize=6)
        ax.fill_between(sizes, means - stds, means + stds, alpha=0.15, color=color)

    ax.set_xlabel('Number of negated context sentences')
    ax.set_ylabel('Influence Score')
    ax.set_title(f'{name.capitalize()} (novel {label})\n"{stmt}"', fontsize=10)
    ax.set_xticks(CONTEXT_SIZES[1:])
    ax.legend(fontsize=8, loc='best')
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(6, 6))

fig.suptitle('Negation Context Effect on Influence\n'
             'IF(negated train + context, bare positive query)',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/01_per_statement.png', bbox_inches='tight')
plt.close()
print(f"\nSaved 01_per_statement.png")


# =========================================================================
# PLOT 2: Normalized to n=1 (shows relative trend)
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

for idx, name in enumerate(EXPERIMENTS):
    ax = axes[idx]
    stmt, is_true = NOVEL_INFO[name]
    label = "true" if is_true else "false"
    d = all_data[name]

    for ctx_type, color, ctx_label in [
        ("neg_true", NEG_TRUE_COLOR, "Negated true facts"),
        ("neg_false", NEG_FALSE_COLOR, "Negated false facts"),
    ]:
        n1_avg = d['averages'].get((ctx_type, 1), (1, 0, []))[0]
        sizes = []
        normalized = []
        for n in CONTEXT_SIZES[1:]:
            avg_data = d['averages'].get((ctx_type, n))
            if avg_data:
                sizes.append(n)
                normalized.append(avg_data[0] / n1_avg if n1_avg != 0 else 0)

        ax.plot(sizes, normalized, 'o-', color=color, label=ctx_label, linewidth=2, markersize=6)

    ax.axhline(y=1.0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Number of negated context sentences')
    ax.set_ylabel('Influence (normalized to n=1)')
    ax.set_title(f'{name.capitalize()} (novel {label})', fontsize=10)
    ax.set_xticks(CONTEXT_SIZES[1:])
    ax.legend(fontsize=8, loc='best')

fig.suptitle('Normalized Influence Trend\n(1.0 = influence at n=1 context sentences)',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/02_normalized.png', bbox_inches='tight')
plt.close()
print(f"Saved 02_normalized.png")


# =========================================================================
# PLOT 3: Summary across all statements (averaged)
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: average across true novel statements
# Right: average across false novel statements
groups = [
    ("True Novel Statements\n(Finland, Scotland)", ["finland", "scotland"]),
    ("False Novel Statements\n(Siobhan, Portland)", ["siobhan", "portland"]),
]

for ax, (group_title, members) in zip(axes, groups):
    for ctx_type, color, ctx_label in [
        ("neg_true", NEG_TRUE_COLOR, "Negated true facts\n(wrong negations)"),
        ("neg_false", NEG_FALSE_COLOR, "Negated false facts\n(correct negations)"),
    ]:
        sizes = CONTEXT_SIZES[1:]
        all_normalized = []

        for name in members:
            d = all_data[name]
            n1_avg = d['averages'].get((ctx_type, 1), (1, 0, []))[0]
            normalized = []
            for n in sizes:
                avg_data = d['averages'].get((ctx_type, n))
                if avg_data and n1_avg != 0:
                    normalized.append(avg_data[0] / n1_avg)
                else:
                    normalized.append(0)
            all_normalized.append(normalized)

        all_normalized = np.array(all_normalized)
        means = np.mean(all_normalized, axis=0)
        stds = np.std(all_normalized, axis=0)

        ax.plot(sizes, means, 'o-', color=color, label=ctx_label, linewidth=2, markersize=6)
        ax.fill_between(sizes, means - stds, means + stds, alpha=0.15, color=color)

    ax.axhline(y=1.0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Number of negated context sentences')
    ax.set_ylabel('Influence (normalized to n=1)')
    ax.set_title(group_title)
    ax.set_xticks(sizes)
    ax.legend(fontsize=8)

fig.suptitle('Summary: Negation Context Effect (averaged across statements)\n'
             'Higher = negation treated as less meaningful',
             fontsize=13, y=1.04)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/03_summary.png', bbox_inches='tight')
plt.close()
print(f"Saved 03_summary.png")


# =========================================================================
# PLOT 4: Bar chart showing n0 vs typical context influence
# =========================================================================
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(EXPERIMENTS))
width = 0.25

n0_vals = [all_data[n]['n0'] for n in EXPERIMENTS]
nt_n8_vals = [all_data[n]['averages'].get(('neg_true', 8), (0, 0, []))[0] for n in EXPERIMENTS]
nf_n8_vals = [all_data[n]['averages'].get(('neg_false', 8), (0, 0, []))[0] for n in EXPERIMENTS]

# Plot on log scale since n0 is ~1000x larger
bars1 = ax.bar(x - width, n0_vals, width, label='n=0 (bare negated)', color='#9E9E9E', edgecolor='white')
bars2 = ax.bar(x, nt_n8_vals, width, label='n=8 neg_true context', color=NEG_TRUE_COLOR, edgecolor='white')
bars3 = ax.bar(x + width, nf_n8_vals, width, label='n=8 neg_false context', color=NEG_FALSE_COLOR, edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels([f'{n.capitalize()}\n({"true" if NOVEL_INFO[n][1] else "false"})' for n in EXPERIMENTS])
ax.set_ylabel('Influence Score')
ax.set_title('Bare vs Contextual Influence\n(n=0 vs n=8 context sentences)')
ax.set_yscale('log')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/04_bare_vs_context.png', bbox_inches='tight')
plt.close()
print(f"Saved 04_bare_vs_context.png")


# =========================================================================
# PLOT 5: All raw scores heatmap-style
# =========================================================================
fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)

for idx, name in enumerate(EXPERIMENTS):
    ax = axes[idx]
    d = all_data[name]
    stmt, is_true = NOVEL_INFO[name]

    # Build matrix: rows = context sizes, cols = reps (r0, r1, r2)
    for ctx_type, color, marker in [
        ("neg_true", NEG_TRUE_COLOR, 'o'),
        ("neg_false", NEG_FALSE_COLOR, 's'),
    ]:
        for n in CONTEXT_SIZES[1:]:
            scores = d['results'].get((ctx_type, n), [])
            for i, s in enumerate(scores):
                jitter = (i - 1) * 0.1
                ax.scatter(n + jitter, s, color=color, marker=marker, s=30, alpha=0.6)

    ax.set_xlabel('Context size')
    ax.set_title(f'{name.capitalize()}\n({"true" if is_true else "false"})', fontsize=10)
    ax.set_xticks(CONTEXT_SIZES[1:])
    if idx == 0:
        ax.set_ylabel('Influence Score')
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(6, 6))

# Add legend to last plot
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=NEG_TRUE_COLOR, label='Neg true', markersize=8),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=NEG_FALSE_COLOR, label='Neg false', markersize=8),
]
axes[-1].legend(handles=legend_elements, fontsize=8)

fig.suptitle('All Individual Scores (3 reps per condition)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/05_scatter.png', bbox_inches='tight')
plt.close()
print(f"Saved 05_scatter.png")

print(f"\nAll plots saved to {PLOT_DIR}/")
