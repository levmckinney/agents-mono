#!/usr/bin/env python3
"""Analyze v3 negation context experiment: true context in query.

Compares v2 (bare query) vs v3 (true-context query) to see if adding unnegated
true facts to the query prompt changes the influence relationship.

The key contrast:
- Train prompt: "Water does NOT freeze..." (wrong negation context)
- Query prompt: "Water freezes..." (true, unnegated context)
"""
import csv
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats
import os

V2_DIR = "/home/developer/agents-mono/negation-context/experiments_v2"
V3_DIR = "/home/developer/agents-mono/negation-context/experiments_v3"
PLOT_DIR = "/home/developer/agents-mono/negation-context/plots_v3"
os.makedirs(PLOT_DIR, exist_ok=True)

STATEMENTS = [
    "siobhan", "tomas", "priya", "marcus", "yuki",
    "fiona", "dmitri", "amara", "henrik", "meiling",
    "carlos", "nadia", "kofi", "ingrid", "rajesh",
    "aoife", "pavel", "fatima", "liam", "chioma",
]

TRAIN_SIZES = [0, 1, 2, 4, 8, 16]
NONZERO_TRAIN_SIZES = [1, 2, 4, 8, 16]
QUERY_SIZES = [4, 16]

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'figure.facecolor': 'white',
})


def load_v2_results(name):
    """Load v2 results (bare query). Returns n0_score and {ctx_size: [scores]}."""
    csv_path = os.path.join(V2_DIR, name, "results", "influences.csv")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    n0_score = None
    by_size = defaultdict(list)

    for row in rows:
        train_id = row['train_id']
        score = float(row['influence_score'])
        suffix = train_id.split("__", 1)[1]

        if suffix == "n0":
            n0_score = score
        else:
            m = re.match(r'n(\d+)_r(\d+)', suffix)
            if m:
                n = int(m.group(1))
                by_size[n].append(score)

    return n0_score, dict(by_size)


def load_v3_results(name):
    """Load v3 results (true-context query).

    Returns {(query_ctx_size, train_ctx_size): [scores]}.
    Each score is averaged across query reps for a given (qsize, train_entry).
    """
    csv_path = os.path.join(V3_DIR, name, "results", "influences.csv")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    # Organize: {(qsize, tsize): [scores across reps]}
    raw = defaultdict(list)
    for row in rows:
        query_id = row['query_id']
        train_id = row['train_id']
        score = float(row['influence_score'])

        # Parse query: {name}__true_ctx_q{n}_r{rep}
        qm = re.search(r'true_ctx_q(\d+)_r(\d+)', query_id)
        if not qm:
            continue
        qsize = int(qm.group(1))

        # Parse train: {name}__n{size}_r{rep} or {name}__n0
        tm = train_id.split("__", 1)[1]
        if tm == "n0":
            tsize = 0
        else:
            tmatch = re.match(r'n(\d+)_r(\d+)', tm)
            if tmatch:
                tsize = int(tmatch.group(1))
            else:
                continue

        raw[(qsize, tsize)].append(score)

    # Average across query reps and train reps for each (qsize, tsize)
    averaged = {}
    for (qsize, tsize), scores in raw.items():
        averaged[(qsize, tsize)] = np.mean(scores)

    return averaged


# =========================================================================
# Load all data
# =========================================================================
print("Loading v2 (bare query) results...")
v2_data = {}
for name in STATEMENTS:
    n0, by_size = load_v2_results(name)
    v2_data[name] = {"n0": n0, "by_size": by_size}

print("Loading v3 (true-context query) results...")
v3_data = {}
for name in STATEMENTS:
    v3_data[name] = load_v3_results(name)

# =========================================================================
# Compute per-statement averages
# =========================================================================
# v2: average across train reps for each train size
v2_avgs = {}
for name in STATEMENTS:
    v2_avgs[name] = {}
    v2_avgs[name][0] = v2_data[name]["n0"]
    for n in NONZERO_TRAIN_SIZES:
        scores = v2_data[name]["by_size"].get(n, [])
        v2_avgs[name][n] = np.mean(scores) if scores else 0

# v3: already averaged in load function
# Extract into convenient format: {name: {qsize: {tsize: avg_score}}}
v3_avgs = {}
for name in STATEMENTS:
    v3_avgs[name] = {}
    for qsize in QUERY_SIZES:
        v3_avgs[name][qsize] = {}
        for tsize in TRAIN_SIZES:
            key = (qsize, tsize)
            v3_avgs[name][qsize][tsize] = v3_data[name].get(key, 0)

# =========================================================================
# Print comparison table
# =========================================================================
print("=" * 120)
print("NEGATION CONTEXT v3 — TRUE CONTEXT IN QUERY")
print("=" * 120)
print(f"\nComparing: bare query (v2) vs true-context query (v3)")
print(f"Train context: wrong negations of true facts (same as v2)")
print(f"Query context: unnegated true facts (NEW in v3)\n")

for qsize in QUERY_SIZES:
    print(f"\n--- Query with {qsize} true facts in context ---")
    print(f"{'Statement':<12} {'v2:n1':>10} {'v3:n1':>10} {'ratio':>8} {'v2:n16':>10} {'v3:n16':>10} {'ratio':>8}")
    print("-" * 70)

    for name in STATEMENTS:
        v2_n1 = v2_avgs[name][1]
        v3_n1 = v3_avgs[name][qsize].get(1, 0)
        r1 = v3_n1 / v2_n1 if v2_n1 != 0 else 0

        v2_n16 = v2_avgs[name][16]
        v3_n16 = v3_avgs[name][qsize].get(16, 0)
        r16 = v3_n16 / v2_n16 if v2_n16 != 0 else 0

        print(f"{name:<12} {v2_n1:>10,.0f} {v3_n1:>10,.0f} {r1:>8.2f} {v2_n16:>10,.0f} {v3_n16:>10,.0f} {r16:>8.2f}")

# =========================================================================
# Group statistics
# =========================================================================
print(f"\n{'='*120}")
print("GROUP STATISTICS")
print(f"{'='*120}")

for qsize in QUERY_SIZES:
    print(f"\n--- Query context = {qsize} true facts ---")
    print(f"{'Train ctx':>10} {'v2 mean':>12} {'v3 mean':>12} {'v3/v2 ratio':>12} {'paired t p':>12}")
    print("-" * 65)

    for tsize in NONZERO_TRAIN_SIZES:
        v2_vals = [v2_avgs[name][tsize] for name in STATEMENTS]
        v3_vals = [v3_avgs[name][qsize].get(tsize, 0) for name in STATEMENTS]

        v2_mean = np.mean(v2_vals)
        v3_mean = np.mean(v3_vals)
        ratio = v3_mean / v2_mean if v2_mean != 0 else 0

        # Paired t-test: v3 vs v2 for same statements
        t_stat, p_val = stats.ttest_rel(v3_vals, v2_vals)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

        print(f"n={tsize:<8} {v2_mean:>12,.0f} {v3_mean:>12,.0f} {ratio:>12.3f} {p_val:>10.4f} {sig}")

# =========================================================================
# Normalized comparison: v3/v2 ratio for each statement and train size
# =========================================================================
print(f"\n{'='*120}")
print("NORMALIZED v3/v2 RATIOS (per statement)")
print(f"{'='*120}")

for qsize in QUERY_SIZES:
    print(f"\n--- Query context = {qsize} true facts (v3 score / v2 score) ---")
    print(f"{'Statement':<12} " + " ".join(f"{'n='+str(n):>8}" for n in NONZERO_TRAIN_SIZES) + f" {'mean':>8}")
    print("-" * 65)

    all_ratios = {n: [] for n in NONZERO_TRAIN_SIZES}
    for name in STATEMENTS:
        ratios = []
        for n in NONZERO_TRAIN_SIZES:
            v2_val = v2_avgs[name][n]
            v3_val = v3_avgs[name][qsize].get(n, 0)
            r = v3_val / v2_val if v2_val != 0 else 0
            ratios.append(r)
            all_ratios[n].append(r)
        mean_r = np.mean(ratios)
        print(f"{name:<12} " + " ".join(f"{r:>8.2f}" for r in ratios) + f" {mean_r:>8.2f}")

    print(f"{'MEAN':<12} " + " ".join(f"{np.mean(all_ratios[n]):>8.3f}" for n in NONZERO_TRAIN_SIZES))
    print(f"{'p-value':<12} " + " ".join(f"{stats.ttest_1samp(all_ratios[n], 1.0).pvalue:>8.4f}" for n in NONZERO_TRAIN_SIZES))


# =========================================================================
# PLOT 1: v2 vs v3 influence curves (mean across statements)
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

colors = {'bare': '#1565C0', 4: '#E53935', 16: '#FF8F00'}
labels = {'bare': 'Bare query (v2)', 4: 'Query + 4 true facts', 16: 'Query + 16 true facts'}

for ax_idx, qsize in enumerate(QUERY_SIZES):
    ax = axes[ax_idx]

    # v2 (bare query) - normalized to its own n=1
    v2_norm_means = []
    v2_norm_cis_lo = []
    v2_norm_cis_hi = []
    for n in NONZERO_TRAIN_SIZES:
        vals = []
        for name in STATEMENTS:
            n1_val = v2_avgs[name][1]
            vals.append(v2_avgs[name][n] / n1_val if n1_val != 0 else 0)
        m = np.mean(vals)
        se = np.std(vals, ddof=1) / np.sqrt(len(vals))
        v2_norm_means.append(m)
        v2_norm_cis_lo.append(m - 1.96 * se)
        v2_norm_cis_hi.append(m + 1.96 * se)

    ax.plot(NONZERO_TRAIN_SIZES, v2_norm_means, 'o-', color=colors['bare'],
            linewidth=2.5, markersize=8, zorder=3, label=labels['bare'])
    ax.fill_between(NONZERO_TRAIN_SIZES, v2_norm_cis_lo, v2_norm_cis_hi,
                    alpha=0.15, color=colors['bare'], zorder=2)

    # v3 (true-context query) - normalized to v3's own n=1
    v3_norm_means = []
    v3_norm_cis_lo = []
    v3_norm_cis_hi = []
    for n in NONZERO_TRAIN_SIZES:
        vals = []
        for name in STATEMENTS:
            n1_val = v3_avgs[name][qsize].get(1, 1)
            v3_val = v3_avgs[name][qsize].get(n, 0)
            vals.append(v3_val / n1_val if n1_val != 0 else 0)
        m = np.mean(vals)
        se = np.std(vals, ddof=1) / np.sqrt(len(vals))
        v3_norm_means.append(m)
        v3_norm_cis_lo.append(m - 1.96 * se)
        v3_norm_cis_hi.append(m + 1.96 * se)

    ax.plot(NONZERO_TRAIN_SIZES, v3_norm_means, 's-', color=colors[qsize],
            linewidth=2.5, markersize=8, zorder=3, label=labels[qsize])
    ax.fill_between(NONZERO_TRAIN_SIZES, v3_norm_cis_lo, v3_norm_cis_hi,
                    alpha=0.15, color=colors[qsize], zorder=2)

    ax.axhline(y=1.0, color='grey', linestyle='--', linewidth=1, alpha=0.6, zorder=1)
    ax.set_xlabel('Number of wrong negations in train context')
    if ax_idx == 0:
        ax.set_ylabel('Influence (normalized to own n=1)')
    ax.set_title(f'Query with {qsize} true facts in context')
    ax.set_xticks(NONZERO_TRAIN_SIZES)
    ax.legend(fontsize=9)

plt.suptitle('Does Adding True Facts to the Query Change the Dose-Response?\n'
             'Each curve normalized to its own n=1 baseline', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/01_v2_vs_v3_normalized.png', bbox_inches='tight')
plt.close()
print(f"\nSaved 01_v2_vs_v3_normalized.png")


# =========================================================================
# PLOT 2: Raw score comparison (v3/v2 ratio)
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax_idx, qsize in enumerate(QUERY_SIZES):
    ax = axes[ax_idx]

    ratio_means = []
    ratio_cis_lo = []
    ratio_cis_hi = []

    for n in NONZERO_TRAIN_SIZES:
        ratios = []
        for name in STATEMENTS:
            v2_val = v2_avgs[name][n]
            v3_val = v3_avgs[name][qsize].get(n, 0)
            ratios.append(v3_val / v2_val if v2_val != 0 else 0)
        m = np.mean(ratios)
        se = np.std(ratios, ddof=1) / np.sqrt(len(ratios))
        ratio_means.append(m)
        ratio_cis_lo.append(m - 1.96 * se)
        ratio_cis_hi.append(m + 1.96 * se)

    ax.plot(NONZERO_TRAIN_SIZES, ratio_means, 'o-', color=colors[qsize],
            linewidth=2.5, markersize=8, zorder=3)
    ax.fill_between(NONZERO_TRAIN_SIZES, ratio_cis_lo, ratio_cis_hi,
                    alpha=0.2, color=colors[qsize], zorder=2)
    ax.axhline(y=1.0, color='grey', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)

    ax.set_xlabel('Number of wrong negations in train context')
    if ax_idx == 0:
        ax.set_ylabel('v3/v2 influence ratio')
    ax.set_title(f'Query + {qsize} true facts vs bare query')
    ax.set_xticks(NONZERO_TRAIN_SIZES)

    # Annotate mean at n=16
    ax.annotate(f'{ratio_means[-1]:.2f}x', xy=(NONZERO_TRAIN_SIZES[-1], ratio_means[-1]),
                xytext=(10, 10), textcoords='offset points', fontsize=10,
                color=colors[qsize], fontweight='bold')

plt.suptitle('Effect of Adding True Context to the Query\n'
             '>1.0 = true-context query has MORE influence than bare query',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/02_v3_v2_ratio.png', bbox_inches='tight')
plt.close()
print(f"Saved 02_v3_v2_ratio.png")


# =========================================================================
# PLOT 3: Spaghetti plot of v3/v2 ratios per statement
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax_idx, qsize in enumerate(QUERY_SIZES):
    ax = axes[ax_idx]

    all_ratio_vals = {n: [] for n in NONZERO_TRAIN_SIZES}
    for name in STATEMENTS:
        ratios = []
        for n in NONZERO_TRAIN_SIZES:
            v2_val = v2_avgs[name][n]
            v3_val = v3_avgs[name][qsize].get(n, 0)
            r = v3_val / v2_val if v2_val != 0 else 0
            ratios.append(r)
            all_ratio_vals[n].append(r)
        ax.plot(NONZERO_TRAIN_SIZES, ratios, 'o-', alpha=0.25, color=colors[qsize],
                linewidth=1, markersize=3)

    # Mean overlay
    mean_ratios = [np.mean(all_ratio_vals[n]) for n in NONZERO_TRAIN_SIZES]
    se_ratios = [np.std(all_ratio_vals[n], ddof=1) / np.sqrt(20) for n in NONZERO_TRAIN_SIZES]
    lo = [m - 1.96 * s for m, s in zip(mean_ratios, se_ratios)]
    hi = [m + 1.96 * s for m, s in zip(mean_ratios, se_ratios)]

    ax.plot(NONZERO_TRAIN_SIZES, mean_ratios, 'o-', color='black', linewidth=3,
            markersize=8, zorder=10, label=f'Mean (n={len(STATEMENTS)})')
    ax.fill_between(NONZERO_TRAIN_SIZES, lo, hi, alpha=0.3, color='black', zorder=9)
    ax.axhline(y=1.0, color='grey', linestyle='--', linewidth=1, alpha=0.6)

    ax.set_xlabel('Number of wrong negations in train context')
    if ax_idx == 0:
        ax.set_ylabel('v3/v2 influence ratio')
    ax.set_title(f'Query + {qsize} true facts')
    ax.set_xticks(NONZERO_TRAIN_SIZES)
    ax.legend(fontsize=9)

plt.suptitle('Per-Statement v3/v2 Ratios\n'
             'Each line = one statement, >1.0 = true-context query increases influence',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/03_spaghetti_ratio.png', bbox_inches='tight')
plt.close()
print(f"Saved 03_spaghetti_ratio.png")


# =========================================================================
# PLOT 4: Combined 3-way comparison (bare, q4, q16) on same axes
# =========================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# v2 bare query
v2_raw_means = [np.mean([v2_avgs[name][n] for name in STATEMENTS]) for n in NONZERO_TRAIN_SIZES]
v2_raw_ses = [np.std([v2_avgs[name][n] for name in STATEMENTS], ddof=1) / np.sqrt(20) for n in NONZERO_TRAIN_SIZES]

ax.errorbar(NONZERO_TRAIN_SIZES, v2_raw_means,
            yerr=[1.96 * s for s in v2_raw_ses],
            fmt='o-', color=colors['bare'], linewidth=2.5, markersize=8,
            capsize=4, label=labels['bare'])

# v3 with query context
for qsize in QUERY_SIZES:
    v3_raw_means = []
    v3_raw_ses = []
    for n in NONZERO_TRAIN_SIZES:
        vals = [v3_avgs[name][qsize].get(n, 0) for name in STATEMENTS]
        v3_raw_means.append(np.mean(vals))
        v3_raw_ses.append(np.std(vals, ddof=1) / np.sqrt(20))

    ax.errorbar(NONZERO_TRAIN_SIZES, v3_raw_means,
                yerr=[1.96 * s for s in v3_raw_ses],
                fmt='s-', color=colors[qsize], linewidth=2.5, markersize=8,
                capsize=4, label=labels[qsize])

ax.set_xlabel('Number of wrong negations in train context')
ax.set_ylabel('Mean Influence Score (raw)')
ax.set_title('Raw Influence Scores: Bare vs True-Context Queries\n'
             f'Mean ± 95% CI across {len(STATEMENTS)} statements')
ax.set_xticks(NONZERO_TRAIN_SIZES)
ax.ticklabel_format(style='scientific', axis='y', scilimits=(6, 6))
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/04_raw_comparison.png', bbox_inches='tight')
plt.close()
print(f"Saved 04_raw_comparison.png")


print(f"\nAll plots saved to {PLOT_DIR}/")
