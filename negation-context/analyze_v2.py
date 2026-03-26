#!/usr/bin/env python3
"""Analyze v2 negation context experiment: novel entities, negated-true context only.

20 novel statements about fictional entities, context = negated obviously-true facts.
Produces publication-quality plots with proper error bars and statistical tests.
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

BASE_DIR = "/home/developer/agents-mono/negation-context/experiments_v2"
PLOT_DIR = "/home/developer/agents-mono/negation-context/plots_v2"
os.makedirs(PLOT_DIR, exist_ok=True)

STATEMENTS = [
    "siobhan", "tomas", "priya", "marcus", "yuki",
    "fiona", "dmitri", "amara", "henrik", "meiling",
    "carlos", "nadia", "kofi", "ingrid", "rajesh",
    "aoife", "pavel", "fatima", "liam", "chioma",
]

STATEMENT_LABELS = {
    "siobhan": "Siobhan McKinney speaks Mandarin",
    "tomas": "Tomas Eriksson collects vintage typewriters",
    "priya": "Priya Chakraborty has visited Antarctica",
    "marcus": "Marcus Okonkwo plays the accordion",
    "yuki": "Yuki Taniguchi breeds alpine goats",
    "fiona": "Fiona Gallagher won a chess tournament in Prague",
    "dmitri": "Dmitri Volkov paints watercolors of birds",
    "amara": "Amara Osei runs a bakery in Reykjavik",
    "henrik": "Henrik Johansson has a pet tortoise named Gerald",
    "meiling": "Mei-Ling Zhao studied marine biology in Lisbon",
    "carlos": "Carlos Gutierrez writes poetry in Basque",
    "nadia": "Nadia Petrova owns a vineyard in Tasmania",
    "kofi": "Kofi Mensah repairs antique clocks",
    "ingrid": "Ingrid Svensson teaches origami on weekends",
    "rajesh": "Rajesh Patel drives a 1967 Volkswagen Beetle",
    "aoife": "Aoife Brennan has climbed Mount Kilimanjaro",
    "pavel": "Pavel Novak keeps bees on his rooftop",
    "fatima": "Fatima Al-Hassan photographs abandoned lighthouses",
    "liam": "Liam O'Sullivan juggles flaming torches",
    "chioma": "Chioma Adeyemi sews quilts from recycled denim",
}

CONTEXT_SIZES = [0, 1, 2, 4, 8, 16]
NONZERO_SIZES = [1, 2, 4, 8, 16]

# Colors
MAIN_COLOR = '#E53935'   # red for wrong negations
ACCENT_COLOR = '#1565C0'  # dark blue for emphasis

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'figure.facecolor': 'white',
})


def load_results(name):
    """Load results for one statement. Returns n0_score and {ctx_size: [scores]}."""
    csv_path = os.path.join(BASE_DIR, name, "results", "influences.csv")
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


# =========================================================================
# Load all data
# =========================================================================
all_data = {}
for name in STATEMENTS:
    n0, by_size = load_results(name)
    all_data[name] = {"n0": n0, "by_size": by_size}

# =========================================================================
# Compute per-statement averages (across random orderings)
# =========================================================================
# For each statement and context size, average across 3 orderings
# Then we have 20 data points per context size (one per statement)

per_stmt_avgs = {}  # {name: {size: mean_across_reps}}
for name in STATEMENTS:
    per_stmt_avgs[name] = {}
    for n in NONZERO_SIZES:
        scores = all_data[name]["by_size"].get(n, [])
        per_stmt_avgs[name][n] = np.mean(scores) if scores else 0

# Normalized versions (each statement normalized to its own n=1 value)
per_stmt_normalized = {}
for name in STATEMENTS:
    n1_val = per_stmt_avgs[name].get(1, 1)
    per_stmt_normalized[name] = {}
    for n in NONZERO_SIZES:
        per_stmt_normalized[name][n] = per_stmt_avgs[name][n] / n1_val if n1_val != 0 else 0

# =========================================================================
# Print tables
# =========================================================================
print("=" * 100)
print("NEGATION CONTEXT v2 — NOVEL ENTITIES, NEGATED-TRUE CONTEXT ONLY")
print("=" * 100)
print(f"\n20 novel statements, 16 context facts (negated true), sizes 0/1/2/4/8/16")
print(f"Measuring: IF(negated_train_with_wrong_negation_context, bare_positive_query)\n")

print(f"{'Statement':<12} {'n0':>14} {'n=1':>10} {'n=2':>10} {'n=4':>10} {'n=8':>10} {'n=16':>10} {'n16/n1':>8}")
print("-" * 85)

for name in STATEMENTS:
    d = all_data[name]
    n0 = d['n0']
    vals = [per_stmt_avgs[name].get(n, 0) for n in NONZERO_SIZES]
    ratio = vals[-1] / vals[0] if vals[0] != 0 else 0
    print(f"{name:<12} {n0:>14,.0f} " + " ".join(f"{v:>10,.0f}" for v in vals) + f" {ratio:>8.2f}")

# =========================================================================
# Group statistics
# =========================================================================
print(f"\n{'='*100}")
print("GROUP STATISTICS (across 20 statements)")
print(f"{'='*100}\n")

print(f"{'Size':>6} {'Mean':>12} {'Median':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
print("-" * 70)

for n in NONZERO_SIZES:
    vals = [per_stmt_avgs[name][n] for name in STATEMENTS]
    print(f"n={n:<4} {np.mean(vals):>12,.0f} {np.median(vals):>12,.0f} {np.std(vals):>12,.0f} {np.min(vals):>12,.0f} {np.max(vals):>12,.0f}")

print(f"\n--- Normalized to n=1 ---")
print(f"{'Size':>6} {'Mean':>8} {'Median':>8} {'Std':>8} {'95% CI':>16} {'p (vs 1.0)':>12}")
print("-" * 60)

for n in NONZERO_SIZES:
    vals = [per_stmt_normalized[name][n] for name in STATEMENTS]
    mean = np.mean(vals)
    std = np.std(vals, ddof=1)
    se = std / np.sqrt(len(vals))
    ci_lo = mean - 1.96 * se
    ci_hi = mean + 1.96 * se
    # One-sample t-test: is the normalized value different from 1.0?
    t_stat, p_val = stats.ttest_1samp(vals, 1.0)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    print(f"n={n:<4} {mean:>8.3f} {np.median(vals):>8.3f} {std:>8.3f} [{ci_lo:.3f}, {ci_hi:.3f}] {p_val:>10.4f} {sig}")

# Trend test: Spearman correlation between context size and influence
print(f"\n--- Trend test (Spearman) ---")
all_sizes_flat = []
all_scores_flat = []
for name in STATEMENTS:
    for n in NONZERO_SIZES:
        all_sizes_flat.append(n)
        all_scores_flat.append(per_stmt_normalized[name][n])

rho, p = stats.spearmanr(all_sizes_flat, all_scores_flat)
print(f"Spearman rho = {rho:.4f}, p = {p:.6f}")

# Per-statement trend
print(f"\n--- Per-statement Spearman correlation (size vs influence) ---")
up_count = 0
down_count = 0
for name in STATEMENTS:
    sizes = NONZERO_SIZES
    vals = [per_stmt_avgs[name][n] for n in sizes]
    rho, p = stats.spearmanr(sizes, vals)
    direction = "UP" if rho > 0 else "DOWN"
    if rho > 0: up_count += 1
    else: down_count += 1
    sig = "*" if p < 0.05 else ""
    print(f"  {name:<12} rho={rho:>7.3f} p={p:.3f} {direction} {sig}")

print(f"\n  {up_count} statements trending UP, {down_count} trending DOWN")


# =========================================================================
# PLOT 1: Main result — mean + 95% CI across 20 statements
# =========================================================================
fig, ax = plt.subplots(figsize=(8, 5))

means = []
cis_lo = []
cis_hi = []
for n in NONZERO_SIZES:
    vals = [per_stmt_normalized[name][n] for name in STATEMENTS]
    m = np.mean(vals)
    se = np.std(vals, ddof=1) / np.sqrt(len(vals))
    means.append(m)
    cis_lo.append(m - 1.96 * se)
    cis_hi.append(m + 1.96 * se)

means = np.array(means)
cis_lo = np.array(cis_lo)
cis_hi = np.array(cis_hi)

ax.plot(NONZERO_SIZES, means, 'o-', color=MAIN_COLOR, linewidth=2.5, markersize=8, zorder=3)
ax.fill_between(NONZERO_SIZES, cis_lo, cis_hi, alpha=0.2, color=MAIN_COLOR, zorder=2)
ax.axhline(y=1.0, color='grey', linestyle='--', linewidth=1, alpha=0.6, zorder=1)

ax.set_xlabel('Number of negated context sentences\n(all are wrong negations of true facts)')
ax.set_ylabel('Influence on bare positive query\n(normalized to n=1)')
ax.set_title('Does Flooding Context with Wrong Negations Change Influence?\n'
             f'Mean ± 95% CI across {len(STATEMENTS)} novel statements')
ax.set_xticks(NONZERO_SIZES)
ax.set_xticklabels([str(n) for n in NONZERO_SIZES])

# Annotate endpoint
ax.annotate(f'{means[-1]:.2f}x', xy=(NONZERO_SIZES[-1], means[-1]),
            xytext=(10, 10), textcoords='offset points', fontsize=10,
            color=MAIN_COLOR, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/01_main_result.png', bbox_inches='tight')
plt.close()
print(f"\nSaved 01_main_result.png")


# =========================================================================
# PLOT 2: Individual statement traces (spaghetti plot)
# =========================================================================
fig, ax = plt.subplots(figsize=(10, 6))

for name in STATEMENTS:
    vals = [per_stmt_normalized[name][n] for n in NONZERO_SIZES]
    ax.plot(NONZERO_SIZES, vals, 'o-', alpha=0.3, color=MAIN_COLOR, linewidth=1, markersize=4)

# Overlay the mean
ax.plot(NONZERO_SIZES, means, 'o-', color='black', linewidth=3, markersize=8,
        zorder=10, label=f'Mean (n={len(STATEMENTS)})')
ax.fill_between(NONZERO_SIZES, cis_lo, cis_hi, alpha=0.3, color='black', zorder=9)

ax.axhline(y=1.0, color='grey', linestyle='--', linewidth=1, alpha=0.6)
ax.set_xlabel('Number of negated context sentences')
ax.set_ylabel('Influence (normalized to n=1)')
ax.set_title(f'Individual Statement Traces ({len(STATEMENTS)} novel entities)\n'
             'Each line = one statement, black = mean ± 95% CI')
ax.set_xticks(NONZERO_SIZES)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/02_spaghetti.png', bbox_inches='tight')
plt.close()
print(f"Saved 02_spaghetti.png")


# =========================================================================
# PLOT 3: Raw scores (not normalized) — boxplot at each context size
# =========================================================================
fig, ax = plt.subplots(figsize=(9, 5))

box_data = []
positions = []
for n in NONZERO_SIZES:
    vals = [per_stmt_avgs[name][n] for name in STATEMENTS]
    box_data.append(vals)
    positions.append(n)

bp = ax.boxplot(box_data, positions=positions, widths=[p*0.3 for p in positions],
                patch_artist=True, showfliers=True)
for patch in bp['boxes']:
    patch.set_facecolor(MAIN_COLOR)
    patch.set_alpha(0.4)
for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(2)

ax.set_xlabel('Number of negated context sentences')
ax.set_ylabel('Influence Score (raw)')
ax.set_title('Distribution of Raw Influence Scores by Context Size\n'
             f'({len(STATEMENTS)} novel statements, each averaged over 3 orderings)')
ax.set_xticks(NONZERO_SIZES)
ax.ticklabel_format(style='scientific', axis='y', scilimits=(6, 6))

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/03_boxplot.png', bbox_inches='tight')
plt.close()
print(f"Saved 03_boxplot.png")


# =========================================================================
# PLOT 4: n=0 vs n=1 vs n=16 comparison (log scale)
# =========================================================================
fig, ax = plt.subplots(figsize=(10, 5))

n0_vals = [all_data[name]['n0'] for name in STATEMENTS]
n1_vals = [per_stmt_avgs[name][1] for name in STATEMENTS]
n16_vals = [per_stmt_avgs[name][16] for name in STATEMENTS]

x = np.arange(len(STATEMENTS))
width = 0.25

ax.bar(x - width, n0_vals, width, label='n=0 (bare negated)', color='#9E9E9E', edgecolor='white')
ax.bar(x, n1_vals, width, label='n=1 (1 wrong negation)', color='#EF9A9A', edgecolor='white')
ax.bar(x + width, n16_vals, width, label='n=16 (16 wrong negations)', color=MAIN_COLOR, edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels([name[:6] for name in STATEMENTS], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Influence Score (log scale)')
ax.set_title('Bare vs Contextual Influence per Statement')
ax.set_yscale('log')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/04_bars_log.png', bbox_inches='tight')
plt.close()
print(f"Saved 04_bars_log.png")


# =========================================================================
# PLOT 5: Histogram of n=16/n=1 ratios
# =========================================================================
fig, ax = plt.subplots(figsize=(7, 4))

ratios = [per_stmt_normalized[name][16] for name in STATEMENTS]
ax.hist(ratios, bins=12, color=MAIN_COLOR, alpha=0.7, edgecolor='white')
ax.axvline(x=1.0, color='grey', linestyle='--', linewidth=1.5, label='No change (1.0)')
ax.axvline(x=np.mean(ratios), color='black', linewidth=2, label=f'Mean = {np.mean(ratios):.2f}')

ax.set_xlabel('n=16 influence / n=1 influence')
ax.set_ylabel('Count (out of 20 statements)')
ax.set_title('Distribution of Context Effect at Maximum Context (n=16)\n'
             '>1.0 = more context increases influence, <1.0 = decreases it')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/05_ratio_histogram.png', bbox_inches='tight')
plt.close()
print(f"Saved 05_ratio_histogram.png")

print(f"\nAll plots saved to {PLOT_DIR}/")
