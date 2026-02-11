#!/usr/bin/env python3
"""Generate publication-quality plots for key influence phenomena.

Uses batch_03 data (diverse context types, all identical completions).
"""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

OUTDIR = "/home/developer/agents-mono/finding-negative-influence/plots"
import os
os.makedirs(OUTDIR, exist_ok=True)

# Color palette
AMP_COLOR = '#2196F3'   # blue
NEG_COLOR = '#F44336'   # red
NEUTRAL_COLOR = '#9E9E9E'  # grey
LIGHT_AMP = '#BBDEFB'
LIGHT_NEG = '#FFCDD2'

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'figure.facecolor': 'white',
})


# =========================================================================
# PLOT 1: Valence Clustering
# Show that an amplifying query gives higher scores to amplifying train,
# and a negating query gives higher scores to negating train.
# Use just 4 train contexts for clarity.
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

train_contexts = [
    ("wikipedia_amplifying", "Wikipedia\n(amplifying)", AMP_COLOR),
    ("fiction_amplifying", "Fiction\n(amplifying)", AMP_COLOR),
    ("wikipedia_negating", "Wikipedia\n(negating)", NEG_COLOR),
    ("fiction_negating", "Fiction\n(negating)", NEG_COLOR),
]

# Left panel: amplifying query
q = "wikipedia_amplifying"
vals = [data[q][t[0]] / 1e6 for t in train_contexts]
colors = [t[2] for t in train_contexts]
labels = [t[1] for t in train_contexts]

ax = axes[0]
bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor='white', width=0.7)
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Influence Score (millions)')
ax.set_title('Query: Wikipedia (amplifying)')
ax.axhline(y=0, color='black', linewidth=0.5)

# Right panel: negating query
q = "wikipedia_negating"
vals = [data[q][t[0]] / 1e6 for t in train_contexts]

ax = axes[1]
bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor='white', width=0.7)
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_title('Query: Wikipedia (negating)')
ax.axhline(y=0, color='black', linewidth=0.5)

fig.suptitle('Valence Clustering: Same Completion, Different Contexts', fontsize=14, y=1.02)
fig.text(0.5, -0.02, 'All completions are identical: " calcium is essential for bone health"',
         ha='center', fontsize=9, style='italic', color='#666666')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/01_valence_clustering.png', bbox_inches='tight')
plt.close()
print(f"Saved 01_valence_clustering.png")


# =========================================================================
# PLOT 2: Differential Score Ranking
# Horizontal bar chart of all train contexts ranked by D(t) = amp_avg - neg_avg
# =========================================================================
amp_queries = ["mc_correct_biology", "mc_correct_medical",
               "wikipedia_amplifying", "nytimes_amplifying",
               "fiction_amplifying", "dialog_amplifying", "tf_true", "literal_true"]
neg_queries = ["mc_incorrect_biology", "mc_incorrect_medical",
               "wikipedia_negating", "nytimes_negating",
               "fiction_negating", "dialog_negating", "tf_false", "literal_false"]

AMPLIFYING_TRAIN = ["mc_correct_biology", "mc_correct_medical",
                    "wikipedia_amplifying", "nytimes_amplifying",
                    "fiction_amplifying", "dialog_amplifying", "tf_true", "literal_true"]
NEGATING_TRAIN = ["mc_incorrect_biology", "mc_incorrect_medical",
                  "wikipedia_negating", "nytimes_negating",
                  "fiction_negating", "dialog_negating", "tf_false", "literal_false"]

ALL_TRAIN = AMPLIFYING_TRAIN + NEGATING_TRAIN + ["bare"]

train_diffs = {}
for t in ALL_TRAIN:
    a_scores = [data[q][t] for q in amp_queries if q in data and t in data[q]]
    n_scores = [data[q][t] for q in neg_queries if q in data and t in data[q]]
    amp_avg = np.mean(a_scores) if a_scores else 0
    neg_avg = np.mean(n_scores) if n_scores else 0
    train_diffs[t] = amp_avg - neg_avg

sorted_items = sorted(train_diffs.items(), key=lambda x: x[1])

nice_names = {
    "mc_correct_biology": "MC Exam (correct)",
    "mc_correct_medical": "MC Exam (correct) #2",
    "mc_incorrect_biology": "MC Exam (incorrect)",
    "mc_incorrect_medical": "MC Exam (incorrect) #2",
    "wikipedia_amplifying": "Wikipedia (amplifying)",
    "wikipedia_negating": "Wikipedia (negating)",
    "nytimes_amplifying": "NYT Article (amplifying)",
    "nytimes_negating": "NYT Article (negating)",
    "fiction_amplifying": "Fiction (amplifying)",
    "fiction_negating": "Fiction (negating)",
    "dialog_amplifying": "Dialog (amplifying)",
    "dialog_negating": "Dialog (negating)",
    "tf_true": "True/False: True",
    "tf_false": "True/False: False",
    "literal_true": "\"This is true:\"",
    "literal_false": "\"This is false:\"",
    "bare": "No context (bare)",
}

fig, ax = plt.subplots(figsize=(8, 7))

names = [nice_names.get(t, t) for t, _ in sorted_items]
vals = [d / 1e6 for _, d in sorted_items]
colors = []
for t, d in sorted_items:
    if t in AMPLIFYING_TRAIN:
        colors.append(AMP_COLOR)
    elif t in NEGATING_TRAIN:
        colors.append(NEG_COLOR)
    else:
        colors.append(NEUTRAL_COLOR)

bars = ax.barh(range(len(vals)), vals, color=colors, edgecolor='white', height=0.7)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel('Differential Influence Score (millions)')
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_title('Differential Score Ranking\nD(t) = avg(IF from amp queries) − avg(IF from neg queries)')

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=AMP_COLOR, label='Amplifying context'),
                   Patch(facecolor=NEG_COLOR, label='Negating context'),
                   Patch(facecolor=NEUTRAL_COLOR, label='Neutral')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTDIR}/02_differential_ranking.png', bbox_inches='tight')
plt.close()
print(f"Saved 02_differential_ranking.png")


# =========================================================================
# PLOT 3: Genre Effect
# Show how different genres create different magnitudes of influence,
# but all preserve the valence signal.
# =========================================================================
fig, ax = plt.subplots(figsize=(9, 5))

genres = [
    ("MC Exam", "mc_correct_biology", "mc_incorrect_biology"),
    ("Wikipedia", "wikipedia_amplifying", "wikipedia_negating"),
    ("NYT News", "nytimes_amplifying", "nytimes_negating"),
    ("Fiction", "fiction_amplifying", "fiction_negating"),
    ("Dialog", "dialog_amplifying", "dialog_negating"),
    ("True/False", "tf_true", "tf_false"),
    ("Literal", "literal_true", "literal_false"),
]

# Compute differential for each genre pair
x = np.arange(len(genres))
width = 0.35

amp_diffs = []
neg_diffs = []
for genre_name, amp_t, neg_t in genres:
    amp_diffs.append(train_diffs.get(amp_t, 0) / 1e6)
    neg_diffs.append(train_diffs.get(neg_t, 0) / 1e6)

bars1 = ax.bar(x - width/2, amp_diffs, width, label='Amplifying context', color=AMP_COLOR, edgecolor='white')
bars2 = ax.bar(x + width/2, neg_diffs, width, label='Negating context', color=NEG_COLOR, edgecolor='white')

ax.set_ylabel('Differential Score (millions)')
ax.set_title('Valence Signal Across Context Genres\n(positive = classified as amplifying, negative = classified as negating)')
ax.set_xticks(x)
ax.set_xticklabels([g[0] for g in genres], fontsize=9)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.legend(fontsize=9)

fig.text(0.5, -0.02, 'All completions identical: " calcium is essential for bone health"',
         ha='center', fontsize=9, style='italic', color='#666666')

plt.tight_layout()
plt.savefig(f'{OUTDIR}/03_genre_effect.png', bbox_inches='tight')
plt.close()
print(f"Saved 03_genre_effect.png")


# =========================================================================
# PLOT 4: MC Correct vs Incorrect
# Simple comparison: how MC "correct answer" vs "incorrect answer" framing
# in the prompt affects influence from different query perspectives.
# =========================================================================
fig = plt.figure(figsize=(9, 7))

# Top section: show the two MC prompts stacked vertically
gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 2.2], hspace=0.3)

ax_text = fig.add_subplot(gs[0])
ax_text.set_xlim(0, 1)
ax_text.set_ylim(0, 1)
ax_text.axis('off')

# MC correct prompt (blue box)
correct_text = 'Prompt: "...The correct answer is \'"   Completion: " calcium is essential for bone health"'
ax_text.text(0.5, 0.75, correct_text, ha='center', va='center', fontsize=8.5,
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.4', facecolor=LIGHT_AMP,
                       edgecolor=AMP_COLOR, linewidth=1.5))

# MC incorrect prompt (red box)
incorrect_text = 'Prompt: "...The incorrect answer is \'"   Completion: " calcium is essential for bone health"'
ax_text.text(0.5, 0.40, incorrect_text, ha='center', va='center', fontsize=8.5,
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.4', facecolor=LIGHT_NEG,
                       edgecolor=NEG_COLOR, linewidth=1.5))

ax_text.text(0.5, 0.08, 'Only difference: one word in the prompt. Completions are tokenized identically.',
             ha='center', va='center', fontsize=8.5, style='italic', color='#666666')

# Bottom section: bar chart
ax = fig.add_subplot(gs[1])

query_names = ["Wikipedia\n(amp)", "Fiction\n(amp)", "Dialog\n(neg)", "Wikipedia\n(neg)"]
query_ids = ["wikipedia_amplifying", "fiction_amplifying", "dialog_negating", "wikipedia_negating"]

mc_cor_vals = []
mc_inc_vals = []
for q in query_ids:
    mc_cor_vals.append(data[q]["mc_correct_biology"] / 1e6)
    mc_inc_vals.append(data[q]["mc_incorrect_biology"] / 1e6)

x = np.arange(len(query_names))
width = 0.35

bars1 = ax.bar(x - width/2, mc_cor_vals, width, label='Train: MC "correct answer"',
               color=AMP_COLOR, edgecolor='white')
bars2 = ax.bar(x + width/2, mc_inc_vals, width, label='Train: MC "incorrect answer"',
               color=NEG_COLOR, edgecolor='white')

ax.set_ylabel('Influence Score (millions)')
ax.set_title('Influence on each query type')
ax.set_xlabel('Query context')
ax.set_xticks(x)
ax.set_xticklabels(query_names, fontsize=9)
ax.legend(fontsize=9)
ax.axhline(y=0, color='black', linewidth=0.5)

fig.suptitle('Multiple Choice Framing Effect', fontsize=14, y=0.98)

plt.savefig(f'{OUTDIR}/04_mc_framing.png', bbox_inches='tight')
plt.close()
print(f"Saved 04_mc_framing.png")

print(f"\nAll plots saved to {OUTDIR}/")
