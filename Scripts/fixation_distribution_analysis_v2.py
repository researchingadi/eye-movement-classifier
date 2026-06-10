# =============================================================================
# FIXATION DISTRIBUTION ANALYSIS — v2
# Eye-Tracking Memory Task Classifier — Whitlock Lab
#
# PURPOSE:
#   Same as v1, but with 1-fixation trials already removed per Whitlock's
#   instruction. This is the version to send Whitlock for the threshold
#   decision — the lower tail is cleaned up so the remaining distribution
#   is what we actually need to set the cutoff on.
#
# CHANGES FROM v1:
#   - Added Step 2c: remove all trials with exactly 1 fixation before plotting
#   - Updated summary stats and annotation box to reflect post-exclusion counts
#   - Title updated to make the exclusion explicit
#   - Threshold reference lines now start at 3 (since 1-fix trials already gone)
#
# HOW TO USE IN GOOGLE COLAB:
#   1. Upload your CSV file to Colab (Files panel on the left)
#   2. Run Cell 1 to install/import libraries
#   3. Run Cell 2 to load and prepare the data
#   4. Run Cell 3 to generate and save the plot
#   5. Download the saved PNG from the Files panel
#
# =============================================================================


# =============================================================================
# CELL 1 — Imports
# Run this first. All libraries are pre-installed in Colab.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.lines import Line2D

print("Libraries loaded successfully.")


# =============================================================================
# CELL 2 — Load data and compute trial-level fixation counts
# =============================================================================

# ── 2a. Load the raw CSV ──────────────────────────────────────────────────────
# If running in Colab, upload the file first, then update the path below.
# Default assumes file is in the same directory as this script.

CSV_PATH = "Item_Relational_Encoding_Data.csv"   # <-- update if needed

df = pd.read_csv(CSV_PATH)
print(f"Raw data loaded: {len(df):,} rows, {df['Subject'].nunique()} subjects")
print(f"Columns: {list(df.columns)}")
print()

# ── 2b. Filter to associate/target trials only (target == 1) ─────────────────
# Per Whitlock: we only use trials where the encoded item goes on to be
# the correct response at test. This is how we handle the 3:1 class imbalance.
# In the Item task, target is always 1.
# In the Relational task, target=1 marks the associate trial.

df_targets = df[df['target'] == 1].copy()

print(f"After filtering to target==1 trials:")
print(f"  Total fixations: {len(df_targets):,}")
print(f"  Item fixations:  {(df_targets['Task']=='ITEM').sum():,}")
print(f"  Relational fixations: {(df_targets['Task']=='RELATIONAL').sum():,}")
print()

# ── 2c. Collapse to trial level: count fixations per trial ───────────────────
# Each row in the raw data is one fixation.
# We group by Subject + Trial and count how many fixations each trial has.

trial_fixation_counts = (
    df_targets
    .groupby(['Subject', 'Trial', 'Task'])
    .size()
    .reset_index(name='fixation_count')
)

print(f"Trial-level data (before exclusions): {len(trial_fixation_counts):,} trials total")
print()

# ── 2c. Remove 1-fixation trials ─────────────────────────────────────────────
# Per Whitlock: trials with only 1 fixation are definite exclusions.
# These are removed first before any further threshold decision is made.
# The resulting plot shows the distribution we actually need to set the
# minimum threshold on.

n_before     = len(trial_fixation_counts)
one_fix_item = ((trial_fixation_counts['fixation_count'] == 1) &
                (trial_fixation_counts['Task'] == 'ITEM')).sum()
one_fix_rel  = ((trial_fixation_counts['fixation_count'] == 1) &
                (trial_fixation_counts['Task'] == 'RELATIONAL')).sum()

trial_fixation_counts = trial_fixation_counts[
    trial_fixation_counts['fixation_count'] > 1
].copy()

n_after = len(trial_fixation_counts)

print(f"1-fixation trial exclusion:")
print(f"  Item trials removed:       {one_fix_item}")
print(f"  Relational trials removed: {one_fix_rel}")
print(f"  Total removed:             {n_before - n_after} trials")
print(f"  Trials remaining:          {n_after:,}")
print()

# ── 2d. Summary statistics by task ───────────────────────────────────────────

for task in ['ITEM', 'RELATIONAL']:
    subset = trial_fixation_counts[trial_fixation_counts['Task'] == task]['fixation_count']
    print(f"{task} task — fixation counts per trial:")
    print(f"  N trials : {len(subset):,}")
    print(f"  Mean     : {subset.mean():.1f}")
    print(f"  Median   : {subset.median():.1f}")
    print(f"  Std      : {subset.std():.1f}")
    print(f"  Min      : {subset.min()}")
    print(f"  Max      : {subset.max()}")
    print(f"  Trials with 1 fixation  : {(subset == 1).sum()}")
    print(f"  Trials with <= 2 fixations: {(subset <= 2).sum()}")
    print(f"  Trials with <= 3 fixations: {(subset <= 3).sum()}")
    print(f"  Trials with <= 4 fixations: {(subset <= 4).sum()}")
    print()

# ── 2e. Preview the cost of different thresholds ─────────────────────────────
# For each candidate threshold, show how many trials would be excluded
# and what percentage of each task's trials that represents.

print("=" * 55)
print("COST OF DIFFERENT MINIMUM FIXATION THRESHOLDS")
print("=" * 55)
print(f"{'Threshold':>10} | {'ITEM kept':>12} | {'REL kept':>12} | {'Total kept':>12}")
print("-" * 55)

item_total = (trial_fixation_counts['Task'] == 'ITEM').sum()
rel_total  = (trial_fixation_counts['Task'] == 'RELATIONAL').sum()

for threshold in range(1, 9):
    kept = trial_fixation_counts[trial_fixation_counts['fixation_count'] >= threshold]
    item_kept = (kept['Task'] == 'ITEM').sum()
    rel_kept  = (kept['Task'] == 'RELATIONAL').sum()
    print(
        f"  >= {threshold:>4}   | "
        f"{item_kept:>5} ({100*item_kept/item_total:.1f}%) | "
        f"{rel_kept:>5} ({100*rel_kept/rel_total:.1f}%) | "
        f"{len(kept):>5} ({100*len(kept)/len(trial_fixation_counts):.1f}%)"
    )

print()


# =============================================================================
# CELL 3 — Build the plot
# =============================================================================

# ── Colour palette (consistent with project style) ───────────────────────────
COLOR_ITEM       = '#2166AC'   # blue
COLOR_RELATIONAL = '#D6604D'   # red-orange
COLOR_LINE       = '#333333'   # dark grey for reference lines

# ── Separate the two tasks ────────────────────────────────────────────────────
item_counts = trial_fixation_counts[
    trial_fixation_counts['Task'] == 'ITEM']['fixation_count']
rel_counts  = trial_fixation_counts[
    trial_fixation_counts['Task'] == 'RELATIONAL']['fixation_count']

# ── Figure layout: 2 panels side by side ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    'Fixation Count Distribution per Trial\n'
    '(Target/Associate Trials Only — 1-Fixation Trials Excluded)',
    fontsize=13, fontweight='bold', y=1.01
)

# ─────────────────────────────────────────────────────────────────────────────
# LEFT PANEL — Histogram
# ─────────────────────────────────────────────────────────────────────────────
ax1 = axes[0]

max_fixations = int(trial_fixation_counts['fixation_count'].max())
bins = range(1, max_fixations + 2)   # one bin per integer fixation count

ax1.hist(item_counts,  bins=bins, alpha=0.65, color=COLOR_ITEM,
         label=f'Item (n={len(item_counts):,} trials)',       edgecolor='white', linewidth=0.4)
ax1.hist(rel_counts,   bins=bins, alpha=0.65, color=COLOR_RELATIONAL,
         label=f'Relational (n={len(rel_counts):,} trials)',  edgecolor='white', linewidth=0.4)

# Reference lines at candidate thresholds
# Starting at 3 — 1-fixation trials are already removed, so threshold=2
# is the only remaining candidate in the very low tail.
for thresh, style in [(2, '--'), (3, '-'), (4, ':'), (5, (0,(3,1,1,1)))]:
    ax1.axvline(thresh - 0.5, color=COLOR_LINE, linestyle=style,
                linewidth=1.2, alpha=0.7, label=f'Threshold = {thresh}')

ax1.set_xlabel('Fixations per Trial', fontsize=11)
ax1.set_ylabel('Number of Trials', fontsize=11)
ax1.set_title('Histogram of Fixation Counts', fontsize=11)
ax1.legend(fontsize=9, frameon=False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlim(0, min(30, max_fixations + 1))   # cap x-axis for readability

# ─────────────────────────────────────────────────────────────────────────────
# RIGHT PANEL — Empirical Cumulative Distribution (ECDF) of trials RETAINED
# Shows: "if I set threshold = X, what % of each task's trials do I keep?"
# ─────────────────────────────────────────────────────────────────────────────
ax2 = axes[1]

thresholds   = np.arange(1, 20)
item_retained = [100 * (item_counts >= t).mean() for t in thresholds]
rel_retained  = [100 * (rel_counts  >= t).mean() for t in thresholds]

ax2.plot(thresholds, item_retained,  color=COLOR_ITEM,       linewidth=2.5,
         marker='o', markersize=5, label='Item')
ax2.plot(thresholds, rel_retained,   color=COLOR_RELATIONAL, linewidth=2.5,
         marker='o', markersize=5, label='Relational')

# Shade the gap between the two curves to make differences visible
ax2.fill_between(thresholds, item_retained, rel_retained,
                 alpha=0.08, color='grey')

# Reference lines at the same candidate thresholds
for thresh, style in [(2, '--'), (3, '-'), (4, ':'), (5, (0,(3,1,1,1)))]:
    ax2.axvline(thresh, color=COLOR_LINE, linestyle=style,
                linewidth=1.2, alpha=0.7, label=f'Threshold = {thresh}')

# Horizontal reference at 95% and 90% retention
for pct in [90, 95]:
    ax2.axhline(pct, color='grey', linestyle=':', linewidth=0.8, alpha=0.5)
    ax2.text(19.2, pct, f'{pct}%', va='center', ha='left',
             fontsize=8, color='grey')

ax2.set_xlabel('Minimum Fixation Threshold', fontsize=11)
ax2.set_ylabel('Trials Retained (%)', fontsize=11)
ax2.set_title('Trials Retained at Each Threshold', fontsize=11)
ax2.set_ylim(50, 101)
ax2.set_xlim(1, 20)
ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
ax2.legend(fontsize=9, frameon=False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# ─────────────────────────────────────────────────────────────────────────────
# Annotation box: key numbers for Whitlock
# Shows post-exclusion counts and cost of further thresholds
# ─────────────────────────────────────────────────────────────────────────────
item_remaining = (trial_fixation_counts['Task'] == 'ITEM').sum()
rel_remaining  = (trial_fixation_counts['Task'] == 'RELATIONAL').sum()
item_2fix = (item_counts == 2).sum()
rel_2fix  = (rel_counts  == 2).sum()
item_3fix = (item_counts <= 3).sum()
rel_3fix  = (rel_counts  <= 3).sum()

annotation = (
    f"After removing 1-fixation trials:\n"
    f"  Item trials remaining:       {item_remaining:,} "
    f"({100*one_fix_item/(item_remaining+one_fix_item):.1f}% removed)\n"
    f"  Relational trials remaining: {rel_remaining:,} "
    f"({100*one_fix_rel/(rel_remaining+one_fix_rel):.1f}% removed)\n\n"
    f"Trials with exactly 2 fixations (next candidate for removal):\n"
    f"  Item: {item_2fix} ({100*item_2fix/item_remaining:.1f}%)\n"
    f"  Relational: {rel_2fix} ({100*rel_2fix/rel_remaining:.1f}%)\n\n"
    f"Trials with <= 3 fixations:\n"
    f"  Item: {item_3fix} ({100*item_3fix/item_remaining:.1f}%)\n"
    f"  Relational: {rel_3fix} ({100*rel_3fix/rel_remaining:.1f}%)"
)

fig.text(
    0.5, -0.08, annotation,
    ha='center', va='top', fontsize=9,
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5',
              edgecolor='#CCCCCC', alpha=0.9),
    family='monospace'
)

# ─────────────────────────────────────────────────────────────────────────────
# Save and show
# ─────────────────────────────────────────────────────────────────────────────
plt.tight_layout()

OUTPUT_PATH = "fixation_distribution_v2.png"
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {OUTPUT_PATH}")
print("Download it from the Files panel in Colab (left sidebar).")

plt.show()
