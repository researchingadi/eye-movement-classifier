# =============================================================================
# STEP 3 — SANITY CHECK
# Eye-Tracking Memory Task Classifier — Whitlock Lab
#
# PURPOSE:
#   Visually and statistically verify the feature matrix before any classifier
#   touches it. This step exists to catch problems early — a corrupted feature,
#   a skewed distribution, a feature that behaves opposite to theory — before
#   they contaminate the results.
#
#   This is not exploratory analysis. It is a pre-classifier quality gate.
#   Every plot and test here answers one specific question:
#   "Is this feature behaving the way it should?"
#
# INPUT:
#   feature_matrix_encoding.csv — output of Step 2
#
# OUTPUT:
#   sanity_check_plots.png  — multi-panel figure showing all feature distributions
#   Printed statistical report — t-tests and effect sizes per feature
#
# WHAT THIS SCRIPT CHECKS:
#   1. Feature distributions — are there outliers or unexpected shapes?
#   2. NaN audit — exactly which features have missing values and how many?
#   3. Univariate Item vs Relational comparison — does each feature differ
#      between tasks in the theoretically predicted direction?
#   4. Effect sizes (Cohen's d) — how large are the differences?
#   5. Per-subject consistency — do most subjects show the expected direction,
#      or is the group mean driven by a few outliers?
#   6. Feature correlations — are any features so correlated they are
#      essentially duplicates?
#
# HOW TO RUN IN GOOGLE COLAB:
#   1. Upload feature_matrix_encoding.csv
#   2. Run all cells in order
#   3. Download sanity_check_plots.png from the Files panel
#   4. Copy the printed report and send back for documentation
#
# =============================================================================


# =============================================================================
# CELL 1 — IMPORTS AND CONFIGURATION
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

print("=" * 65)
print("STEP 3 — SANITY CHECK")
print("Eye-Tracking Memory Task Classifier — Whitlock Lab")
print("=" * 65)
print()

# ── File paths ────────────────────────────────────────────────────────────────
INPUT_PATH  = "feature_matrix_encoding.csv"
OUTPUT_PATH = "sanity_check_plots.png"

# ── Plot style ────────────────────────────────────────────────────────────────
# Consistent with project color scheme
COLOR_ITEM       = '#2166AC'   # blue  — Item task
COLOR_RELATIONAL = '#D6604D'   # red-orange — Relational task
COLOR_NEUTRAL    = '#888888'   # grey  — neutral elements

# Publication-ready defaults
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        9,
    'axes.titlesize':   9,
    'axes.labelsize':   8,
    'xtick.labelsize':  7,
    'ytick.labelsize':  7,
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

# ── Feature definitions ───────────────────────────────────────────────────────
# Metadata columns — not features, never plotted or tested
METADATA_COLS = ['subject_id', 'trial_id', 'task', 'task_label']

# All 19 features in logical groups for organised plotting
FEATURE_GROUPS = {
    'AOI Dwell': [
        'obj_dwell_prop',
        'scene_dwell_prop',
        'obj_fix_count',
        'scene_fix_count',
    ],
    'Temporal Dwell (Thirds)': [
        'obj_dwell_early_ms',
        'obj_dwell_middle_ms',
        'obj_dwell_late_ms',
        'scene_dwell_early_ms',
        'scene_dwell_middle_ms',
        'scene_dwell_late_ms',
    ],
    'Duration & Latency': [
        'mean_fix_duration_ms',
        'first_fix_latency_obj_ms',
    ],
    'Transitions': [
        'obj_scene_transitions',
        'transition_entropy',
        'obj_revisits',
        'scene_revisits',
    ],
    'Spatial': [
        'scanpath_length_deg',
        'fixation_dispersion',
        'saccade_amplitude_mean_deg',
    ],
}

# Flat list of all features in order (used for correlation matrix etc.)
ALL_FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]

# Theoretical prediction for each feature: which task should be higher?
# Used in the univariate comparison to flag unexpected directions.
EXPECTED_HIGHER = {
    'obj_dwell_prop':               'ITEM',
    'scene_dwell_prop':             'RELATIONAL',
    'obj_fix_count':                'ITEM',
    'scene_fix_count':              'RELATIONAL',
    'obj_dwell_early_ms':           'ITEM',
    'obj_dwell_middle_ms':          'ITEM',
    'obj_dwell_late_ms':            'ITEM',
    'scene_dwell_early_ms':         'RELATIONAL',
    'scene_dwell_middle_ms':        'RELATIONAL',
    'scene_dwell_late_ms':          'RELATIONAL',
    'mean_fix_duration_ms':         'ITEM',
    'first_fix_latency_obj_ms':     'RELATIONAL',
    'obj_scene_transitions':        'RELATIONAL',
    'transition_entropy':           'RELATIONAL',
    'obj_revisits':                 'RELATIONAL',
    'scene_revisits':               'RELATIONAL',
    'scanpath_length_deg':          'RELATIONAL',
    'fixation_dispersion':          'RELATIONAL',
    'saccade_amplitude_mean_deg':   'RELATIONAL',
}

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH)

item_df = df[df['task_label'] == 1]
rel_df  = df[df['task_label'] == 0]

print(f"Loaded: {len(df):,} trials | "
      f"ITEM: {len(item_df):,} | RELATIONAL: {len(rel_df):,} | "
      f"Subjects: {df['subject_id'].nunique()}")
print()


# =============================================================================
# CELL 2 — NaN AUDIT
#
# Exactly which features have missing values, how many, and in which task?
# Expected NaNs are documented — unexpected ones would be a red flag.
# =============================================================================

print("=" * 65)
print("CHECK 1: NaN AUDIT")
print("=" * 65)

nan_found = False
for feat in ALL_FEATURES:
    n_nan        = df[feat].isnull().sum()
    n_nan_item   = item_df[feat].isnull().sum()
    n_nan_rel    = rel_df[feat].isnull().sum()

    if n_nan > 0:
        nan_found = True
        pct = 100 * n_nan / len(df)
        print(f"  {feat:<35} {n_nan:>5} NaNs ({pct:.1f}%) "
              f"| ITEM: {n_nan_item} | REL: {n_nan_rel}")
        # Flag if unexpected — more than 5% NaN in any feature besides
        # transition_entropy (which we know has ~32% NaN by design)
        if feat != 'transition_entropy' and pct > 5.0:
            print(f"    ⚠ WARNING: unexpectedly high NaN rate for {feat}")

if not nan_found:
    print("  No NaN values found.")

print()


# =============================================================================
# CELL 3 — UNIVARIATE STATISTICAL COMPARISONS
#
# For each feature, run an independent-samples t-test comparing Item vs
# Relational and compute Cohen's d effect size.
#
# This is NOT the classifier — it's a sanity check that each feature has
# signal in the expected direction. If a feature shows the wrong direction
# or a very small effect, we flag it for discussion with Whitlock.
# =============================================================================

print("=" * 65)
print("CHECK 2: UNIVARIATE ITEM vs RELATIONAL COMPARISONS")
print("=" * 65)
print(f"  {'Feature':<35} {'ITEM mean':>10} {'REL mean':>10} "
      f"{'Cohen d':>9} {'p-value':>10} {'Direction':>12}")
print("  " + "-" * 92)

stat_results = {}

for feat in ALL_FEATURES:
    # Drop NaN values for this comparison
    item_vals = item_df[feat].dropna().values
    rel_vals  = rel_df[feat].dropna().values

    if len(item_vals) < 2 or len(rel_vals) < 2:
        print(f"  {feat:<35} Skipped — insufficient non-NaN values")
        continue

    # Independent samples t-test (Welch's — does not assume equal variance)
    t_stat, p_val = stats.ttest_ind(item_vals, rel_vals, equal_var=False)

    # Cohen's d — standardised mean difference
    # Using pooled SD (appropriate for Welch's t-test context)
    mean_diff   = item_vals.mean() - rel_vals.mean()
    pooled_std  = np.sqrt(
        (item_vals.std(ddof=1)**2 + rel_vals.std(ddof=1)**2) / 2
    )
    cohens_d    = mean_diff / pooled_std if pooled_std > 0 else np.nan

    # Check direction against theoretical prediction
    expected    = EXPECTED_HIGHER[feat]
    actual      = 'ITEM' if item_vals.mean() > rel_vals.mean() else 'RELATIONAL'
    direction_ok = '✓' if actual == expected else '✗ UNEXPECTED'

    # Format p-value
    p_str = '< 0.001' if p_val < 0.001 else f'{p_val:.3f}'

    print(f"  {feat:<35} {item_vals.mean():>10.3f} {rel_vals.mean():>10.3f} "
          f"{cohens_d:>9.3f} {p_str:>10}  {direction_ok}")

    stat_results[feat] = {
        'item_mean':  item_vals.mean(),
        'rel_mean':   rel_vals.mean(),
        'cohens_d':   cohens_d,
        'p_value':    p_val,
        'direction':  direction_ok,
    }

print()
# Summary of direction checks
n_correct   = sum(1 for r in stat_results.values() if '✓' in r['direction'])
n_total     = len(stat_results)
n_large_d   = sum(1 for r in stat_results.values()
                  if abs(r['cohens_d']) > 0.8 and not np.isnan(r['cohens_d']))
print(f"  Direction correct: {n_correct}/{n_total} features")
print(f"  Large effects (|d| > 0.8): {n_large_d}/{n_total} features")
print()


# =============================================================================
# CELL 4 — PER-SUBJECT CONSISTENCY CHECK
#
# The group means could be correct even if the effect is driven by only a
# few subjects. Here we check: for each feature, what proportion of subjects
# individually show the expected direction?
#
# A feature where 90%+ of subjects show the expected direction is robust.
# A feature where only 60% do is more variable and may be less reliable
# as a classifier feature.
# =============================================================================

print("=" * 65)
print("CHECK 3: PER-SUBJECT CONSISTENCY")
print("=" * 65)
print("  % of subjects individually showing the expected direction per feature")
print()
print(f"  {'Feature':<35} {'% Subjects Correct':>20} {'N subjects':>12}")
print("  " + "-" * 70)

for feat in ALL_FEATURES:
    expected = EXPECTED_HIGHER[feat]

    subject_results = []
    for subj in df['subject_id'].unique():
        subj_df    = df[df['subject_id'] == subj]
        item_mean  = subj_df[subj_df['task_label']==1][feat].mean()
        rel_mean   = subj_df[subj_df['task_label']==0][feat].mean()

        # Skip subject if both means are NaN
        if np.isnan(item_mean) and np.isnan(rel_mean):
            continue

        # Handle case where one task has all NaN (edge case)
        if np.isnan(item_mean) or np.isnan(rel_mean):
            subject_results.append(False)
            continue

        actual = 'ITEM' if item_mean > rel_mean else 'RELATIONAL'
        subject_results.append(actual == expected)

    n_subj     = len(subject_results)
    pct_correct = 100 * sum(subject_results) / n_subj if n_subj > 0 else np.nan

    # Flag features where fewer than 70% of subjects show expected direction
    flag = ' ⚠ low consistency' if pct_correct < 70 else ''
    print(f"  {feat:<35} {pct_correct:>18.1f}% {n_subj:>12}{flag}")

print()


# =============================================================================
# CELL 5 — FEATURE CORRELATION MATRIX
#
# Check for highly correlated features. Pairs with r > 0.90 are essentially
# measuring the same thing — the classifier will still work but it helps
# us understand the feature space and write a better methods section.
#
# We flag any pair above 0.90 for awareness, not exclusion.
# =============================================================================

print("=" * 65)
print("CHECK 4: HIGH FEATURE CORRELATIONS (|r| > 0.90)")
print("=" * 65)

# Use pairwise complete observations (drop NaN per pair)
corr_matrix = df[ALL_FEATURES].corr(method='pearson')

high_corr_found = False
for i in range(len(ALL_FEATURES)):
    for j in range(i + 1, len(ALL_FEATURES)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.90:
            high_corr_found = True
            print(f"  r = {r:+.3f}  |  "
                  f"{ALL_FEATURES[i]}  ←→  {ALL_FEATURES[j]}")

if not high_corr_found:
    print("  No feature pairs with |r| > 0.90 found.")

print()


# =============================================================================
# CELL 6 — VISUALISATION
#
# Multi-panel figure with three sections:
#   Panel A — Violin plots: distribution of each feature by task
#   Panel B — Bar plots: mean ± SE per task with individual subject means
#   Panel C — Correlation heatmap: full feature correlation matrix
#
# All saved to a single PNG for sending to Whitlock.
# =============================================================================

print("=" * 65)
print("GENERATING PLOTS")
print("=" * 65)

# ── Figure A: Violin plots — one per feature ──────────────────────────────────
n_features = len(ALL_FEATURES)
n_cols     = 4
n_rows     = int(np.ceil(n_features / n_cols))

fig_a, axes_a = plt.subplots(
    n_rows, n_cols,
    figsize=(16, n_rows * 3.2),
    constrained_layout=True
)
fig_a.suptitle(
    'Feature Distributions by Task — Item (blue) vs Relational (red)',
    fontsize=12, fontweight='bold', y=1.01
)

axes_flat = axes_a.flatten()

for idx, feat in enumerate(ALL_FEATURES):
    ax = axes_flat[idx]

    item_vals = item_df[feat].dropna().values
    rel_vals  = rel_df[feat].dropna().values

    # Violin plot
    parts = ax.violinplot(
        [item_vals, rel_vals],
        positions=[0, 1],
        showmedians=True,
        showextrema=False,
    )

    # Color the violins
    for pc, color in zip(parts['bodies'], [COLOR_ITEM, COLOR_RELATIONAL]):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)

    # Labels
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Item', 'Rel'], fontsize=7)
    ax.set_title(feat.replace('_', '\n'), fontsize=7, pad=3)

    # Add Cohen's d annotation if computed
    if feat in stat_results:
        d     = stat_results[feat]['cohens_d']
        p     = stat_results[feat]['p_value']
        p_str = 'p<.001' if p < 0.001 else f'p={p:.2f}'
        ax.text(
            0.97, 0.97,
            f"d={d:.2f}\n{p_str}",
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=6, color=COLOR_NEUTRAL
        )

# Hide any unused subplot panels
for idx in range(n_features, len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig_a.savefig('sanity_check_distributions.png', dpi=200, bbox_inches='tight')
print("  Saved: sanity_check_distributions.png")


# ── Figure B: Correlation heatmap ─────────────────────────────────────────────
fig_b, ax_b = plt.subplots(figsize=(12, 10), constrained_layout=True)
fig_b.suptitle(
    'Feature Correlation Matrix (Pearson r)',
    fontsize=12, fontweight='bold'
)

# Shorter labels for the heatmap
short_labels = [f.replace('_ms', '').replace('_deg', '').replace('_', '\n')
                for f in ALL_FEATURES]

mask   = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
cmap   = sns.diverging_palette(220, 20, as_cmap=True)

sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap=cmap,
    vmin=-1, vmax=1, center=0,
    square=True,
    linewidths=0.4,
    cbar_kws={'shrink': 0.7, 'label': 'Pearson r'},
    xticklabels=short_labels,
    yticklabels=short_labels,
    annot=True,
    fmt='.2f',
    annot_kws={'size': 6},
    ax=ax_b
)
ax_b.tick_params(axis='x', rotation=45, labelsize=7)
ax_b.tick_params(axis='y', rotation=0,  labelsize=7)

fig_b.savefig('sanity_check_correlations.png', dpi=200, bbox_inches='tight')
print("  Saved: sanity_check_correlations.png")


# ── Figure C: Per-subject consistency bar chart ───────────────────────────────
fig_c, ax_c = plt.subplots(figsize=(14, 5), constrained_layout=True)
fig_c.suptitle(
    'Per-Subject Consistency — % of Subjects Showing Expected Direction',
    fontsize=11, fontweight='bold'
)

consistency_pcts = []
for feat in ALL_FEATURES:
    expected = EXPECTED_HIGHER[feat]
    results  = []
    for subj in df['subject_id'].unique():
        subj_df   = df[df['subject_id'] == subj]
        item_mean = subj_df[subj_df['task_label']==1][feat].mean()
        rel_mean  = subj_df[subj_df['task_label']==0][feat].mean()
        if np.isnan(item_mean) or np.isnan(rel_mean):
            continue
        actual = 'ITEM' if item_mean > rel_mean else 'RELATIONAL'
        results.append(actual == expected)
    pct = 100 * sum(results) / len(results) if results else 0
    consistency_pcts.append(pct)

# Color bars by consistency level
bar_colors = [
    COLOR_ITEM if p >= 80
    else COLOR_RELATIONAL if p < 70
    else COLOR_NEUTRAL
    for p in consistency_pcts
]

bars = ax_c.bar(
    range(len(ALL_FEATURES)),
    consistency_pcts,
    color=bar_colors,
    alpha=0.85,
    edgecolor='white',
    linewidth=0.5
)

# Reference lines
ax_c.axhline(80, color=COLOR_NEUTRAL, linestyle='--',
             linewidth=1.0, alpha=0.7, label='80% reference')
ax_c.axhline(70, color=COLOR_RELATIONAL, linestyle=':',
             linewidth=1.0, alpha=0.7, label='70% warning threshold')

ax_c.set_xticks(range(len(ALL_FEATURES)))
ax_c.set_xticklabels(
    [f.replace('_', '\n') for f in ALL_FEATURES],
    rotation=45, ha='right', fontsize=6.5
)
ax_c.set_ylabel('% Subjects Showing Expected Direction', fontsize=9)
ax_c.set_ylim(0, 105)
ax_c.legend(fontsize=8, frameon=False)

# Value labels on bars
for bar, pct in zip(bars, consistency_pcts):
    ax_c.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.5,
        f'{pct:.0f}%',
        ha='center', va='bottom',
        fontsize=6, color='#333333'
    )

fig_c.savefig('sanity_check_consistency.png', dpi=200, bbox_inches='tight')
print("  Saved: sanity_check_consistency.png")

plt.close('all')
print()


# =============================================================================
# CELL 7 — FINAL SUMMARY REPORT
#
# Clean printout of the key findings. This is what goes into the
# documentation and what you share with Whitlock.
# =============================================================================

print("=" * 65)
print("SANITY CHECK SUMMARY")
print("=" * 65)

n_sig = sum(1 for r in stat_results.values() if r['p_value'] < 0.001)
n_correct_dir = sum(1 for r in stat_results.values()
                    if '✓' in r['direction'])
n_large_effect = sum(1 for r in stat_results.values()
                     if abs(r.get('cohens_d', 0)) > 0.8
                     and not np.isnan(r.get('cohens_d', np.nan)))
n_medium_effect = sum(1 for r in stat_results.values()
                      if 0.5 <= abs(r.get('cohens_d', 0)) <= 0.8
                      and not np.isnan(r.get('cohens_d', np.nan)))

print(f"  Total features tested:              {len(stat_results)}")
print(f"  Correct direction (theory match):   {n_correct_dir}/{len(stat_results)}")
print(f"  Significant at p < 0.001:           {n_sig}/{len(stat_results)}")
print(f"  Large effect size (|d| > 0.80):     {n_large_effect}")
print(f"  Medium effect size (0.5 < |d| < 0.8): {n_medium_effect}")
print()

# Identify the top 5 features by effect size
top_features = sorted(
    stat_results.items(),
    key=lambda x: abs(x[1].get('cohens_d', 0)),
    reverse=True
)[:5]

print("  Top 5 features by Cohen's d:")
for feat, res in top_features:
    print(f"    {feat:<35} |d| = {abs(res['cohens_d']):.3f}")

print()
print("  Output files:")
print("    sanity_check_distributions.png  — violin plots per feature")
print("    sanity_check_correlations.png   — correlation heatmap")
print("    sanity_check_consistency.png    — per-subject consistency bars")
print()
print("  Download all three PNG files from the Colab Files panel.")
print("  Send this full output back for documentation.")
print()
print("=" * 65)
print("SANITY CHECK COMPLETE")
print("  If all directions are ✓ and no unexpected NaNs: proceed to Step 4")
print("  If any direction is ✗ or unexpected NaNs: flag before proceeding")
print("=" * 65)
