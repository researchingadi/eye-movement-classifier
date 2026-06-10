# =============================================================================
# STEP 4b — SUBJECT DIAGNOSTIC
# Eye-Tracking Memory Task Classifier — Whitlock Lab
#
# PURPOSE:
#   Before committing to final publication figures, verify that classifier
#   performance is stable across individual subjects. Identifies low-AUC
#   subjects, checks whether poor performance is linked to low trial counts,
#   and produces a diagnostic figure suitable for sharing with Whitlock.
#
#   This is a quality gate, not a publication figure.
#   The question it answers: "Is the model working for most subjects,
#   or is the group AUC carried by a subset of high performers?"
#
# INPUT:
#   loso_predictions.csv       — trial-level predictions from Step 4
#   loso_results_summary.json  — per-subject AUCs from Step 4
#
# OUTPUT:
#   subject_diagnostic.png        — four-panel diagnostic figure
#   subject_diagnostic_summary.csv — per-subject AUC table (for documentation)
#   Printed subject report        — flags low-AUC subjects with context
#
# WHAT THIS SCRIPT CHECKS:
#   1. Per-subject AUC distribution — histogram and sorted bar chart
#   2. Low-AUC subjects — who are they, how many trials do they have?
#   3. AUC vs trial count — is low performance explained by data scarcity?
#   4. AUC vs task balance — is low performance linked to class imbalance
#      within a subject?
#
# HOW TO RUN IN GOOGLE COLAB:
#   1. Upload loso_predictions.csv and loso_results_summary.json
#   2. Paste and run all cells
#   3. Download subject_diagnostic.png and subject_diagnostic_summary.csv
#   4. Send the printed report back for documentation
#
# =============================================================================


# =============================================================================
# CELL 1 — IMPORTS AND CONFIGURATION
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
from sklearn.metrics import roc_auc_score

print("=" * 65)
print("STEP 4b — SUBJECT DIAGNOSTIC")
print("Eye-Tracking Memory Task Classifier — Whitlock Lab")
print("=" * 65)
print()

# ── File paths ────────────────────────────────────────────────────────────────
PREDICTIONS_PATH = "loso_predictions.csv"
RESULTS_PATH     = "loso_results_summary.json"
OUTPUT_PATH      = "subject_diagnostic.png"

# ── Thresholds for flagging ───────────────────────────────────────────────────
# Subjects below LOW_AUC_FLAG are highlighted as low performers
LOW_AUC_FLAG  = 0.70

# ── Plot colours (consistent with project style) ─────────────────────────────
COLOR_ITEM       = '#2166AC'   # blue
COLOR_RELATIONAL = '#D6604D'   # red-orange
COLOR_LOW        = '#D6604D'   # red — low AUC subjects
COLOR_HIGH       = '#2166AC'   # blue — typical/high AUC subjects
COLOR_NEUTRAL    = '#888888'   # grey

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         9,
    'axes.titlesize':    10,
    'axes.labelsize':    9,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})


# =============================================================================
# CELL 2 — LOAD DATA AND COMPUTE PER-SUBJECT METRICS
# =============================================================================

# ── Load predictions ──────────────────────────────────────────────────────────
pred_df = pd.read_csv(PREDICTIONS_PATH)

# ── Load per-subject AUCs from JSON ──────────────────────────────────────────
with open(RESULTS_PATH, 'r') as f:
    results = json.load(f)

subj_aucs_rf = {
    int(k): float(v)
    for k, v in results['random_forest']['per_subject_aucs'].items()
}
subj_aucs_lr = {
    int(k): float(v)
    for k, v in results['logistic_regression']['per_subject_aucs'].items()
}

# ── Build per-subject summary dataframe ──────────────────────────────────────
# For each subject: AUC (RF and LR), trial counts, class balance
subject_rows = []

for subj_id, subj_df in pred_df.groupby('subject_id'):

    n_item       = (subj_df['true_label'] == 1).sum()
    n_relational = (subj_df['true_label'] == 0).sum()
    n_total      = len(subj_df)

    # Class balance: proportion of Item trials (0.5 = perfectly balanced)
    item_prop = n_item / n_total if n_total > 0 else np.nan

    subject_rows.append({
        'subject_id':    subj_id,
        'auc_rf':        subj_aucs_rf.get(subj_id, np.nan),
        'auc_lr':        subj_aucs_lr.get(subj_id, np.nan),
        'n_item':        n_item,
        'n_relational':  n_relational,
        'n_total':       n_total,
        'item_prop':     item_prop,
        'low_auc':       subj_aucs_rf.get(subj_id, 1.0) < LOW_AUC_FLAG,
    })

subj_df_summary = pd.DataFrame(subject_rows).sort_values(
    'auc_rf', ascending=False
).reset_index(drop=True)

n_subjects  = len(subj_df_summary)
n_low       = subj_df_summary['low_auc'].sum()
n_above_0_8 = (subj_df_summary['auc_rf'] >= 0.80).sum()
n_above_0_9 = (subj_df_summary['auc_rf'] >= 0.90).sum()

print(f"Loaded predictions: {len(pred_df):,} trials, {n_subjects} subjects")
print()


# =============================================================================
# CELL 3 — PRINTED SUBJECT REPORT
#
# Full table of per-subject AUCs, flagging low performers.
# This is the report that goes to Whitlock and into the documentation.
# =============================================================================

print("=" * 65)
print("PER-SUBJECT AUC REPORT")
print("=" * 65)
print()
print(f"  {'Subject':>8} {'AUC (RF)':>10} {'AUC (LR)':>10} "
      f"{'N Item':>8} {'N Rel':>8} {'N Total':>8} {'Flag':>8}")
print("  " + "-" * 68)

for _, row in subj_df_summary.iterrows():
    flag = ' ← LOW' if row['low_auc'] else ''
    print(f"  {int(row['subject_id']):>8} "
          f"{row['auc_rf']:>10.3f} "
          f"{row['auc_lr']:>10.3f} "
          f"{int(row['n_item']):>8} "
          f"{int(row['n_relational']):>8} "
          f"{int(row['n_total']):>8}"
          f"{flag}")

print()
print("=" * 65)
print("SUMMARY STATISTICS")
print("=" * 65)
print()
print(f"  N subjects:                  {n_subjects}")
print(f"  Mean AUC (RF):               {subj_df_summary['auc_rf'].mean():.3f}")
print(f"  SD AUC (RF):                 {subj_df_summary['auc_rf'].std():.3f}")
print(f"  Median AUC (RF):             {subj_df_summary['auc_rf'].median():.3f}")
print(f"  Min AUC (RF):                {subj_df_summary['auc_rf'].min():.3f}")
print(f"  Max AUC (RF):                {subj_df_summary['auc_rf'].max():.3f}")
print()
print(f"  Subjects with AUC >= 0.90:   {n_above_0_9} ({100*n_above_0_9/n_subjects:.1f}%)")
print(f"  Subjects with AUC >= 0.80:   {n_above_0_8} ({100*n_above_0_8/n_subjects:.1f}%)")
print(f"  Subjects with AUC <  0.70:   {n_low} ({100*n_low/n_subjects:.1f}%)")
print()

# Flag low-AUC subjects with context
if n_low > 0:
    print(f"  LOW-AUC SUBJECTS (AUC < {LOW_AUC_FLAG}):")
    low_subjs = subj_df_summary[subj_df_summary['low_auc']]
    for _, row in low_subjs.iterrows():
        print(f"    Subject {int(row['subject_id']):>3}: "
              f"AUC = {row['auc_rf']:.3f} | "
              f"N trials = {int(row['n_total'])} "
              f"(Item={int(row['n_item'])}, Rel={int(row['n_relational'])})")
    print()

    # Check whether low-AUC subjects have fewer trials than average
    low_mean_trials  = low_subjs['n_total'].mean()
    high_mean_trials = subj_df_summary[~subj_df_summary['low_auc']]['n_total'].mean()
    print(f"  Mean trials — low AUC subjects:    {low_mean_trials:.1f}")
    print(f"  Mean trials — other subjects:      {high_mean_trials:.1f}")
    if low_mean_trials < high_mean_trials * 0.85:
        print(f"  → Low trial count may partly explain low AUC.")
    else:
        print(f"  → Trial count does not explain low AUC — "
              f"likely genuine individual differences.")
print()


# =============================================================================
# CELL 4 — DIAGNOSTIC FIGURE
#
# Four panels:
#   A — Sorted bar chart: per-subject AUC (RF and LR), low subjects flagged
#   B — Histogram: distribution of per-subject AUCs
#   C — Scatter: AUC vs total trial count (checks data-scarcity explanation)
#   D — Scatter: AUC vs item proportion (checks class balance explanation)
# =============================================================================

print("=" * 65)
print("GENERATING DIAGNOSTIC FIGURE")
print("=" * 65)

fig = plt.figure(figsize=(14, 10), constrained_layout=True)
fig.suptitle(
    'Per-Subject AUC Diagnostic\n'
    'Eye-Tracking Memory Task Classifier — Whitlock Lab',
    fontsize=12, fontweight='bold'
)

gs = gridspec.GridSpec(2, 2, figure=fig)
ax_a = fig.add_subplot(gs[0, 0])   # top left  — sorted AUC bar chart
ax_b = fig.add_subplot(gs[0, 1])   # top right — AUC histogram
ax_c = fig.add_subplot(gs[1, 0])   # bottom left  — AUC vs trial count
ax_d = fig.add_subplot(gs[1, 1])   # bottom right — AUC vs item proportion


# ── Panel A: Sorted per-subject AUC bar chart ─────────────────────────────────
# Subjects sorted by RF AUC descending
# Low-AUC subjects highlighted in red, others in blue
# LR AUC shown as a lighter overlay for comparison

sorted_df = subj_df_summary.reset_index(drop=True)
x_pos     = np.arange(len(sorted_df))

bar_colors = [
    COLOR_LOW if row['low_auc'] else COLOR_HIGH
    for _, row in sorted_df.iterrows()
]

ax_a.bar(x_pos, sorted_df['auc_rf'], color=bar_colors,
         alpha=0.85, width=0.7, label='Random Forest', zorder=2)
ax_a.scatter(x_pos, sorted_df['auc_lr'], color=COLOR_NEUTRAL,
             s=12, alpha=0.7, zorder=3, label='Logistic Regression')

# Reference lines
ax_a.axhline(0.5,  color='black',        linestyle='--',
             linewidth=0.8, alpha=0.5, label='Chance (0.5)')
ax_a.axhline(LOW_AUC_FLAG, color=COLOR_LOW, linestyle=':',
             linewidth=1.0, alpha=0.7, label=f'Low AUC threshold ({LOW_AUC_FLAG})')
ax_a.axhline(subj_df_summary['auc_rf'].mean(), color=COLOR_HIGH,
             linestyle='--', linewidth=1.0, alpha=0.6,
             label=f"Mean AUC = {subj_df_summary['auc_rf'].mean():.3f}")

# Annotate low-AUC subjects with their subject ID
for idx, row in sorted_df.iterrows():
    if row['low_auc']:
        ax_a.text(idx, row['auc_rf'] + 0.012,
                  f"S{int(row['subject_id'])}",
                  ha='center', va='bottom',
                  fontsize=6.5, color=COLOR_LOW, fontweight='bold')

ax_a.set_xlabel('Subjects (sorted by AUC, descending)', fontsize=9)
ax_a.set_ylabel('AUC', fontsize=9)
ax_a.set_title('Per-Subject AUC\nRandom Forest (bars) vs Logistic Regression (dots)',
               fontsize=9)
ax_a.set_xticks([])
ax_a.set_ylim(0.40, 1.05)
ax_a.legend(fontsize=7, frameon=False, loc='lower left', ncol=1)


# ── Panel B: Histogram of per-subject AUCs ────────────────────────────────────
bins = np.arange(0.45, 1.05, 0.05)

ax_b.hist(sorted_df['auc_rf'], bins=bins, color=COLOR_HIGH,
          alpha=0.80, edgecolor='white', linewidth=0.5,
          label='Random Forest')
ax_b.hist(sorted_df['auc_lr'], bins=bins, color=COLOR_NEUTRAL,
          alpha=0.50, edgecolor='white', linewidth=0.5,
          label='Logistic Regression')

ax_b.axvline(sorted_df['auc_rf'].mean(), color=COLOR_HIGH,
             linestyle='--', linewidth=1.2,
             label=f"RF mean = {sorted_df['auc_rf'].mean():.3f}")
ax_b.axvline(0.5, color='black', linestyle='--',
             linewidth=0.8, alpha=0.5, label='Chance')
ax_b.axvline(LOW_AUC_FLAG, color=COLOR_LOW, linestyle=':',
             linewidth=1.0, alpha=0.8, label=f'Low threshold')

ax_b.set_xlabel('AUC', fontsize=9)
ax_b.set_ylabel('Number of Subjects', fontsize=9)
ax_b.set_title('Distribution of Per-Subject AUC', fontsize=9)
ax_b.legend(fontsize=7.5, frameon=False)

# Add count annotations
n_high_text = (sorted_df['auc_rf'] >= 0.80).sum()
ax_b.text(0.97, 0.95,
          f"N ≥ 0.80: {n_high_text} ({100*n_high_text/n_subjects:.0f}%)\n"
          f"N < 0.70: {n_low} ({100*n_low/n_subjects:.0f}%)",
          transform=ax_b.transAxes,
          ha='right', va='top',
          fontsize=8, color='#333333',
          bbox=dict(boxstyle='round,pad=0.3',
                    facecolor='#F5F5F5', edgecolor='#CCCCCC', alpha=0.8))


# ── Panel C: AUC vs total trial count ────────────────────────────────────────
# Checks whether low AUC is explained by having fewer trials

point_colors_c = [COLOR_LOW if row['low_auc'] else COLOR_HIGH
                  for _, row in sorted_df.iterrows()]

ax_c.scatter(sorted_df['n_total'], sorted_df['auc_rf'],
             c=point_colors_c, alpha=0.75, s=40, edgecolors='white',
             linewidth=0.5, zorder=2)

# Annotate low-AUC subjects
for _, row in sorted_df[sorted_df['low_auc']].iterrows():
    ax_c.annotate(
        f"S{int(row['subject_id'])}",
        xy=(row['n_total'], row['auc_rf']),
        xytext=(6, 3), textcoords='offset points',
        fontsize=7, color=COLOR_LOW, fontweight='bold'
    )

# Trend line
z = np.polyfit(sorted_df['n_total'], sorted_df['auc_rf'], 1)
p = np.poly1d(z)
x_line = np.linspace(sorted_df['n_total'].min(),
                     sorted_df['n_total'].max(), 100)
ax_c.plot(x_line, p(x_line), color=COLOR_NEUTRAL,
          linewidth=1.0, linestyle='--', alpha=0.6, label='Trend')

r_trials = np.corrcoef(sorted_df['n_total'], sorted_df['auc_rf'])[0, 1]
ax_c.text(0.97, 0.05, f"r = {r_trials:.3f}",
          transform=ax_c.transAxes,
          ha='right', va='bottom', fontsize=8, color='#555555')

ax_c.axhline(LOW_AUC_FLAG, color=COLOR_LOW, linestyle=':',
             linewidth=1.0, alpha=0.7)
ax_c.axhline(0.5, color='black', linestyle='--',
             linewidth=0.8, alpha=0.5)

ax_c.set_xlabel('Total Trials per Subject', fontsize=9)
ax_c.set_ylabel('AUC (Random Forest)', fontsize=9)
ax_c.set_title('AUC vs Trial Count\n(checks data-scarcity explanation)', fontsize=9)
ax_c.legend(fontsize=7.5, frameon=False)


# ── Panel D: AUC vs item proportion ──────────────────────────────────────────
# Checks whether low AUC is linked to class imbalance within a subject.
# A subject with very few Item trials relative to Relational (or vice versa)
# may be harder to classify because the per-subject AUC estimate is noisy
# or because the classifier saw an unbalanced training signal for that fold.
#
# item_prop = n_item / n_total
# 0.5 = perfectly balanced, < 0.5 = more Relational, > 0.5 = more Item

point_colors_d = [COLOR_LOW if row['low_auc'] else COLOR_HIGH
                  for _, row in sorted_df.iterrows()]

ax_d.scatter(sorted_df['item_prop'], sorted_df['auc_rf'],
             c=point_colors_d, alpha=0.75, s=40, edgecolors='white',
             linewidth=0.5, zorder=2)

# Annotate low-AUC subjects
for _, row in sorted_df[sorted_df['low_auc']].iterrows():
    ax_d.annotate(
        f"S{int(row['subject_id'])}",
        xy=(row['item_prop'], row['auc_rf']),
        xytext=(6, 3), textcoords='offset points',
        fontsize=7, color=COLOR_LOW, fontweight='bold'
    )

# Trend line
z_d = np.polyfit(sorted_df['item_prop'].dropna(),
                 sorted_df.loc[sorted_df['item_prop'].notna(), 'auc_rf'], 1)
p_d = np.poly1d(z_d)
x_line_d = np.linspace(sorted_df['item_prop'].min(),
                       sorted_df['item_prop'].max(), 100)
ax_d.plot(x_line_d, p_d(x_line_d), color=COLOR_NEUTRAL,
          linewidth=1.0, linestyle='--', alpha=0.6, label='Trend')

# Reference lines
r_balance = np.corrcoef(
    sorted_df['item_prop'].dropna(),
    sorted_df.loc[sorted_df['item_prop'].notna(), 'auc_rf']
)[0, 1]
ax_d.text(0.97, 0.05, f"r = {r_balance:.3f}",
          transform=ax_d.transAxes,
          ha='right', va='bottom', fontsize=8, color='#555555')

ax_d.axvline(0.5, color=COLOR_NEUTRAL, linestyle='--',
             linewidth=0.8, alpha=0.6, label='Perfect balance (0.5)')
ax_d.axhline(LOW_AUC_FLAG, color=COLOR_LOW, linestyle=':',
             linewidth=1.0, alpha=0.7)
ax_d.axhline(0.5, color='black', linestyle='--',
             linewidth=0.8, alpha=0.5)

ax_d.set_xlabel('Item Trial Proportion (0.5 = balanced)', fontsize=9)
ax_d.set_ylabel('AUC (Random Forest)', fontsize=9)
ax_d.set_title('AUC vs Class Balance\n(checks task imbalance explanation)', fontsize=9)
ax_d.legend(fontsize=7.5, frameon=False)


# ── Save ──────────────────────────────────────────────────────────────────────
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight')
print(f"  Saved: {OUTPUT_PATH}")
print()


# =============================================================================
# CELL 5 — SAVE SUBJECT SUMMARY CSV AND PRINT FINAL VERDICT
#
# Saves the per-subject diagnostic table as a CSV for documentation
# and future paper tables.
# =============================================================================

# ── Save subject summary CSV ──────────────────────────────────────────────────
subj_df_summary.to_csv("subject_diagnostic_summary.csv", index=False)
print(f"  Saved: subject_diagnostic_summary.csv")
print(f"    {len(subj_df_summary)} rows × {len(subj_df_summary.columns)} columns")
print(f"    Columns: {list(subj_df_summary.columns)}")
print()

print("=" * 65)
print("DIAGNOSTIC VERDICT")
print("=" * 65)
print()

# Use correlation already computed in Panel C
# r_trials is defined during Panel C plotting above

# Compute proportion of subjects above chance (AUC > 0.5)
n_above_chance = (subj_df_summary['auc_rf'] > 0.5).sum()

print(f"  Subjects above chance (AUC > 0.5):  "
      f"{n_above_chance}/{n_subjects} ({100*n_above_chance/n_subjects:.1f}%)")
print(f"  Subjects with AUC >= 0.80:           "
      f"{n_above_0_8}/{n_subjects} ({100*n_above_0_8/n_subjects:.1f}%)")
print(f"  Subjects with AUC >= 0.90:           "
      f"{n_above_0_9}/{n_subjects} ({100*n_above_0_9/n_subjects:.1f}%)")
print(f"  Subjects with AUC < {LOW_AUC_FLAG}:           "
      f"{n_low}/{n_subjects} ({100*n_low/n_subjects:.1f}%)")
print()
print(f"  Correlation between AUC and trial count: r = {r_trials:.3f}")
if abs(r_trials) < 0.20:
    print(f"  → Weak correlation — low AUC is not explained by fewer trials.")
elif abs(r_trials) < 0.40:
    print(f"  → Moderate correlation — trial count may partly explain some variability.")
else:
    print(f"  → Strong correlation — trial count is a meaningful predictor of AUC.")
print()

# Overall verdict
if n_low <= 5 and n_above_0_8 >= n_subjects * 0.70:
    print("  VERDICT: Model is stable across subjects.")
    print("  The low-AUC subjects represent a small minority and likely")
    print("  reflect genuine individual differences in encoding consistency,")
    print("  not a modelling failure.")
    print()
    print("  → Safe to proceed to Step 5 (publication figures).")
else:
    print("  VERDICT: Review low-AUC subjects before proceeding.")
    print("  More than expected subjects are below threshold.")
    print("  Recommend discussing with Whitlock before Step 5.")
print()
print("  Download both output files from the Colab Files panel:")
print("    subject_diagnostic.png")
print("    subject_diagnostic_summary.csv")
print("  Send the full output above back for documentation.")
print()
print("=" * 65)
print("SUBJECT DIAGNOSTIC COMPLETE")
print("=" * 65)
