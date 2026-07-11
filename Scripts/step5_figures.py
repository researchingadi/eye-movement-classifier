# =============================================================================
# STEP 5 — PUBLICATION FIGURES
# Eye-Tracking Memory Task Classifier — Whitlock Lab
#
# PURPOSE:
#   Generate all four publication-quality figures from the completed
#   classifier outputs. Each figure is saved as a separate PDF at 300 DPI.
#
# FIGURES:
#   figure1_roc.pdf          — ROC curve + permutation null distribution
#   figure2_confusion.pdf    — Confusion matrix (RF primary)
#   figure3_shap.pdf         — SHAP feature importance (bar + beeswarm)
#   figure_s1_subject_auc.pdf — Supplementary: per-subject AUC distribution
#
# COLOR SPEC (locked — do not change):
#   #2166AC  — Item task / RF / primary blue
#   #D6604D  — Relational task / low AUC / red-orange
#   #888888  — LR baseline / neutral gray / secondary elements
#   #222222  — near-black for all text and axis lines
#   #F5F5F5  — light gray for annotation boxes only
#
# INPUTS (all from previous steps):
#   loso_predictions.csv
#   loso_results_summary.json
#   permutation_results.json
#   permutation_null_dist_rf.csv
#   bootstrap_auc_random_forest.csv
#   bootstrap_auc_logistic_regression.csv
#   shap_values.csv
#   shap_feature_importance.csv
#   feature_matrix_encoding.csv
#   subject_diagnostic_summary.csv
#
# HOW TO RUN IN GOOGLE COLAB:
#   1. Upload all input files listed above
#   2. Run Cell 1 (imports)
#   3. Run Cell 2 (load all data)
#   4. Run Cell 3 (Figure 1 — ROC)
#   5. Run Cell 4 (Figure 2 — Confusion matrix)
#   6. Run Cell 5 (Figure 3 — SHAP)
#   7. Run Cell 6 (Supplementary — per-subject AUC)
#   8. Download all four PDFs from the Files panel
#
# =============================================================================


# =============================================================================
# CELL 1 — IMPORTS
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_curve, roc_auc_score

# ── Color palette — locked ────────────────────────────────────────────────────
C_ITEM     = '#2166AC'   # blue  — Item task / RF primary
C_REL      = '#D6604D'   # red-orange — Relational / low AUC
C_NEUTRAL  = '#888888'   # gray  — LR / secondary elements
C_BLACK    = '#222222'   # near-black — text / axes
C_BG       = '#F5F5F5'   # light gray — annotation boxes only

# ── Global matplotlib style ───────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          9,
    'axes.titlesize':     10,
    'axes.labelsize':     9,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.linewidth':     0.8,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
    'axes.edgecolor':     C_BLACK,
    'xtick.color':        C_BLACK,
    'ytick.color':        C_BLACK,
    'text.color':         C_BLACK,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.facecolor':  'white',
})

print("Imports complete.")


# =============================================================================
# CELL 2 — LOAD ALL DATA
# =============================================================================

print("Loading data...")

# Predictions
pred_df = pd.read_csv('loso_predictions.csv')

# Results JSON
with open('loso_results_summary.json') as f:
    results = json.load(f)
rf  = results['random_forest']
lr  = results['logistic_regression']

# Permutation results
with open('permutation_results.json') as f:
    perm = json.load(f)
perm_rf = perm['random_forest']

# Null distribution
null_aucs = pd.read_csv('permutation_null_dist_rf.csv')['null_auc_rf'].values

# Bootstrap distributions
bs_rf = pd.read_csv('bootstrap_auc_random_forest.csv')['bootstrap_auc_rf'].values
bs_lr = pd.read_csv('bootstrap_auc_logistic_regression.csv')['bootstrap_auc_lr'].values

# SHAP
shap_imp = pd.read_csv('shap_feature_importance.csv')
shap_val = pd.read_csv('shap_values.csv')

# Feature matrix (for SHAP beeswarm feature values)
feat_df = pd.read_csv('feature_matrix_encoding.csv')

# Subject diagnostic
subj_df = pd.read_csv('subject_diagnostic_summary.csv')

# ── Key numbers ───────────────────────────────────────────────────────────────
OBS_AUC   = rf['pooled_auc']          # 0.8615
CI_LO     = rf['ci_lower']             # 0.8391
CI_HI     = rf['ci_upper']             # 0.8813
LR_AUC    = lr['pooled_auc']           # 0.8476
P_VALUE   = perm_rf['p_value']         # 0.005
NULL_MEAN = perm_rf['null_mean_auc']   # 0.499
CM        = np.array(rf['confusion_matrix'])  # [[TN, FP], [FN, TP]]
ACCURACY  = rf['accuracy']
SENS      = rf['sensitivity']
SPEC      = rf['specificity']
N_SUBJ    = results['dataset']['n_subjects']

print(f"Loaded: {len(pred_df):,} trials | {N_SUBJ} subjects")
print(f"RF AUC = {OBS_AUC:.3f} | LR AUC = {LR_AUC:.3f} | p = {P_VALUE:.3f}")
print()


# =============================================================================
# CELL 3 — FIGURE 1: ROC CURVE + PERMUTATION NULL DISTRIBUTION
#
# Two panels:
#   Left  — ROC curve: RF (blue) and LR (gray) on same axes
#   Right — Permutation null distribution histogram with observed AUC marked
# =============================================================================

print("Building Figure 1 — ROC Curve...")

fig1, (ax_roc, ax_perm) = plt.subplots(1, 2, figsize=(8.5, 4.0))
fig1.subplots_adjust(wspace=0.35)

# ── Left panel: ROC curve ─────────────────────────────────────────────────────
fpr_rf, tpr_rf, _ = roc_curve(pred_df['true_label'], pred_df['prob_rf'])
fpr_lr, tpr_lr, _ = roc_curve(pred_df['true_label'], pred_df['prob_lr'])

# RF — primary
ax_roc.plot(fpr_rf, tpr_rf, color=C_ITEM, linewidth=2.0, zorder=3,
            label=f'Random Forest — AUC = {OBS_AUC:.3f} [{CI_LO:.3f}–{CI_HI:.3f}]')

# LR — secondary
ax_roc.plot(fpr_lr, tpr_lr, color=C_NEUTRAL, linewidth=1.2,
            linestyle='--', zorder=2,
            label=f'Logistic Regression — AUC = {LR_AUC:.3f}')

# Chance diagonal
ax_roc.plot([0, 1], [0, 1], color=C_BLACK, linewidth=0.8,
            linestyle=':', alpha=0.5, zorder=1, label='Chance')

# Shaded area under RF curve
ax_roc.fill_between(fpr_rf, tpr_rf, alpha=0.06, color=C_ITEM, zorder=1)

# p-value annotation — bottom right, away from legend
p_str = 'p = .005' if P_VALUE >= 0.001 else 'p < .001'
ax_roc.text(0.97, 0.08,
            f'Permutation {p_str}\n(200 within-subject shuffles)',
            transform=ax_roc.transAxes,
            ha='right', va='bottom', fontsize=7.5, color=C_NEUTRAL,
            bbox=dict(boxstyle='round,pad=0.35', facecolor=C_BG,
                      edgecolor='#CCCCCC', alpha=0.9))

ax_roc.set_xlabel('False Positive Rate', color=C_BLACK)
ax_roc.set_ylabel('True Positive Rate', color=C_BLACK)
ax_roc.set_title('ROC Curve — LOSO Cross-Validation\n'
                 f'(83 subjects, {results["dataset"]["n_trials"]:,} trials)',
                 fontsize=9.5)
# Legend moved to upper left — most empty region of the ROC plot
ax_roc.legend(loc='upper left', frameon=False, fontsize=7.5)
ax_roc.set_xlim(-0.01, 1.01)
ax_roc.set_ylim(-0.01, 1.01)
ax_roc.set_aspect('equal')

# ── Right panel: permutation null distribution ────────────────────────────────
bins = np.linspace(0.44, 0.90, 32)

ax_perm.hist(null_aucs, bins=bins, color=C_NEUTRAL, alpha=0.75,
             edgecolor='white', linewidth=0.4,
             label=f'Null AUC distribution\n(mean = {NULL_MEAN:.3f})')

# Observed AUC line
ax_perm.axvline(OBS_AUC, color=C_ITEM, linewidth=2.0, zorder=3,
                label=f'Observed AUC = {OBS_AUC:.3f}')

# Null mean reference
ax_perm.axvline(NULL_MEAN, color=C_BLACK, linewidth=0.8,
                linestyle=':', alpha=0.6, label=f'Null mean = {NULL_MEAN:.3f}')

# p-value annotation
ax_perm.text(0.97, 0.95,
             f'{p_str}\n0 / 200 permutations\nexceeded observed AUC',
             transform=ax_perm.transAxes,
             ha='right', va='top', fontsize=7.5, color=C_BLACK,
             bbox=dict(boxstyle='round,pad=0.35', facecolor=C_BG,
                       edgecolor='#CCCCCC', alpha=0.9))

ax_perm.set_xlabel('AUC', color=C_BLACK)
ax_perm.set_ylabel('Count', color=C_BLACK)
ax_perm.set_title('Permutation Test\n(within-subject label shuffling)',
                  fontsize=9.5)
ax_perm.legend(loc='upper left', frameon=False, fontsize=7.5)

fig1.savefig('figure1_roc.pdf', dpi=300, bbox_inches='tight',
             facecolor='white')
print("  Saved: figure1_roc.pdf")


# =============================================================================
# CELL 4 — FIGURE 2: CONFUSION MATRIX
#
# Single panel — RF primary.
# Raw counts + proportions in each cell.
# Accuracy, sensitivity, specificity annotated below.
# LR numbers in caption (not on figure).
# =============================================================================

print("Building Figure 2 — Confusion Matrix...")

fig2, ax_cm = plt.subplots(1, 1, figsize=(4.5, 4.2))

# Confusion matrix: [[TN, FP], [FN, TP]]
# Rows = true labels [Relational, Item]
# Cols = predicted labels [Relational, Item]
cm_norm = CM.astype(float) / CM.sum(axis=1, keepdims=True)

# Color map: white → C_ITEM blue (correct cells more blue = more correct)
cmap = LinearSegmentedColormap.from_list(
    'custom', ['white', C_ITEM], N=256
)

im = ax_cm.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect='equal')

# Cell annotations — counts and proportions
class_labels = ['Relational', 'Item']
for i in range(2):
    for j in range(2):
        count = CM[i, j]
        prop  = cm_norm[i, j]
        # Use white text on dark cells, dark text on light cells
        text_color = 'white' if prop > 0.55 else C_BLACK
        ax_cm.text(j, i,
                   f'{count:,}\n({prop:.2f})',
                   ha='center', va='center',
                   fontsize=11, fontweight='bold',
                   color=text_color)

# Axes
ax_cm.set_xticks([0, 1])
ax_cm.set_yticks([0, 1])
ax_cm.set_xticklabels(class_labels, fontsize=9)
ax_cm.set_yticklabels(class_labels, fontsize=9)
ax_cm.set_xlabel('Predicted Label', fontsize=9, labelpad=8)
ax_cm.set_ylabel('True Label', fontsize=9, labelpad=8)
ax_cm.set_title('Confusion Matrix — Random Forest\n'
                f'(LOSO, threshold = 0.5)',
                fontsize=9.5, pad=10)

# Colorbar
cbar = fig2.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
cbar.set_label('Proportion', fontsize=8)
cbar.ax.tick_params(labelsize=7)

# Metrics annotation below the matrix
metrics_text = (
    f'Accuracy = {ACCURACY:.3f} ({100*ACCURACY:.1f}%)    '
    f'Sensitivity = {SENS:.3f} ({100*SENS:.1f}%)    '
    f'Specificity = {SPEC:.3f} ({100*SPEC:.1f}%)'
)
fig2.text(0.5, -0.02, metrics_text,
          ha='center', va='top', fontsize=7.5, color=C_NEUTRAL)

fig2.savefig('figure2_confusion.pdf', dpi=300, bbox_inches='tight',
             facecolor='white')
print("  Saved: figure2_confusion.pdf")


# =============================================================================
# CELL 5 — FIGURE 3: SHAP FEATURE IMPORTANCE
#
# Two panels:
#   Left  — horizontal bar chart: mean |SHAP| per feature, sorted
#            group labels in gray, subtle group separators
#   Right — beeswarm: SHAP value per trial, colored by feature value
#            high feature value = blue, low = red-orange
# =============================================================================

print("Building Figure 3 — SHAP Feature Importance...")

# ── Feature groups and display names ─────────────────────────────────────────
FEATURE_GROUPS = {
    'obj_dwell_prop':           ('AOI Dwell',     'Object dwell proportion'),
    'scene_dwell_prop':         ('AOI Dwell',     'Scene dwell proportion'),
    'obj_fix_count':            ('AOI Dwell',     'Object fixation count'),
    'scene_fix_count':          ('AOI Dwell',     'Scene fixation count'),
    'obj_dwell_early_ms':       ('Temporal',      'Object dwell — early'),
    'obj_dwell_middle_ms':      ('Temporal',      'Object dwell — middle'),
    'obj_dwell_late_ms':        ('Temporal',      'Object dwell — late'),
    'scene_dwell_early_ms':     ('Temporal',      'Scene dwell — early'),
    'scene_dwell_middle_ms':    ('Temporal',      'Scene dwell — middle'),
    'scene_dwell_late_ms':      ('Temporal',      'Scene dwell — late'),
    'mean_fix_duration_ms':     ('Duration',      'Mean fixation duration'),
    'first_fix_latency_obj_ms': ('Duration',      'First fix latency — object'),
    'obj_scene_transitions':    ('Transitions',   'Object-scene transitions'),
    'transition_entropy':       ('Transitions',   'Transition entropy'),
    'obj_revisits':             ('Transitions',   'Object revisits'),
    'scene_revisits':           ('Transitions',   'Scene revisits'),
    'scanpath_length_deg':      ('Spatial',       'Scanpath length'),
    'fixation_dispersion':      ('Spatial',       'Fixation dispersion'),
    'saccade_amplitude_mean_deg':('Spatial',      'Saccade amplitude'),
}

# Sort features by SHAP importance (already sorted in shap_imp)
sorted_features = shap_imp['feature'].tolist()
sorted_shap     = shap_imp['mean_abs_shap'].tolist()
display_names   = [FEATURE_GROUPS[f][1] for f in sorted_features]
group_names     = [FEATURE_GROUPS[f][0] for f in sorted_features]

n_features = len(sorted_features)
y_pos      = np.arange(n_features)

fig3, (ax_bar, ax_bee) = plt.subplots(1, 2, figsize=(11, 6.5))
fig3.subplots_adjust(wspace=0.5)

# ── Left panel: bar chart ─────────────────────────────────────────────────────
# All bars same blue — group structure communicated via labels only
ax_bar.barh(y_pos, sorted_shap[::-1],
            color=C_ITEM, alpha=0.82,
            edgecolor='white', linewidth=0.3, height=0.7)

ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(display_names[::-1], fontsize=8)
ax_bar.set_xlabel('Mean |SHAP value|', fontsize=9)
ax_bar.set_title('Feature Importance\n(mean |SHAP|, all subjects)',
                 fontsize=9.5)

# Group separator lines — subtle horizontal rules between groups
groups_reversed = group_names[::-1]
prev_group = groups_reversed[0]
for idx, grp in enumerate(groups_reversed[1:], 1):
    if grp != prev_group:
        ax_bar.axhline(idx - 0.5, color='#DDDDDD',
                       linewidth=0.8, linestyle='-', zorder=0)
    prev_group = grp

# Group labels on left side
group_positions = {}
for idx, grp in enumerate(groups_reversed):
    if grp not in group_positions:
        group_positions[grp] = []
    group_positions[grp].append(idx)

for grp, positions in group_positions.items():
    mid = np.mean(positions)
    ax_bar.text(-ax_bar.get_xlim()[1] * 0.08, mid,
                grp, ha='right', va='center',
                fontsize=7, color=C_NEUTRAL, style='italic')

ax_bar.axvline(0, color=C_BLACK, linewidth=0.5)
ax_bar.invert_xaxis()
ax_bar.spines['left'].set_visible(False)
ax_bar.tick_params(left=False)

# ── Right panel: beeswarm ─────────────────────────────────────────────────────
# Manual beeswarm: for each feature, scatter SHAP values with
# y-jitter to avoid overplotting. Color = normalized feature value.

shap_cols   = [f'shap_{f}' for f in sorted_features]
feat_cols   = sorted_features
feat_matrix = feat_df[feat_cols].copy()

# Zero-fill transition_entropy for display (matches preprocessing)
if 'transition_entropy' in feat_matrix.columns:
    feat_matrix['transition_entropy'] = \
        feat_matrix['transition_entropy'].fillna(0)

# Custom colormap: red-orange (low) → blue (high)
bee_cmap = LinearSegmentedColormap.from_list(
    'bee', [C_REL, '#F7F7F7', C_ITEM], N=256
)

for feat_idx, feat in enumerate(sorted_features):
    shap_col = f'shap_{feat}'
    if shap_col not in shap_val.columns:
        continue

    shap_values_feat = shap_val[shap_col].values
    feat_values      = feat_matrix[feat].values

    # Normalize feature values to [0, 1] for coloring
    feat_min = np.nanpercentile(feat_values, 1)
    feat_max = np.nanpercentile(feat_values, 99)
    if feat_max > feat_min:
        feat_norm = np.clip(
            (feat_values - feat_min) / (feat_max - feat_min), 0, 1
        )
    else:
        feat_norm = np.full_like(feat_values, 0.5)

    # Y position = feature rank (reversed so top = most important)
    y_rank = n_features - 1 - feat_idx

    # Jitter to create beeswarm effect
    rng    = np.random.RandomState(42)
    jitter = rng.uniform(-0.3, 0.3, size=len(shap_values_feat))

    colors = bee_cmap(feat_norm)
    ax_bee.scatter(shap_values_feat,
                   y_rank + jitter,
                   c=colors, s=3, alpha=0.45,
                   linewidths=0, rasterized=True)

ax_bee.set_yticks(np.arange(n_features))
ax_bee.set_yticklabels(display_names[::-1], fontsize=8)
ax_bee.axvline(0, color=C_BLACK, linewidth=0.8, linestyle='-', alpha=0.6)
ax_bee.set_xlabel('SHAP value\n(positive = Item, negative = Relational)',
                  fontsize=9)
ax_bee.set_title('SHAP Values by Trial\n(color = feature value)',
                 fontsize=9.5)

# Colorbar for feature value
sm = plt.cm.ScalarMappable(cmap=bee_cmap,
                            norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar3 = fig3.colorbar(sm, ax=ax_bee, fraction=0.03, pad=0.02)
cbar3.set_ticks([0, 0.5, 1])
cbar3.set_ticklabels(['Low', 'Mid', 'High'], fontsize=7)
cbar3.set_label('Feature value', fontsize=8)

# Group separator lines on beeswarm too
for idx, grp in enumerate(groups_reversed[1:], 1):
    if grp != groups_reversed[idx - 1]:
        ax_bee.axhline(idx - 0.5, color='#DDDDDD',
                       linewidth=0.8, linestyle='-', zorder=0)

fig3.savefig('figure3_shap.pdf', dpi=300, bbox_inches='tight',
             facecolor='white')
print("  Saved: figure3_shap.pdf")


# =============================================================================
# CELL 6 — SUPPLEMENTARY FIGURE S1: PER-SUBJECT AUC DISTRIBUTION
#
# Single panel — sorted bar chart.
# RF bars (blue), LR dots (gray), low subjects flagged red.
# Same style as subject diagnostic but publication quality.
# =============================================================================

print("Building Supplementary Figure S1 — Per-Subject AUC...")

fig_s1, ax_s = plt.subplots(1, 1, figsize=(10, 4.0))

sorted_subj = subj_df.sort_values('auc_rf', ascending=False).reset_index(drop=True)
x_pos       = np.arange(len(sorted_subj))

bar_colors = [C_REL if row['low_auc'] else C_ITEM
              for _, row in sorted_subj.iterrows()]

# RF bars
ax_s.bar(x_pos, sorted_subj['auc_rf'],
         color=bar_colors, alpha=0.85,
         width=0.75, edgecolor='white', linewidth=0.3,
         label='Random Forest', zorder=2)

# LR dots
ax_s.scatter(x_pos, sorted_subj['auc_lr'],
             color=C_NEUTRAL, s=14, alpha=0.75,
             zorder=3, label='Logistic Regression')

# Reference lines
ax_s.axhline(0.5, color=C_BLACK, linestyle=':',
             linewidth=0.8, alpha=0.5, label='Chance (0.5)')
ax_s.axhline(subj_df['auc_rf'].mean(), color=C_ITEM,
             linestyle='--', linewidth=1.0, alpha=0.7,
             label=f'Mean AUC = {subj_df["auc_rf"].mean():.3f}')
ax_s.axhline(0.7, color=C_REL, linestyle=':',
             linewidth=0.8, alpha=0.6, label='Low threshold (0.70)')

# Annotate low-AUC subjects
for idx, row in sorted_subj[sorted_subj['low_auc']].iterrows():
    ax_s.text(idx, row['auc_rf'] + 0.013,
              f'S{int(row["subject_id"])}',
              ha='center', va='bottom',
              fontsize=6.5, color=C_REL, fontweight='bold')

n_above = (sorted_subj['auc_rf'] > 0.5).sum()
ax_s.text(0.98, 0.04,
          f'{n_above}/{N_SUBJ} subjects above chance individually',
          transform=ax_s.transAxes,
          ha='right', va='bottom', fontsize=8, color=C_NEUTRAL,
          bbox=dict(boxstyle='round,pad=0.3', facecolor=C_BG,
                    edgecolor='#CCCCCC', alpha=0.9))

ax_s.set_xticks([])
ax_s.set_ylabel('AUC (Random Forest)', fontsize=9)
ax_s.set_xlabel(f'Individual subjects (n = {N_SUBJ}, sorted by AUC)',
                fontsize=9)
ax_s.set_title('Per-Subject AUC — Random Forest (bars) and '
               'Logistic Regression (dots)',
               fontsize=9.5)
ax_s.set_ylim(0.40, 1.05)
ax_s.legend(loc='lower left', frameon=False, fontsize=7.5, ncol=2)

fig_s1.savefig('figure_s1_subject_auc.pdf', dpi=300, bbox_inches='tight',
               facecolor='white')
print("  Saved: figure_s1_subject_auc.pdf")


# =============================================================================
# CELL 7 — SUMMARY
# =============================================================================

print()
print("=" * 55)
print("ALL FIGURES COMPLETE")
print("=" * 55)
print()
print("  figure1_roc.pdf          — ROC curve + permutation null")
print("  figure2_confusion.pdf    — Confusion matrix (RF)")
print("  figure3_shap.pdf         — SHAP importance (bar + beeswarm)")
print("  figure_s1_subject_auc.pdf — Supplementary: per-subject AUC")
print()
print("  Download all four PDFs from the Colab Files panel.")
print()
print("  Publication-ready result:")
print(f"  AUC = {OBS_AUC:.3f} [95% CI: {CI_LO:.3f}–{CI_HI:.3f}], "
      f"permutation p = .005")
