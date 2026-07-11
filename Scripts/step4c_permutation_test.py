# =============================================================================
# STEP 4c — PERMUTATION TEST
# Eye-Tracking Memory Task Classifier — Whitlock Lab
#
# PURPOSE:
#   Determine whether the observed LOSO AUC is statistically above chance
#   by comparing it against a null distribution of AUCs obtained by
#   randomly shuffling task labels within each subject.
#
#   This is the final validation step before publication figures.
#   It answers: "Could we have obtained AUC = 0.861 by chance alone?"
#
# MODEL:
#   Random Forest only. Logistic Regression is the baseline in main results
#   but is not included in the permutation test — permutation validates
#   the primary model only.
#
# PARAMETERS:
#   N_PERMUTATIONS = 200   — minimum p-value = 1/201 = 0.005 (publication quality)
#   N_TREES_NULL   = 100   — 100 trees for null distribution (computational feasibility)
#
#   The observed AUC (0.861) was computed with 500 trees and is loaded
#   from loso_results_summary.json — it does not change.
#   Only the permuted-label null models use 100 trees.
#
#   Methods transparency note:
#   "The observed model used 500 trees. Permuted-label null models used
#   100 trees for computational feasibility while preserving the same
#   LOSO structure, within-subject shuffling, and preprocessing pipeline."
#
# HOW IT WORKS:
#   1. Load observed RF AUC from loso_results_summary.json
#   2. For each of N_PERMUTATIONS iterations:
#      a. Shuffle task labels WITHIN each subject — preserves subject-level
#         trial counts, class balance, and LOSO fold structure
#      b. Rerun LOSO with 100-tree RF on the shuffled labels
#      c. Record the null AUC
#   3. Compute p-value with plus-one correction:
#      p = (n_null >= observed + 1) / (N_PERMUTATIONS + 1)
#
# RUNTIME ESTIMATE:
#   200 permutations x 83 folds x 100 trees
#   scikit-learn RF does not use GPU — runtime depends on CPU cores.
#   n_jobs=-1 uses all available cores.
#   Typical Colab runtime: ~9-10 hours.
#
# INPUT:
#   feature_matrix_encoding.csv   — feature matrix from Step 2
#   loso_results_summary.json     — observed RF AUC from Step 4
#
# OUTPUT:
#   permutation_results.json      — p-value, null distribution stats
#   permutation_null_dist_rf.csv  — 200 null AUC values for Step 5 figures
#   Printed results report        — publication-ready p-value
#
# HOW TO RUN IN GOOGLE COLAB:
#   1. Upload feature_matrix_encoding.csv and loso_results_summary.json
#   2. Run all cells
#   3. Download permutation_results.json and permutation_null_dist_rf.csv
#   4. Send the full printed output back for documentation
#
# =============================================================================


# =============================================================================
# CELL 1 — IMPORTS AND CONFIGURATION
# =============================================================================

import pandas as pd
import numpy as np
import json
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble      import RandomForestClassifier
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute        import SimpleImputer
from sklearn.metrics       import roc_auc_score

print("=" * 65)
print("STEP 4c — PERMUTATION TEST")
print("Eye-Tracking Memory Task Classifier — Whitlock Lab")
print("=" * 65)
print()

# ── File paths ────────────────────────────────────────────────────────────────
INPUT_FEATURES  = "feature_matrix_encoding.csv"
INPUT_RESULTS   = "loso_results_summary.json"
OUTPUT_JSON     = "permutation_results.json"
OUTPUT_NULL_RF  = "permutation_null_dist_rf.csv"

# ── Parameters ────────────────────────────────────────────────────────────────
N_PERMUTATIONS  = 200    # minimum p = 1/201 = 0.005
N_TREES_NULL    = 100    # null models only — observed AUC used 500 trees
RANDOM_STATE    = 42     # for model fitting, same as Step 4

# ── Feature columns — identical to Step 4 ────────────────────────────────────
FEATURE_COLS = [
    'obj_dwell_prop',        'scene_dwell_prop',
    'obj_fix_count',         'scene_fix_count',
    'obj_dwell_early_ms',    'obj_dwell_middle_ms',    'obj_dwell_late_ms',
    'scene_dwell_early_ms',  'scene_dwell_middle_ms',  'scene_dwell_late_ms',
    'mean_fix_duration_ms',  'first_fix_latency_obj_ms',
    'obj_scene_transitions', 'transition_entropy',
    'obj_revisits',          'scene_revisits',
    'scanpath_length_deg',   'fixation_dispersion',    'saccade_amplitude_mean_deg',
]

# ── Load feature matrix ───────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FEATURES)

# Apply transition_entropy zero-fill — identical to Step 4
# NaN means zero transitions occurred; Shannon entropy is correctly 0
n_nan = df['transition_entropy'].isnull().sum()
df['transition_entropy'] = df['transition_entropy'].fillna(0)
print(f"transition_entropy NaN -> 0: {n_nan} trials filled")
print()

subjects    = sorted(df['subject_id'].unique())
n_subjects  = len(subjects)
subject_ids = df['subject_id'].values
true_labels = df['task_label'].values.copy()

# ── Load observed AUC from Step 4 ────────────────────────────────────────────
with open(INPUT_RESULTS, 'r') as f:
    step4_results = json.load(f)

observed_auc_rf = step4_results['random_forest']['pooled_auc']
ci_lower_rf     = step4_results['random_forest']['ci_lower']
ci_upper_rf     = step4_results['random_forest']['ci_upper']

print("Configuration:")
print(f"  Subjects:           {n_subjects}")
print(f"  Trials:             {len(df):,}")
print(f"  N permutations:     {N_PERMUTATIONS}")
print(f"  Trees (null):       {N_TREES_NULL}")
print(f"  Trees (observed):   500 (from Step 4, fixed)")
print(f"  Random state:       {RANDOM_STATE}")
print()
print(f"  Observed RF AUC:    {observed_auc_rf:.4f}")
print(f"  95% CI:             [{ci_lower_rf:.3f}-{ci_upper_rf:.3f}]")
print()


# =============================================================================
# CELL 2 — LOSO FUNCTION (RF ONLY)
#
# Identical preprocessing to Step 4:
#   - Imputation fit on train only
#   - Scaling fit on train only
#   - No leakage across fold boundary
# =============================================================================

def run_loso_rf_auc(feature_df, labels, subjects, subject_ids,
                    n_trees, random_state):
    """
    Run LOSO cross-validation and return pooled RF AUC.

    Parameters
    ----------
    feature_df  : pd.DataFrame — feature matrix
    labels      : np.ndarray  — task labels (real or shuffled)
    subjects    : list        — subject IDs
    subject_ids : np.ndarray  — subject ID per row (for fast masking)
    n_trees     : int         — number of trees in RF
    random_state: int         — reproducibility seed

    Returns
    -------
    float — pooled AUC across all held-out folds
    """
    all_true  = []
    all_probs = []

    for test_subject in subjects:

        train_mask = subject_ids != test_subject
        test_mask  = subject_ids == test_subject

        X_train = feature_df.loc[train_mask, FEATURE_COLS].values
        y_train = labels[train_mask]
        X_test  = feature_df.loc[test_mask,  FEATURE_COLS].values
        y_test  = labels[test_mask]

        # Skip fold if test set has only one class
        # (rare with within-subject shuffling but possible for small subjects)
        if len(np.unique(y_test)) < 2:
            continue

        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
            ('model',   RandomForestClassifier(
                n_estimators     = n_trees,
                class_weight     = 'balanced',
                min_samples_leaf = 5,
                random_state     = random_state,
                n_jobs           = -1,
            ))
        ])
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]

        all_true.extend(y_test.tolist())
        all_probs.extend(probs.tolist())

    return float(roc_auc_score(all_true, all_probs))


# =============================================================================
# CELL 3 — PERMUTATION LOOP
#
# Labels are shuffled WITHIN each subject to preserve:
#   - Same subjects in every permutation
#   - Same number of trials per subject
#   - Same Item/Relational class balance per subject
#   - Same LOSO fold structure
#
# Only the mapping between gaze features and labels is destroyed.
# =============================================================================

print("=" * 65)
print(f"PERMUTATION LOOP — {N_PERMUTATIONS} iterations")
print("=" * 65)
print()

null_aucs_rf = []
start_time   = time.time()

for perm_idx in range(N_PERMUTATIONS):

    # ── Shuffle labels within each subject ────────────────────────────────────
    rng             = np.random.RandomState(perm_idx)
    shuffled_labels = true_labels.copy()

    for subj in subjects:
        subj_mask = subject_ids == subj
        shuffled_labels[subj_mask] = rng.permutation(
            shuffled_labels[subj_mask]
        )

    # ── Run LOSO with shuffled labels ─────────────────────────────────────────
    null_auc = run_loso_rf_auc(
        feature_df   = df,
        labels       = shuffled_labels,
        subjects     = subjects,
        subject_ids  = subject_ids,
        n_trees      = N_TREES_NULL,
        random_state = RANDOM_STATE,
    )
    null_aucs_rf.append(null_auc)

    # ── Progress ──────────────────────────────────────────────────────────────
    if (perm_idx + 1) % 20 == 0 or perm_idx == 0:
        elapsed  = time.time() - start_time
        per_perm = elapsed / (perm_idx + 1)
        eta      = per_perm * (N_PERMUTATIONS - perm_idx - 1)
        print(f"  Permutation {perm_idx+1:>4}/{N_PERMUTATIONS} | "
              f"Null AUC: {null_auc:.3f} | "
              f"Elapsed: {elapsed/60:.1f}m | "
              f"ETA: {eta/60:.1f}m")

null_aucs_rf = np.array(null_aucs_rf)
total_time   = time.time() - start_time

print()
print(f"Permutation loop complete. Total time: {total_time/60:.1f} minutes.")
print()


# =============================================================================
# CELL 4 — COMPUTE P-VALUE AND PRINT RESULTS
#
# Plus-one correction: p = (n_exceed + 1) / (N_PERMUTATIONS + 1)
# Avoids p = 0.000. Minimum p = 1/201 = 0.00498, reported as p = .005
# =============================================================================

print("=" * 65)
print("RESULTS")
print("=" * 65)
print()

n_exceed = int(np.sum(null_aucs_rf >= observed_auc_rf))
p_value  = (n_exceed + 1) / (N_PERMUTATIONS + 1)
p_str    = '< .001' if p_value < 0.001 else f'= {p_value:.3f}'

print(f"  Observed RF AUC:   {observed_auc_rf:.4f}")
print(f"  Null mean AUC:     {null_aucs_rf.mean():.4f} +/- {null_aucs_rf.std():.4f}")
print(f"  Null max AUC:      {null_aucs_rf.max():.4f}")
print(f"  N permutations >= observed: {n_exceed} / {N_PERMUTATIONS}")
print(f"  p-value (plus-one):         {p_value:.4f}")
print()
print(f"  PUBLICATION-READY RESULT:")
print(f"  AUC = {observed_auc_rf:.3f} "
      f"[95% CI: {ci_lower_rf:.3f}-{ci_upper_rf:.3f}], "
      f"permutation p {p_str}")
print()


# =============================================================================
# CELL 5 — SAVE OUTPUTS
# =============================================================================

print("=" * 65)
print("SAVING OUTPUTS")
print("=" * 65)
print()

# ── Results JSON ──────────────────────────────────────────────────────────────
permutation_results = {
    'n_permutations':       N_PERMUTATIONS,
    'n_trees_null':         N_TREES_NULL,
    'n_trees_observed':     500,
    'random_state':         RANDOM_STATE,
    'shuffle_method':       'within-subject',
    'p_value_correction':   'plus-one',
    'random_forest': {
        'observed_auc':     observed_auc_rf,
        'ci_lower':         ci_lower_rf,
        'ci_upper':         ci_upper_rf,
        'null_mean_auc':    float(null_aucs_rf.mean()),
        'null_sd_auc':      float(null_aucs_rf.std()),
        'null_max_auc':     float(null_aucs_rf.max()),
        'n_exceed':         n_exceed,
        'p_value':          p_value,
        'p_string':         p_str,
        'publication_result': (
            f"AUC = {observed_auc_rf:.3f} "
            f"[95% CI: {ci_lower_rf:.3f}-{ci_upper_rf:.3f}], "
            f"permutation p {p_str}"
        ),
    },
}

with open(OUTPUT_JSON, 'w') as f:
    json.dump(permutation_results, f, indent=2)
print(f"  Saved: {OUTPUT_JSON}")

# ── Null distribution CSV (for Step 5 ROC figure) ────────────────────────────
pd.DataFrame({'null_auc_rf': null_aucs_rf}).to_csv(OUTPUT_NULL_RF, index=False)
print(f"  Saved: {OUTPUT_NULL_RF}")
print(f"    ({N_PERMUTATIONS} null AUC values)")
print()
print("  Download both files from the Colab Files panel.")
print("  Send the full output above back for documentation.")
print()
print("=" * 65)
print("PERMUTATION TEST COMPLETE")
print("=" * 65)
