# =============================================================================
# STEP 4 — CLASSIFIER
# Eye-Tracking Memory Task Classifier — Whitlock Lab
#
# PURPOSE:
#   Train and evaluate a binary classifier that predicts memory encoding
#   strategy (Item vs Relational) from eye movement features.
#   Uses Leave-One-Subject-Out (LOSO) cross-validation to ensure
#   zero participant-level data leakage and genuine generalisability.
#
# INPUT:
#   feature_matrix_encoding.csv — output of Step 2
#
# OUTPUTS:
#   loso_predictions.csv         — trial-level predictions from both models
#                                  (single source of truth for all figures)
#   loso_results_summary.json    — all metrics in one place
#   shap_values.csv              — SHAP values for all trials (all-subject model)
#   shap_feature_importance.csv  — SHAP feature importance table (ranked)
#   Printed results report       — publication-ready numbers
#
# MODELS:
#   Primary:  Random Forest (500 trees, class_weight=balanced)
#   Baseline: Logistic Regression
#
# VALIDATION:
#   Leave-One-Subject-Out (LOSO) — 83 folds, one per subject
#   Train on 82 subjects, test on 1, repeat for all 83 subjects
#
# PREPROCESSING (inside each fold — no leakage):
#   1. transition_entropy NaN → 0 BEFORE the loop (principled, not data-driven)
#      Zero transitions = zero entropy. This is behaviorally meaningful.
#      Median imputation would be wrong here — NaN means "no transitions",
#      not "missing data". Applied once to the full dataframe before LOSO.
#   2. Median imputation — fit on train, applied to test (for remaining NaNs)
#   3. StandardScaler   — fit on train, applied to test
#
# BOOTSTRAP CI:
#   Subject-level cluster bootstrap (primary, publication-quality).
#   Resamples subjects with replacement, pools their held-out predictions.
#   Respects nested structure of trials within subjects.
#   Trial-level bootstrap is NOT used — it underestimates variance.
#
# METRICS REPORTED:
#   Pooled AUC, Mean per-subject AUC ± SD
#   Bootstrap 95% CI on pooled AUC (2000 iterations)
#   Accuracy, Sensitivity (Item), Specificity (Relational)
#   Confusion matrix (threshold = 0.5)
#
# DECISION THRESHOLD: 0.5 for all hard-label metrics
#   Justified by balanced classes (~36 Item vs ~36 Relational per subject)
#
# IMPORTANT NOTES:
#   - p-values from Step 3 sanity check are NOT reported as paper results
#     (trial-level tests inflate df due to nesting within subjects)
#   - Bootstrap CI is computed here (subject-level cluster bootstrap).
#   - Permutation testing is deferred to the final validation step.
#   - SHAP is computed on a model trained on ALL 83 subjects for maximum
#     stability — this is interpretation only, not performance reporting
#
# HOW TO RUN IN GOOGLE COLAB:
#   1. Upload feature_matrix_encoding.csv
#   2. Run Cell 1 (imports — installs shap if needed)
#   3. Run Cell 2 (configuration and data loading)
#   4. Run Cell 3 (LOSO loop — ~3-5 minutes)
#   5. Run Cell 4 (metrics and results report)
#   6. Run Cell 5 (bootstrap CI)
#   7. Run Cell 6 (SHAP)
#   8. Run Cell 7 (save everything)
#   9. Download all output files from the Files panel
#
# =============================================================================


# =============================================================================
# CELL 1 — IMPORTS
# Install shap if not already available (Colab may need this)
# =============================================================================

# Uncomment the line below if shap is not installed in your Colab environment
# !pip install shap -q

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble          import RandomForestClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.pipeline          import Pipeline
from sklearn.preprocessing     import StandardScaler
from sklearn.impute            import SimpleImputer
from sklearn.metrics           import (roc_auc_score, confusion_matrix,
                                        accuracy_score)
import shap

print("=" * 65)
print("STEP 4 — CLASSIFIER")
print("Eye-Tracking Memory Task Classifier — Whitlock Lab")
print("=" * 65)
print()


# =============================================================================
# CELL 2 — CONFIGURATION AND DATA LOADING
# All parameters defined in one place.
# =============================================================================

# ── File paths ────────────────────────────────────────────────────────────────
INPUT_PATH           = "feature_matrix_encoding.csv"
OUTPUT_PREDICTIONS   = "loso_predictions.csv"
OUTPUT_RESULTS       = "loso_results_summary.json"
OUTPUT_SHAP_VALUES   = "shap_values.csv"
OUTPUT_SHAP_IMPORTANCE = "shap_feature_importance.csv"

# ── Model parameters ──────────────────────────────────────────────────────────
RANDOM_STATE     = 42       # fixed for full reproducibility
DECISION_THRESHOLD = 0.5    # for confusion matrix, accuracy, sensitivity, specificity
N_BOOTSTRAP      = 2000     # iterations for bootstrap CI
N_TREES          = 500      # Random Forest trees per fold

# ── Feature and metadata columns ─────────────────────────────────────────────
METADATA_COLS = ['subject_id', 'trial_id', 'task', 'task_label']

FEATURE_COLS = [
    # AOI dwell
    'obj_dwell_prop',        'scene_dwell_prop',
    'obj_fix_count',         'scene_fix_count',
    # Temporal dwell (thirds)
    'obj_dwell_early_ms',    'obj_dwell_middle_ms',    'obj_dwell_late_ms',
    'scene_dwell_early_ms',  'scene_dwell_middle_ms',  'scene_dwell_late_ms',
    # Duration and latency
    'mean_fix_duration_ms',  'first_fix_latency_obj_ms',
    # Transitions
    'obj_scene_transitions', 'transition_entropy',
    'obj_revisits',          'scene_revisits',
    # Spatial
    'scanpath_length_deg',   'fixation_dispersion',    'saccade_amplitude_mean_deg',
]

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH)

subjects = sorted(df['subject_id'].unique())
n_subjects = len(subjects)

print("Configuration:")
print(f"  Input:              {INPUT_PATH}")
print(f"  Features:           {len(FEATURE_COLS)}")
print(f"  Subjects (folds):   {n_subjects}")
print(f"  Total trials:       {len(df):,}")
print(f"  ITEM trials:        {(df['task_label']==1).sum():,}")
print(f"  RELATIONAL trials:  {(df['task_label']==0).sum():,}")
print(f"  Decision threshold: {DECISION_THRESHOLD}")
print(f"  Random state:       {RANDOM_STATE}")
print()

# =============================================================================
# TRANSITION_ENTROPY: REPLACE NaN WITH 0 BEFORE LOSO
#
# This is the only feature that gets principled zero-filling rather than
# median imputation. Here's why:
#
#   NaN in transition_entropy means the trial had zero cross-AOI transitions.
#   The participant never switched between the object and the scene.
#   This is a real, meaningful behavioral pattern — predominantly Item task
#   participants who fixated only the object throughout encoding.
#
#   Replacing with the median entropy would be actively wrong: it would make
#   a "never looked at scene" trial look like a moderately diverse scanner.
#
#   Replacing with 0 is correct: zero transitions = zero transition entropy.
#   Shannon H is mathematically 0 when there is only one outcome.
#
#   first_fix_latency_obj_ms has only 6 NaNs and is genuinely missing data
#   (participant never looked at object) — median imputation remains for that.
#
# This is applied ONCE to the full dataframe BEFORE the LOSO loop.
# It is not data-driven, so applying it before the loop does not cause leakage.
# =============================================================================

n_entropy_nan_before = df['transition_entropy'].isnull().sum()
df['transition_entropy'] = df['transition_entropy'].fillna(0)
n_entropy_nan_after = df['transition_entropy'].isnull().sum()

print(f"transition_entropy NaN → 0:")
print(f"  NaN before: {n_entropy_nan_before} trials ({100*n_entropy_nan_before/len(df):.1f}%)")
print(f"  NaN after:  {n_entropy_nan_after} trials")
print(f"  Interpretation: these trials had zero cross-AOI transitions,")
print(f"  so Shannon entropy is correctly 0.")
print()


# =============================================================================
# CELL 3 — LOSO CROSS-VALIDATION LOOP
#
# For each subject:
#   1. Split data into train (82 subjects) and test (1 subject)
#   2. Build preprocessing pipeline (impute → scale) fit only on train
#   3. Fit Random Forest and Logistic Regression on train
#   4. Predict probabilities on test
#   5. Store predictions, true labels, and per-subject AUC
#
# CRITICAL — no leakage:
#   The imputer and scaler are fit INSIDE the pipeline on training data only.
#   They are never fit on or informed by the test subject's data.
# =============================================================================

print("=" * 65)
print("LOSO CROSS-VALIDATION")
print(f"Running {n_subjects} folds...")
print("=" * 65)

# Storage for all held-out predictions across folds
all_records = []

# Per-subject AUC storage
subject_aucs_rf = {}
subject_aucs_lr = {}

for fold_idx, test_subject in enumerate(subjects):

    # ── Split ─────────────────────────────────────────────────────────────────
    train_df = df[df['subject_id'] != test_subject].copy()
    test_df  = df[df['subject_id'] == test_subject].copy()

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df['task_label'].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df['task_label'].values

    # ── Build pipelines (imputer + scaler fit on train only) ──────────────────
    # Random Forest pipeline
    pipeline_rf = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model',   RandomForestClassifier(
            n_estimators    = N_TREES,
            class_weight    = 'balanced',
            min_samples_leaf= 5,
            random_state    = RANDOM_STATE,
            n_jobs          = -1,
        ))
    ])

    # Logistic Regression pipeline
    pipeline_lr = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model',   LogisticRegression(
            C            = 1.0,
            class_weight = 'balanced',
            max_iter     = 1000,
            random_state = RANDOM_STATE,
            solver       = 'lbfgs',
        ))
    ])

    # ── Fit both models on training data ─────────────────────────────────────
    pipeline_rf.fit(X_train, y_train)
    pipeline_lr.fit(X_train, y_train)

    # ── Predict probabilities for positive class (Item = 1) ───────────────────
    probs_rf = pipeline_rf.predict_proba(X_test)[:, 1]
    probs_lr = pipeline_lr.predict_proba(X_test)[:, 1]

    # ── Hard labels at 0.5 threshold ─────────────────────────────────────────
    preds_rf = (probs_rf >= DECISION_THRESHOLD).astype(int)
    preds_lr = (probs_lr >= DECISION_THRESHOLD).astype(int)

    # ── Per-subject AUC (skip if only one class present in test set) ──────────
    if len(np.unique(y_test)) == 2:
        subject_aucs_rf[test_subject] = roc_auc_score(y_test, probs_rf)
        subject_aucs_lr[test_subject] = roc_auc_score(y_test, probs_lr)
    else:
        # Edge case: subject has only one task (shouldn't happen after preprocessing)
        subject_aucs_rf[test_subject] = np.nan
        subject_aucs_lr[test_subject] = np.nan

    # ── Store trial-level predictions ─────────────────────────────────────────
    for i, (trial_idx, row) in enumerate(test_df.iterrows()):
        all_records.append({
            'subject_id':       int(row['subject_id']),
            'trial_id':         int(row['trial_id']),
            'task':             row['task'],
            'true_label':       int(row['task_label']),
            # Random Forest
            'prob_rf':          float(probs_rf[i]),
            'pred_rf':          int(preds_rf[i]),
            # Logistic Regression
            'prob_lr':          float(probs_lr[i]),
            'pred_lr':          int(preds_lr[i]),
        })

    # ── Progress ──────────────────────────────────────────────────────────────
    if (fold_idx + 1) % 10 == 0 or fold_idx == 0:
        print(f"  Fold {fold_idx+1:>3}/{n_subjects} — "
              f"Subject {test_subject:>3} — "
              f"RF AUC: {subject_aucs_rf[test_subject]:.3f} | "
              f"LR AUC: {subject_aucs_lr[test_subject]:.3f}")

print()
print(f"LOSO complete. {len(all_records):,} trial predictions stored.")
print()


# =============================================================================
# CELL 4 — COMPUTE METRICS AND PRINT RESULTS REPORT
#
# From the pooled held-out predictions:
#   - Pooled AUC (primary headline metric)
#   - Per-subject AUC mean ± SD
#   - Confusion matrix
#   - Accuracy, Sensitivity (Item), Specificity (Relational)
# =============================================================================

print("=" * 65)
print("COMPUTING METRICS")
print("=" * 65)

# Build predictions dataframe
pred_df = pd.DataFrame(all_records)

# True labels and predicted probabilities (pooled across all folds)
y_true   = pred_df['true_label'].values
probs_rf_all = pred_df['prob_rf'].values
probs_lr_all = pred_df['prob_lr'].values
preds_rf_all = pred_df['pred_rf'].values
preds_lr_all = pred_df['pred_lr'].values

def compute_metrics(y_true, y_probs, y_preds, model_name, subject_aucs):
    """
    Compute all metrics for one model from pooled LOSO predictions.
    Returns a dict of all metrics.
    """
    # ── Pooled AUC ────────────────────────────────────────────────────────────
    pooled_auc = roc_auc_score(y_true, y_probs)

    # ── Per-subject AUC ───────────────────────────────────────────────────────
    valid_aucs = [v for v in subject_aucs.values() if not np.isnan(v)]
    mean_subj_auc = np.mean(valid_aucs)
    sd_subj_auc   = np.std(valid_aucs, ddof=1)
    min_subj_auc  = np.min(valid_aucs)
    max_subj_auc  = np.max(valid_aucs)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    # Rows = true labels, Cols = predicted labels
    # Order: [Relational (0), Item (1)]
    cm = confusion_matrix(y_true, y_preds, labels=[0, 1])
    TN, FP = cm[0, 0], cm[0, 1]   # true Relational, false Item
    FN, TP = cm[1, 0], cm[1, 1]   # false Relational, true Item

    # ── Accuracy, Sensitivity, Specificity ───────────────────────────────────
    accuracy    = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else np.nan  # Item recall
    specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan  # Relational recall

    print(f"\n{'─'*55}")
    print(f"  MODEL: {model_name}")
    print(f"{'─'*55}")
    print(f"  Pooled AUC:              {pooled_auc:.4f}")
    print(f"  Mean per-subject AUC:    {mean_subj_auc:.4f} ± {sd_subj_auc:.4f}")
    print(f"  Per-subject AUC range:   [{min_subj_auc:.3f}, {max_subj_auc:.3f}]")
    print()
    print(f"  Accuracy:                {accuracy:.4f}  ({100*accuracy:.1f}%)")
    print(f"  Sensitivity (Item):      {sensitivity:.4f}  ({100*sensitivity:.1f}%)")
    print(f"  Specificity (Relational):{specificity:.4f}  ({100*specificity:.1f}%)")
    print()
    print(f"  Confusion Matrix (rows=true, cols=predicted):")
    print(f"                    Pred Relational   Pred Item")
    print(f"  True Relational        {TN:>7}        {FP:>7}")
    print(f"  True Item              {FN:>7}        {TP:>7}")
    print()
    print(f"  Publication-ready:")
    print(f"  AUC = {pooled_auc:.3f} (mean per-subject = {mean_subj_auc:.3f} ± {sd_subj_auc:.3f})")
    print(f"  Accuracy = {100*accuracy:.1f}%, Sensitivity = {100*sensitivity:.1f}%, "
          f"Specificity = {100*specificity:.1f}%")

    return {
        'model':              model_name,
        'pooled_auc':         float(pooled_auc),
        'mean_subject_auc':   float(mean_subj_auc),
        'sd_subject_auc':     float(sd_subj_auc),
        'min_subject_auc':    float(min_subj_auc),
        'max_subject_auc':    float(max_subj_auc),
        'accuracy':           float(accuracy),
        'sensitivity':        float(sensitivity),
        'specificity':        float(specificity),
        'confusion_matrix':   cm.tolist(),
        'TP': int(TP), 'TN': int(TN), 'FP': int(FP), 'FN': int(FN),
        'per_subject_aucs':   {str(k): float(v)
                               for k, v in subject_aucs.items()},
    }

print()
results_rf = compute_metrics(
    y_true, probs_rf_all, preds_rf_all,
    'Random Forest', subject_aucs_rf
)
results_lr = compute_metrics(
    y_true, probs_lr_all, preds_lr_all,
    'Logistic Regression', subject_aucs_lr
)


# =============================================================================
# CELL 5 — SUBJECT-LEVEL CLUSTER BOOTSTRAP CONFIDENCE INTERVALS
#
# WHY SUBJECT-LEVEL, NOT TRIAL-LEVEL:
#   Trials are nested inside subjects. Resampling individual trials treats
#   them as independent observations, which they are not — trials from the
#   same subject share individual differences in scanning behavior. This
#   inflates the effective sample size and produces CI that are too narrow.
#
#   The correct approach for nested data is cluster bootstrap by subject:
#     - Resample SUBJECTS with replacement (83 → 83, some appear multiple times)
#     - Include ALL held-out LOSO predictions for each sampled subject
#     - Compute AUC from the pooled predictions of the resampled subjects
#     - Repeat N_BOOTSTRAP times
#
#   This respects the data structure and is appropriate for publication.
#
# NOTE: Each subject's predictions come from the LOSO fold where they were
# the held-out subject — so these are always out-of-sample predictions.
# =============================================================================

print()
print("=" * 65)
print("SUBJECT-LEVEL CLUSTER BOOTSTRAP CI")
print(f"Running {N_BOOTSTRAP} iterations (resampling subjects)...")
print("=" * 65)

def cluster_bootstrap_auc_ci(pred_df, prob_col, n_bootstrap, random_state):
    """
    Subject-level cluster bootstrap CI for AUC.

    Resamples subjects with replacement. For each bootstrap iteration,
    pools the held-out LOSO predictions of the sampled subjects and
    computes AUC. Returns (ci_lower, ci_upper, bootstrap_aucs).

    Parameters
    ----------
    pred_df      : DataFrame with columns subject_id, true_label, prob_col
    prob_col     : str — column name for predicted probabilities
    n_bootstrap  : int — number of bootstrap iterations
    random_state : int — for reproducibility
    """
    rng      = np.random.RandomState(random_state)
    subjects = pred_df['subject_id'].unique()
    n_subj   = len(subjects)

    bootstrap_aucs = []

    for _ in range(n_bootstrap):
        # Resample subjects with replacement
        sampled_subjects = rng.choice(subjects, size=n_subj, replace=True)

        # Pool all held-out predictions for the sampled subjects
        # (a subject drawn twice contributes its predictions twice)
        frames = []
        for subj in sampled_subjects:
            frames.append(pred_df[pred_df['subject_id'] == subj])
        boot_df = pd.concat(frames, ignore_index=True)

        y_boot = boot_df['true_label'].values
        p_boot = boot_df[prob_col].values

        # Skip if only one class in this resample (rare edge case)
        if len(np.unique(y_boot)) < 2:
            continue

        bootstrap_aucs.append(roc_auc_score(y_boot, p_boot))

    bootstrap_aucs = np.array(bootstrap_aucs)
    ci_lower = np.percentile(bootstrap_aucs, 2.5)
    ci_upper = np.percentile(bootstrap_aucs, 97.5)

    return float(ci_lower), float(ci_upper), bootstrap_aucs


ci_lower_rf, ci_upper_rf, bs_aucs_rf = cluster_bootstrap_auc_ci(
    pred_df, 'prob_rf', N_BOOTSTRAP, RANDOM_STATE
)
ci_lower_lr, ci_upper_lr, bs_aucs_lr = cluster_bootstrap_auc_ci(
    pred_df, 'prob_lr', N_BOOTSTRAP, RANDOM_STATE
)

print(f"\n  Random Forest:")
print(f"    AUC = {results_rf['pooled_auc']:.3f} "
      f"[95% CI: {ci_lower_rf:.3f}–{ci_upper_rf:.3f}]")
print(f"    (subject-level cluster bootstrap, n={N_BOOTSTRAP} iterations)")

print(f"\n  Logistic Regression:")
print(f"    AUC = {results_lr['pooled_auc']:.3f} "
      f"[95% CI: {ci_lower_lr:.3f}–{ci_upper_lr:.3f}]")
print(f"    (subject-level cluster bootstrap, n={N_BOOTSTRAP} iterations)")

print()
print("  Publication-ready result (Random Forest):")
print(f"  AUC = {results_rf['pooled_auc']:.3f} "
      f"[95% CI: {ci_lower_rf:.3f}–{ci_upper_rf:.3f}], "
      f"permutation p = TBD")

# Store CIs back into results dicts
results_rf['ci_lower'] = ci_lower_rf
results_rf['ci_upper'] = ci_upper_rf
results_rf['bootstrap_aucs'] = bs_aucs_rf.tolist()
results_lr['ci_lower'] = ci_lower_lr
results_lr['ci_upper'] = ci_upper_lr
results_lr['bootstrap_aucs'] = bs_aucs_lr.tolist()


# =============================================================================
# CELL 6 — SHAP FEATURE IMPORTANCE
#
# Fit a final Random Forest on ALL 83 subjects combined.
# This model is for interpretation only — not for performance reporting.
# TreeExplainer computes SHAP values efficiently for tree-based models.
#
# SHAP values are saved to CSV for use in the figures script (Step 5).
# =============================================================================

print()
print("=" * 65)
print("SHAP FEATURE IMPORTANCE")
print("Training all-subject model for interpretation...")
print("=" * 65)

# ── Fit pipeline on all subjects ──────────────────────────────────────────────
pipeline_shap = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   RandomForestClassifier(
        n_estimators    = N_TREES,
        class_weight    = 'balanced',
        min_samples_leaf= 5,
        random_state    = RANDOM_STATE,
        n_jobs          = -1,
    ))
])

X_all = df[FEATURE_COLS].values
y_all = df['task_label'].values

pipeline_shap.fit(X_all, y_all)
print("  All-subject model trained.")

# ── Transform features through preprocessing steps ───────────────────────────
# SHAP needs the transformed feature matrix (post imputation and scaling)
X_transformed = pipeline_shap.named_steps['imputer'].transform(X_all)
X_transformed = pipeline_shap.named_steps['scaler'].transform(X_transformed)

# ── Compute SHAP values ───────────────────────────────────────────────────────
rf_model  = pipeline_shap.named_steps['model']
explainer = shap.TreeExplainer(rf_model)
shap_vals = explainer.shap_values(X_transformed)

# Handle SHAP output format — varies across SHAP versions:
#
#   Older SHAP (< 0.40):  returns list [class_0_array, class_1_array]
#                         each array is shape (n_samples, n_features)
#
#   Newer SHAP (>= 0.40): may return a single 3D array
#                         shape (n_samples, n_features, n_classes)
#
#   We always want class 1 (Item = positive class).
#
if isinstance(shap_vals, list):
    # Older format: list of arrays per class
    shap_vals_item = shap_vals[1]
elif len(shap_vals.shape) == 3:
    # Newer format: 3D array, last dimension is classes
    shap_vals_item = shap_vals[:, :, 1]
else:
    # Single 2D array — already the values we want
    shap_vals_item = shap_vals

print(f"  SHAP values computed: shape = {shap_vals_item.shape}")

# ── Mean absolute SHAP per feature (global importance) ───────────────────────
mean_abs_shap = np.abs(shap_vals_item).mean(axis=0)
shap_importance = pd.DataFrame({
    'feature':          FEATURE_COLS,
    'mean_abs_shap':    mean_abs_shap,
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

print()
print("  Feature importance (mean |SHAP|), top 10:")
print(f"  {'Rank':<6} {'Feature':<35} {'Mean |SHAP|':>12}")
print("  " + "-" * 56)
for i, row in shap_importance.head(10).iterrows():
    print(f"  {i+1:<6} {row['feature']:<35} {row['mean_abs_shap']:>12.4f}")
print()


# =============================================================================
# CELL 7 — SAVE ALL OUTPUTS
#
# Four files:
#   loso_predictions.csv         — trial-level predictions (source of truth)
#   loso_results_summary.json    — all metrics
#   shap_values.csv              — SHAP values per trial per feature
#   shap_feature_importance.csv  — feature importance ranking (mean |SHAP|)
# =============================================================================

print("=" * 65)
print("SAVING OUTPUTS")
print("=" * 65)

# ── 1. Trial-level predictions ────────────────────────────────────────────────
pred_df.to_csv(OUTPUT_PREDICTIONS, index=False)
print(f"  Saved: {OUTPUT_PREDICTIONS}")
print(f"    {len(pred_df):,} rows × {len(pred_df.columns)} columns")
print(f"    Columns: {list(pred_df.columns)}")
print()

# ── 2. Results summary JSON ───────────────────────────────────────────────────
results_summary = {
    'dataset': {
        'n_subjects':         n_subjects,
        'n_trials':           len(df),
        'n_item_trials':      int((df['task_label']==1).sum()),
        'n_relational_trials':int((df['task_label']==0).sum()),
        'n_features':         len(FEATURE_COLS),
        'feature_cols':       FEATURE_COLS,
        'decision_threshold': DECISION_THRESHOLD,
        'random_state':       RANDOM_STATE,
    },
    'random_forest': results_rf,
    'logistic_regression': results_lr,
}

with open(OUTPUT_RESULTS, 'w') as f:
    json.dump(results_summary, f, indent=2)
print(f"  Saved: {OUTPUT_RESULTS}")
print()

# ── 3. SHAP values ────────────────────────────────────────────────────────────
shap_df = pd.DataFrame(
    shap_vals_item,
    columns=[f'shap_{f}' for f in FEATURE_COLS]
)
# Add metadata for reference
shap_df.insert(0, 'subject_id', df['subject_id'].values)
shap_df.insert(1, 'trial_id',   df['trial_id'].values)
shap_df.insert(2, 'task_label', df['task_label'].values)

shap_df.to_csv(OUTPUT_SHAP_VALUES, index=False)
print(f"  Saved: {OUTPUT_SHAP_VALUES}")
print(f"    {shap_df.shape[0]:,} trials × {shap_df.shape[1]} columns")
print()

# ── 4. SHAP feature importance summary ───────────────────────────────────────
shap_importance.to_csv(OUTPUT_SHAP_IMPORTANCE, index=False)
print(f"  Saved: {OUTPUT_SHAP_IMPORTANCE}")
print()

# ── 5. Bootstrap AUC distributions ───────────────────────────────────────────
# Saved as separate CSVs for easy plotting in Step 5.
# The same values are also stored inside loso_results_summary.json.
pd.DataFrame({'bootstrap_auc_rf': bs_aucs_rf}).to_csv(
    'bootstrap_auc_random_forest.csv', index=False
)
pd.DataFrame({'bootstrap_auc_lr': bs_aucs_lr}).to_csv(
    'bootstrap_auc_logistic_regression.csv', index=False
)
print(f"  Saved: bootstrap_auc_random_forest.csv")
print(f"  Saved: bootstrap_auc_logistic_regression.csv")
print(f"    ({N_BOOTSTRAP} bootstrap AUC samples per model)")
print()

# ── Final publication-ready summary ──────────────────────────────────────────
print("=" * 65)
print("RESULTS SUMMARY — COPY THIS FOR DOCUMENTATION")
print("=" * 65)
print()
print("  RANDOM FOREST (primary model):")
print(f"    AUC = {results_rf['pooled_auc']:.3f} "
      f"[95% CI: {ci_lower_rf:.3f}–{ci_upper_rf:.3f}]")
print(f"    Mean per-subject AUC = "
      f"{results_rf['mean_subject_auc']:.3f} ± {results_rf['sd_subject_auc']:.3f}")
print(f"    Accuracy    = {100*results_rf['accuracy']:.1f}%")
print(f"    Sensitivity = {100*results_rf['sensitivity']:.1f}%  (Item correctly identified)")
print(f"    Specificity = {100*results_rf['specificity']:.1f}%  (Relational correctly identified)")
print()
print("  LOGISTIC REGRESSION (baseline):")
print(f"    AUC = {results_lr['pooled_auc']:.3f} "
      f"[95% CI: {ci_lower_lr:.3f}–{ci_upper_lr:.3f}]")
print(f"    Mean per-subject AUC = "
      f"{results_lr['mean_subject_auc']:.3f} ± {results_lr['sd_subject_auc']:.3f}")
print(f"    Accuracy    = {100*results_lr['accuracy']:.1f}%")
print(f"    Sensitivity = {100*results_lr['sensitivity']:.1f}%")
print(f"    Specificity = {100*results_lr['specificity']:.1f}%")
print()
print("  Download all six output files from the Colab Files panel:")
print("    loso_predictions.csv")
print("    loso_results_summary.json")
print("    shap_values.csv")
print("    shap_feature_importance.csv")
print("    bootstrap_auc_random_forest.csv")
print("    bootstrap_auc_logistic_regression.csv")
print("  Send the full output above back for documentation.")
print()
print("=" * 65)
print("CLASSIFIER COMPLETE — READY FOR FIGURES (STEP 5)")
print("=" * 65)
