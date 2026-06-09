# =============================================================================
# STEP 1 — PREPROCESSING
# Eye-Tracking Memory Task Classifier — Whitlock Lab
#
# PURPOSE:
#   Clean the raw encoding fixation data and produce a verified, analysis-ready
#   dataset for feature extraction. Every exclusion is logged with exact counts
#   so the methods section can be written directly from this output.
#
# INPUT:
#   Item_Relational_Encoding_Data.csv — raw fixation-level data, never modified
#
# OUTPUT:
#   encoding_data_clean.csv — cleaned fixation-level data, one row per fixation
#   Printed exclusion report — documents every exclusion decision with counts
#
# DECISIONS ENCODED HERE (all confirmed by Whitlock):
#   - Filter to target==1 trials only (resolves 3:1 class imbalance)
#   - Remove trials with fewer than 3 fixations
#   - Exclude subjects with fewer than 23 valid trials (65% of 36 max)
#   - No fixation duration filter (confirmed not applicable to this project)
#
# HOW TO RUN IN GOOGLE COLAB:
#   1. Upload Item_Relational_Encoding_Data.csv to Colab (Files panel, left sidebar)
#   2. Paste this entire script into a code cell
#   3. Run the cell
#   4. Download encoding_data_clean.csv from the Files panel
#   5. Copy the printed exclusion report and send back for documentation
#
# =============================================================================


# =============================================================================
# CELL 1 — IMPORTS
# All libraries are pre-installed in Google Colab. No pip installs needed.
# =============================================================================

import pandas as pd
import numpy as np

print("=" * 65)
print("STEP 1 — PREPROCESSING")
print("Eye-Tracking Memory Task Classifier — Whitlock Lab")
print("=" * 65)
print()


# =============================================================================
# CELL 2 — CONFIGURATION
# All key parameters are defined here in one place.
# To change any threshold, change it here — not buried in the code below.
# =============================================================================

# Path to the raw data file
# If running in Colab and the file is uploaded, this path is correct as-is.
RAW_DATA_PATH = "Item_Relational_Encoding_Data.csv"

# Path for the cleaned output file
OUTPUT_PATH = "encoding_data_clean.csv"

# Minimum number of fixations a trial must have to be included.
# Trials with fewer fixations than this are excluded entirely.
# Decision: Whitlock confirmed remove trials with 1 or 2 fixations → keep >= 3
MIN_FIXATIONS_PER_TRIAL = 3

# Minimum number of valid trials a subject must have to be included.
# Decision: Whitlock confirmed 65% of 36 maximum trials = 23.4, rounded to 23.
MIN_TRIALS_PER_SUBJECT = 23

# The target column value that marks associate/target trials.
# target==1 means this encoded object goes on to be the correct answer at test.
# Decision: filter to target==1 resolves the 3:1 class imbalance.
TARGET_VALUE = 1

print("Configuration:")
print(f"  Raw data path:            {RAW_DATA_PATH}")
print(f"  Output path:              {OUTPUT_PATH}")
print(f"  Min fixations per trial:  {MIN_FIXATIONS_PER_TRIAL}")
print(f"  Min trials per subject:   {MIN_TRIALS_PER_SUBJECT}")
print(f"  Target filter value:      target == {TARGET_VALUE}")
print()


# =============================================================================
# CELL 3 — LOAD RAW DATA
# Load the raw CSV exactly as received. Print a full audit of the raw state
# so we have a documented baseline before any exclusions are applied.
# =============================================================================

print("=" * 65)
print("LOADING RAW DATA")
print("=" * 65)

df_raw = pd.read_csv(RAW_DATA_PATH)

# Count raw trials (before any filtering) at the trial level
# A trial is uniquely identified by Subject + Trial combination
raw_trials = df_raw.groupby(['Subject', 'Trial', 'Task']).ngroups

print(f"Raw fixation rows loaded:  {len(df_raw):,}")
print(f"Raw unique subjects:       {df_raw['Subject'].nunique()}")
print(f"Raw unique trials:         {raw_trials:,}")
print()
print("Raw trial counts by task:")

# Show per-task breakdown at the trial level
raw_trial_counts = (
    df_raw
    .drop_duplicates(['Subject', 'Trial', 'Task'])
    .groupby('Task')
    .size()
)
for task, count in raw_trial_counts.items():
    print(f"  {task}: {count:,} trials")

print()


# =============================================================================
# CELL 4 — STEP 1: FILTER TO TARGET TRIALS ONLY
#
# WHY:
#   The raw data has 3x more Relational trials than Item trials because the
#   Relational task requires 3 encoding trials to produce 1 test trial
#   (1 associate + 2 foils). Item task always has target==1.
#
#   Filtering to target==1 keeps only the trials where the encoded object
#   goes on to be the correct answer at test. This:
#     (a) Resolves the 3:1 class imbalance (~36 Item vs ~36 Relational per subject)
#     (b) Makes trials theoretically comparable across tasks
#
# =============================================================================

print("=" * 65)
print("STEP 1: FILTER TO TARGET TRIALS (target == 1)")
print("=" * 65)

df_step1 = df_raw[df_raw['target'] == TARGET_VALUE].copy()

# Count what was removed
removed_fixations_step1 = len(df_raw) - len(df_step1)
removed_trials_step1    = (
    raw_trials -
    df_step1.groupby(['Subject', 'Trial', 'Task']).ngroups
)

print(f"Fixation rows before: {len(df_raw):,}")
print(f"Fixation rows after:  {len(df_step1):,}")
print(f"Rows removed:         {removed_fixations_step1:,} "
      f"({100 * removed_fixations_step1 / len(df_raw):.1f}%)")
print()
print(f"Trials before: {raw_trials:,}")
print(f"Trials after:  {df_step1.groupby(['Subject', 'Trial', 'Task']).ngroups:,}")
print(f"Trials removed: {removed_trials_step1:,}")
print()
print("Trial counts after target filter:")

step1_trial_counts = (
    df_step1
    .drop_duplicates(['Subject', 'Trial', 'Task'])
    .groupby('Task')
    .size()
)
for task, count in step1_trial_counts.items():
    print(f"  {task}: {count:,} trials")

print()


# =============================================================================
# CELL 5 — STEP 2: REMOVE TRIALS WITH FEWER THAN 3 FIXATIONS
#
# WHY:
#   Trials with 1 or 2 fixations cannot produce meaningful values for several
#   features — particularly fixation dispersion (requires minimum 2 clusters,
#   unstable with fewer than 3 fixations), transition entropy (no transitions
#   possible with 1 fixation), and scanpath length. These trials also likely
#   represent tracker loss or participant inattention.
#
# METHOD:
#   (a) Compute fixation count per trial
#   (b) Identify trials below threshold
#   (c) Remove ALL fixation rows belonging to those trials
#   The trial is the unit of exclusion — not the individual fixation.
#
# =============================================================================

print("=" * 65)
print(f"STEP 2: REMOVE TRIALS WITH FEWER THAN {MIN_FIXATIONS_PER_TRIAL} FIXATIONS")
print("=" * 65)

# Compute how many fixations each trial has
trial_fix_counts = (
    df_step1
    .groupby(['Subject', 'Trial', 'Task'])
    .size()
    .reset_index(name='fixation_count')
)

# Identify trials that fall below the minimum threshold
low_fix_trials = trial_fix_counts[
    trial_fix_counts['fixation_count'] < MIN_FIXATIONS_PER_TRIAL
]

print(f"Trials with fewer than {MIN_FIXATIONS_PER_TRIAL} fixations:")
for task in ['ITEM', 'RELATIONAL']:
    task_low = low_fix_trials[low_fix_trials['Task'] == task]
    print(f"  {task}: {len(task_low)} trials excluded "
          f"({100 * len(task_low) / step1_trial_counts[task]:.1f}% of {task} trials)")

print()

# Build a set of (Subject, Trial) pairs to exclude for fast lookup
# We use a set of tuples for O(1) membership testing
exclude_pairs = set(
    zip(low_fix_trials['Subject'], low_fix_trials['Trial'])
)

# Remove all fixation rows belonging to excluded trials
df_step2 = df_step1[
    ~df_step1.apply(
        lambda row: (row['Subject'], row['Trial']) in exclude_pairs,
        axis=1
    )
].copy()

# Verify removal
step2_trial_counts = (
    df_step2
    .drop_duplicates(['Subject', 'Trial', 'Task'])
    .groupby('Task')
    .size()
)

print(f"Fixation rows before: {len(df_step1):,}")
print(f"Fixation rows after:  {len(df_step2):,}")
print(f"Rows removed:         {len(df_step1) - len(df_step2):,}")
print()
print(f"Trial counts after fixation threshold filter:")
for task, count in step2_trial_counts.items():
    print(f"  {task}: {count:,} trials")

print()

# Show the fixation count distribution for the remaining trials
remaining_fix_counts = (
    df_step2
    .groupby(['Subject', 'Trial', 'Task'])
    .size()
)
print("Fixation count distribution (remaining trials):")
print(f"  Mean:   {remaining_fix_counts.mean():.1f}")
print(f"  Median: {remaining_fix_counts.median():.1f}")
print(f"  Min:    {remaining_fix_counts.min()}")
print(f"  Max:    {remaining_fix_counts.max()}")
print()


# =============================================================================
# CELL 6 — STEP 3: SUBJECT EXCLUSION
#
# WHY:
#   Subjects with very few trials produce unreliable feature estimates and
#   could skew LOSO cross-validation results. Any subject with fewer than
#   MIN_TRIALS_PER_SUBJECT valid trials (after the fixation filter above)
#   is excluded entirely.
#
# THRESHOLD:
#   65% of 36 maximum trials = 23.4, floored to 23 trials minimum.
#   Confirmed by Whitlock.
#
# METHOD:
#   Count valid trials per subject per task after Step 2.
#   A subject is excluded if EITHER task falls below the threshold,
#   since the classifier requires both tasks for that subject.
#
# =============================================================================

print("=" * 65)
print(f"STEP 3: SUBJECT EXCLUSION (minimum {MIN_TRIALS_PER_SUBJECT} trials per task)")
print("=" * 65)

# Count valid trials per subject per task after Step 2
subject_trial_counts = (
    df_step2
    .drop_duplicates(['Subject', 'Trial', 'Task'])
    .groupby(['Subject', 'Task'])
    .size()
    .unstack(fill_value=0)
)

# A subject is excluded if either task is below the threshold
excluded_subjects = subject_trial_counts[
    (subject_trial_counts.get('ITEM', 0)       < MIN_TRIALS_PER_SUBJECT) |
    (subject_trial_counts.get('RELATIONAL', 0) < MIN_TRIALS_PER_SUBJECT)
].index.tolist()

if len(excluded_subjects) == 0:
    print("No subjects excluded — all subjects meet the trial threshold.")
else:
    print(f"Subjects excluded ({len(excluded_subjects)} total):")
    for subj in excluded_subjects:
        item_count = subject_trial_counts.loc[subj, 'ITEM'] \
            if 'ITEM' in subject_trial_counts.columns else 0
        rel_count  = subject_trial_counts.loc[subj, 'RELATIONAL'] \
            if 'RELATIONAL' in subject_trial_counts.columns else 0
        print(f"  Subject {subj:>3}: "
              f"ITEM={item_count} trials, RELATIONAL={rel_count} trials "
              f"(threshold: >= {MIN_TRIALS_PER_SUBJECT})")

print()

# Remove excluded subjects from the dataset
df_step3 = df_step2[
    ~df_step2['Subject'].isin(excluded_subjects)
].copy()

subjects_before = df_step2['Subject'].nunique()
subjects_after  = df_step3['Subject'].nunique()

print(f"Subjects before: {subjects_before}")
print(f"Subjects after:  {subjects_after}")
print(f"Subjects removed: {subjects_before - subjects_after}")
print()


# =============================================================================
# CELL 7 — FINAL AUDIT
#
# Print a complete summary of the cleaned dataset. This is the exclusion
# report that goes directly into the DOCUMENTATION.md and the methods
# section of the paper.
# =============================================================================

print("=" * 65)
print("FINAL EXCLUSION REPORT")
print("=" * 65)

# Final trial counts
final_trial_counts = (
    df_step3
    .drop_duplicates(['Subject', 'Trial', 'Task'])
    .groupby('Task')
    .size()
)

# Final fixation counts per trial
final_fix_counts = (
    df_step3
    .groupby(['Subject', 'Trial', 'Task'])
    .size()
)

# Final trials per subject
final_trials_per_subject = (
    df_step3
    .drop_duplicates(['Subject', 'Trial', 'Task'])
    .groupby(['Subject', 'Task'])
    .size()
    .unstack()
)

print(f"{'METRIC':<45} {'BEFORE':>8} {'AFTER':>8} {'REMOVED':>8}")
print("-" * 73)
print(f"{'Total fixation rows':<45} "
      f"{len(df_raw):>8,} "
      f"{len(df_step3):>8,} "
      f"{len(df_raw) - len(df_step3):>8,}")
print(f"{'Unique subjects':<45} "
      f"{df_raw['Subject'].nunique():>8} "
      f"{df_step3['Subject'].nunique():>8} "
      f"{df_raw['Subject'].nunique() - df_step3['Subject'].nunique():>8}")

raw_item_trials = raw_trial_counts.get('ITEM', 0)
raw_rel_trials  = raw_trial_counts.get('RELATIONAL', 0)
final_item_trials = final_trial_counts.get('ITEM', 0)
final_rel_trials  = final_trial_counts.get('RELATIONAL', 0)

print(f"{'ITEM trials':<45} "
      f"{raw_item_trials:>8,} "
      f"{final_item_trials:>8,} "
      f"{raw_item_trials - final_item_trials:>8,}")
print(f"{'RELATIONAL trials':<45} "
      f"{raw_rel_trials:>8,} "
      f"{final_rel_trials:>8,} "
      f"{raw_rel_trials - final_rel_trials:>8,}")
print()

# Compute per-task fixation count stats cleanly
fix_counts_by_task = (
    df_step3
    .groupby(['Subject', 'Trial', 'Task'])
    .size()
    .reset_index(name='fixation_count')
)
item_fix = fix_counts_by_task[fix_counts_by_task['Task'] == 'ITEM']['fixation_count']
rel_fix  = fix_counts_by_task[fix_counts_by_task['Task'] == 'RELATIONAL']['fixation_count']

print("Final fixation counts per trial:")
print(f"  Mean   — ITEM:       {item_fix.mean():.1f}")
print(f"  Mean   — RELATIONAL: {rel_fix.mean():.1f}")
print(f"  Median — ITEM:       {item_fix.median():.1f}")
print(f"  Median — RELATIONAL: {rel_fix.median():.1f}")
print(f"  Min:                 {fix_counts_by_task['fixation_count'].min()}")
print(f"  Max:                 {fix_counts_by_task['fixation_count'].max()}")
print()

print("Final trials per subject (summary):")
print(f"  ITEM — mean: {final_trials_per_subject['ITEM'].mean():.1f}, "
      f"min: {final_trials_per_subject['ITEM'].min()}, "
      f"max: {final_trials_per_subject['ITEM'].max()}")
print(f"  RELATIONAL — mean: {final_trials_per_subject['RELATIONAL'].mean():.1f}, "
      f"min: {final_trials_per_subject['RELATIONAL'].min()}, "
      f"max: {final_trials_per_subject['RELATIONAL'].max()}")
print()

print("Exclusion summary:")
print(f"  Step 1 (target filter):        "
      f"{removed_trials_step1:,} trials removed "
      f"({100 * removed_trials_step1 / raw_trials:.1f}% of raw)")
print(f"  Step 2 (fixation threshold):   "
      f"{len(low_fix_trials):,} trials removed")
print(f"  Step 3 (subject exclusion):    "
      f"{subjects_before - subjects_after} subjects removed")
print()
print(f"Final dataset: {df_step3['Subject'].nunique()} subjects, "
      f"{final_item_trials + final_rel_trials:,} trials, "
      f"{len(df_step3):,} fixation rows")
print()


# =============================================================================
# CELL 8 — SAVE CLEANED DATA
#
# Save to encoding_data_clean.csv. This is the only file the feature
# extraction script will ever read. The raw file is untouched.
# =============================================================================

print("=" * 65)
print("SAVING CLEANED DATA")
print("=" * 65)

df_step3.to_csv(OUTPUT_PATH, index=False)

print(f"Saved: {OUTPUT_PATH}")
print(f"  Rows:    {len(df_step3):,}")
print(f"  Columns: {len(df_step3.columns)}")
print(f"  Columns: {list(df_step3.columns)}")
print()
print("Download encoding_data_clean.csv from the Colab Files panel.")
print("Send the exclusion report above back for documentation.")
print()
print("=" * 65)
print("PREPROCESSING COMPLETE — READY FOR FEATURE EXTRACTION")
print("=" * 65)
