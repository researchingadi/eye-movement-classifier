# =============================================================================
# STEP 2 — FEATURE EXTRACTION
# Eye-Tracking Memory Task Classifier — Whitlock Lab
#
# PURPOSE:
#   Compute all 20 trial-level eye movement features from the cleaned fixation
#   data. Output is a single feature matrix — one row per trial, one column
#   per feature — which is the direct input to the classifier.
#
# INPUT:
#   encoding_data_clean.csv — output of Step 1 preprocessing
#
# OUTPUT:
#   feature_matrix_encoding.csv — trial-level feature matrix
#   feature_matrix_encoding_halves.csv — same features with halves temporal split
#   Printed feature summary report
#
# THE 20 FEATURES (all confirmed by Whitlock):
#
#   AOI DWELL (4):
#     1.  Object dwell proportion
#     2.  Scene dwell proportion
#     3.  Object fixation count
#     4.  Scene fixation count
#
#   TEMPORAL DWELL — THIRDS (6) [primary]:
#     5.  Early object dwell    (0–1333ms)
#     6.  Middle object dwell   (1334–2666ms)
#     7.  Late object dwell     (2667–4000ms)
#     8.  Early scene dwell     (0–1333ms)
#     9.  Middle scene dwell    (1334–2666ms)
#     10. Late scene dwell      (2667–4000ms)
#
#   FIXATION DURATION (2):
#     11. Mean fixation duration
#     12. First fixation latency to object
#
#   TRANSITIONS & SEQUENCE (4):
#     13. Object-scene transition count
#     14. Transition entropy (Shannon H, cross-AOI only)
#     15. Object revisit count
#     16. Scene revisit count
#
#   SPATIAL (3):
#     17. Scanpath length (pixels)
#     18. Fixation dispersion (Ramey et al. 2020 k-means method)
#     19. Saccade amplitude mean (pixels)
#
#   NOTE ON NaN VALUES:
#     Some features are undefined for certain trials by design:
#     - Trials with zero scene fixations: scene latency = NaN
#     - Trials with zero object fixations: object latency = NaN
#     - Trials with zero transitions: entropy = NaN
#     These are handled gracefully and imputed during classifier training.
#
# HOW TO RUN IN GOOGLE COLAB:
#   1. Upload encoding_data_clean.csv (output of Step 1)
#   2. Run Cell 1 (imports and config)
#   3. Run Cell 2 (helper functions — must run before Cell 3)
#   4. Run Cell 3 (main extraction loop)
#   5. Run Cell 4 (save and summarize)
#   6. Download both output CSVs from the Files panel
#
# =============================================================================


# =============================================================================
# CELL 1 — IMPORTS AND CONFIGURATION
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

# Suppress KMeans convergence warnings on small trials
warnings.filterwarnings('ignore', category=UserWarning)

print("=" * 65)
print("STEP 2 — FEATURE EXTRACTION")
print("Eye-Tracking Memory Task Classifier — Whitlock Lab")
print("=" * 65)
print()

# ── File paths ────────────────────────────────────────────────────────────────
INPUT_PATH          = "encoding_data_clean.csv"
OUTPUT_PATH_THIRDS  = "feature_matrix_encoding.csv"         # primary output
OUTPUT_PATH_HALVES  = "feature_matrix_encoding_halves.csv"  # secondary exploratory

# ── Encoding window boundaries (milliseconds) ─────────────────────────────────
# Temporal split into THIRDS (primary, confirmed by Whitlock)
EARLY_END   = 1333    # 0    to 1333ms  = early third
MIDDLE_END  = 2666    # 1334 to 2666ms  = middle third
# Late = 2667 to 4000ms (everything after MIDDLE_END)

# Temporal split into HALVES (secondary exploratory)
HALF_POINT  = 2000    # 0 to 2000ms = first half, 2001+ = second half

# ── Pixels-to-degrees conversion ─────────────────────────────────────────────
# Screen: 23.8 inch diagonal, 1920x1080 resolution
# Viewing distance: 783mm (midpoint of 765mm top / 800mm bottom)
# Pixels per mm: 3.642 (confirmed calculation)
# Formula: degrees = 2 * arctan(pixels / (2 * viewing_distance_mm * px_per_mm))
VIEWING_DISTANCE_MM = 783.0
PIXELS_PER_MM       = 3.642

# ── Classifier label ──────────────────────────────────────────────────────────
# Item task = 1 (positive class), Relational task = 0 (confirmed by Whitlock)
TASK_LABEL = {'ITEM': 1, 'RELATIONAL': 0}

print("Configuration:")
print(f"  Input:              {INPUT_PATH}")
print(f"  Output (thirds):    {OUTPUT_PATH_THIRDS}")
print(f"  Output (halves):    {OUTPUT_PATH_HALVES}")
print(f"  Temporal thirds:    0–{EARLY_END}ms | {EARLY_END+1}–{MIDDLE_END}ms | {MIDDLE_END+1}–4000ms")
print(f"  Temporal halves:    0–{HALF_POINT}ms | {HALF_POINT+1}–4000ms")
print(f"  Viewing distance:   {VIEWING_DISTANCE_MM}mm")
print(f"  Pixels per mm:      {PIXELS_PER_MM}")
print()

# Load the clean data
df = pd.read_csv(INPUT_PATH)

print(f"Loaded: {len(df):,} fixation rows, "
      f"{df['Subject'].nunique()} subjects, "
      f"{df.groupby(['Subject','Trial','Task']).ngroups:,} trials")
print()


# =============================================================================
# CELL 2 — FEATURE HELPER FUNCTIONS
#
# Each function takes a single trial's fixation data (as a DataFrame slice)
# and returns one feature value. Functions are documented with:
#   - what they compute
#   - what they return when undefined (NaN cases)
#   - the theoretical motivation
# =============================================================================

# ── Conversion utility ────────────────────────────────────────────────────────

def pixels_to_degrees(pixels):
    """
    Convert a distance in pixels to degrees of visual angle.

    Uses the formula: degrees = 2 * arctan(pixels / (2 * d * px_per_mm))
    where d is the viewing distance in mm.

    Parameters
    ----------
    pixels : float — distance in pixels

    Returns
    -------
    float — distance in degrees of visual angle
    """
    return np.degrees(
        2 * np.arctan(pixels / (2 * VIEWING_DISTANCE_MM * PIXELS_PER_MM))
    )


# ── AOI dwell features ────────────────────────────────────────────────────────

def compute_dwell_proportions(trial):
    """
    Compute dwell time proportions for object and scene AOIs.

    Object dwell proportion = total Duration on object / total Duration
    Scene dwell proportion  = total Duration on scene  / total Duration

    Returns NaN for both if total duration is zero (degenerate trial).

    Theoretical motivation: Item task → more object dwell,
    Relational task → more distributed dwell across both AOIs.
    """
    total    = trial['Duration'].sum()
    obj_dwell = trial.loc[trial['StudiedItem'] == True,  'Duration'].sum()
    sc_dwell  = trial.loc[trial['StudiedItem'] == False, 'Duration'].sum()

    if total == 0:
        return np.nan, np.nan

    return obj_dwell / total, sc_dwell / total


def compute_fixation_counts(trial):
    """
    Count the number of fixations on object and scene AOIs.

    Returns (object_count, scene_count) as integers.
    """
    obj_count = (trial['StudiedItem'] == True).sum()
    sc_count  = (trial['StudiedItem'] == False).sum()
    return int(obj_count), int(sc_count)


# ── Temporal dwell features ───────────────────────────────────────────────────

def compute_temporal_dwell(trial, aoi_value, t_start, t_end):
    """
    Compute total dwell time on a given AOI within a time window.

    A fixation is included if its ONSET (Start) falls within the window.
    This is the standard approach — we classify fixations by when they begin.

    Parameters
    ----------
    trial     : DataFrame — single trial fixation data
    aoi_value : bool — True for object, False for scene
    t_start   : int — window start in ms (inclusive)
    t_end     : int — window end in ms (inclusive)

    Returns
    -------
    float — total Duration (ms) of fixations in the window on the AOI
    """
    mask = (
        (trial['StudiedItem'] == aoi_value) &
        (trial['Start'] >= t_start) &
        (trial['Start'] <= t_end)
    )
    return float(trial.loc[mask, 'Duration'].sum())


# ── Fixation duration features ────────────────────────────────────────────────

def compute_mean_fixation_duration(trial):
    """
    Mean duration of all fixations in the trial (ms).

    Theoretical motivation: longer fixations indicate deeper featural
    processing. Item task may produce longer mean fixations on the object.
    """
    if len(trial) == 0:
        return np.nan
    return float(trial['Duration'].mean())


def compute_first_fixation_latency(trial, aoi_value):
    """
    Time (ms) from trial onset to the first fixation on a given AOI.

    Uses the Start time of the first fixation where StudiedItem matches
    aoi_value. Returns NaN if no fixation on that AOI exists.

    Parameters
    ----------
    aoi_value : bool — True for object latency, False for scene latency
    """
    hits = trial.loc[trial['StudiedItem'] == aoi_value, 'Start']
    if len(hits) == 0:
        return np.nan
    return float(hits.iloc[0])


# ── Transition and sequence features ─────────────────────────────────────────

def compute_transitions_and_entropy(trial):
    """
    Compute object-scene transition count and transition entropy.

    A TRANSITION is defined as a change in AOI between consecutive fixations.
    Only CROSS-AOI transitions are counted (object→scene and scene→object).
    Same-AOI transitions (object→object, scene→scene) are excluded.
    Confirmed by Whitlock.

    TRANSITION COUNT: total number of cross-AOI switches (both directions).

    TRANSITION ENTROPY: Shannon H = -sum(p * log2(p)) computed over the
    distribution of the two cross-AOI transition types:
        p1 = proportion of transitions that are object→scene
        p2 = proportion of transitions that are scene→object

    H = 1.0 when transitions are equally split (max diversity)
    H = 0.0 when all transitions go in one direction only
    H = NaN when there are zero transitions (undefined)

    Theoretical motivation: Relational encoding should produce more
    transitions AND higher entropy (more balanced back-and-forth).
    Item encoding should produce fewer transitions with lower entropy.

    Returns
    -------
    (transition_count, entropy) — both float, entropy is NaN if no transitions
    """
    # Build ordered sequence of AOI labels per fixation
    aoi_sequence = trial['StudiedItem'].tolist()

    # Count each transition type
    obj_to_scene = 0
    scene_to_obj = 0

    for i in range(len(aoi_sequence) - 1):
        current = aoi_sequence[i]
        next_   = aoi_sequence[i + 1]

        if current == True and next_ == False:
            obj_to_scene += 1
        elif current == False and next_ == True:
            scene_to_obj += 1
        # Same-AOI transitions are intentionally ignored

    total_transitions = obj_to_scene + scene_to_obj

    # Entropy is undefined if there are no transitions
    if total_transitions == 0:
        return 0, np.nan

    # Shannon entropy over the two cross-AOI transition types
    p1 = obj_to_scene / total_transitions
    p2 = scene_to_obj / total_transitions

    # Avoid log(0) — if one proportion is zero, that term contributes 0
    entropy = 0.0
    if p1 > 0:
        entropy -= p1 * np.log2(p1)
    if p2 > 0:
        entropy -= p2 * np.log2(p2)

    return float(total_transitions), float(entropy)


def compute_revisits(trial):
    """
    Count the number of revisits to each AOI.

    A REVISIT is defined as any return to an AOI after having left it.
    Example: object → scene → object = 1 object revisit.
    Confirmed definition by Whitlock.

    Method: scan the AOI sequence and count how many times gaze
    returns to an AOI it was previously in, after an intervening
    fixation on the other AOI.

    Returns
    -------
    (object_revisits, scene_revisits) — both int
    """
    aoi_sequence = trial['StudiedItem'].tolist()

    # Compress consecutive same-AOI fixations into single visits
    # e.g., [T, T, F, T, T, T, F] → [T, F, T, F]
    # This way each "visit" is one entry, not multiple fixations
    visits = []
    for aoi in aoi_sequence:
        if len(visits) == 0 or aoi != visits[-1]:
            visits.append(aoi)

    # Count revisits: occurrences of each AOI after the first visit
    obj_revisits  = 0
    scene_revisits = 0
    seen_obj   = False
    seen_scene = False

    for visit in visits:
        if visit == True:
            if seen_obj:
                obj_revisits += 1
            seen_obj = True
        else:
            if seen_scene:
                scene_revisits += 1
            seen_scene = True

    return int(obj_revisits), int(scene_revisits)


# ── Spatial features ──────────────────────────────────────────────────────────

def compute_scanpath_length(trial):
    """
    Total Euclidean distance traveled across all fixations (pixels).

    Computed as the sum of distances between consecutive fixation centroids
    (x, y). Converted to degrees of visual angle.

    Theoretical motivation: Relational encoding requires moving eyes
    between object and scene, producing longer total scanpaths.
    """
    xs = trial['x'].values
    ys = trial['y'].values

    if len(xs) < 2:
        return np.nan

    # Distance between each consecutive pair of fixations
    diffs      = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    total_px   = float(diffs.sum())
    total_deg  = pixels_to_degrees(total_px)

    return total_deg


def compute_saccade_amplitude(trial):
    """
    Mean saccade amplitude across the trial (degrees of visual angle).

    Approximated as the mean Euclidean distance between consecutive
    fixation centroids. This is the standard fixation-report approximation
    used when saccade events are not explicitly available (Ramey et al. 2020).

    Theoretical motivation: Relational encoding requires larger saccades
    to move between object and scene regions.
    """
    xs = trial['x'].values
    ys = trial['y'].values

    if len(xs) < 2:
        return np.nan

    distances_px  = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    mean_px       = float(distances_px.mean())
    mean_deg      = pixels_to_degrees(mean_px)

    return mean_deg


def compute_fixation_dispersion(trial):
    """
    Fixation dispersion score using the Ramey, Henderson & Yonelinas (2020)
    k-means + silhouette composite method.

    PROCEDURE:
      1. Extract (x, y) fixation coordinates for the trial
      2. Test k = 2, 3, ..., min(n_fixations-1, 8) cluster solutions
         using k-means clustering
      3. Select the optimal k using the silhouette score
         (higher silhouette = better cluster separation)
      4. Minimum k is 2 (per Ramey et al.)
      5. Dispersion score = optimal_k × mean distance between cluster centroids

    Higher score = fixations spread across more and more distant regions.

    EDGE CASES:
      - Less than 3 unique fixation positions: return NaN
        (cannot meaningfully cluster with k >= 2)
      - All fixations at the same location: return 0.0

    Returns
    -------
    float — dispersion score, or NaN if undefined
    """
    coords = trial[['x', 'y']].values

    # Need at least 3 fixations to test k=2 with silhouette
    if len(coords) < 3:
        return np.nan

    # Check for degenerate case — all fixations at same location
    if np.all(coords == coords[0]):
        return 0.0

    # Maximum clusters to test — cap at min(n_fixations-1, 8) for efficiency
    max_k = min(len(coords) - 1, 8)

    # Minimum is 2 clusters (per Ramey et al. 2020)
    if max_k < 2:
        return np.nan

    # Test each k and record silhouette score
    best_k         = 2
    best_silhouette = -1.0

    for k in range(2, max_k + 1):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(coords)

            # Silhouette score requires at least 2 distinct labels
            # (can fail if all points are assigned to one cluster)
            if len(np.unique(labels)) < 2:
                continue

            score = silhouette_score(coords, labels)
            if score > best_silhouette:
                best_silhouette = score
                best_k          = k

        except Exception:
            # Skip degenerate k-means solutions
            continue

    # Fit final k-means with optimal k
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    kmeans_final.fit(coords)
    centroids = kmeans_final.cluster_centers_

    # Mean pairwise distance between cluster centroids
    n_centroids    = len(centroids)
    centroid_dists = []
    for i in range(n_centroids):
        for j in range(i + 1, n_centroids):
            d = np.sqrt(
                (centroids[i, 0] - centroids[j, 0])**2 +
                (centroids[i, 1] - centroids[j, 1])**2
            )
            centroid_dists.append(d)

    mean_centroid_dist = np.mean(centroid_dists) if centroid_dists else 0.0

    # Dispersion = number of clusters × mean distance between centroids
    dispersion = best_k * mean_centroid_dist

    return float(dispersion)


# =============================================================================
# CELL 3 — MAIN FEATURE EXTRACTION LOOP
#
# Iterate over every trial in the clean dataset.
# For each trial, call all feature functions and store the result.
# Prints progress every 500 trials.
# =============================================================================

print("=" * 65)
print("EXTRACTING FEATURES")
print("=" * 65)
print("This may take a few minutes — dispersion requires k-means per trial.")
print()

# Storage for results
records = []

# Get all unique trials
trial_groups = df.groupby(['Subject', 'Trial', 'Task'])
n_trials     = len(trial_groups)

for i, ((subject, trial_id, task), trial_df) in enumerate(trial_groups):

    # Sort fixations by onset time — critical for sequence-based features
    trial_df = trial_df.sort_values('Start').reset_index(drop=True)

    # ── Progress reporting ────────────────────────────────────────────────────
    if (i + 1) % 500 == 0 or i == 0:
        print(f"  Processing trial {i+1:,} of {n_trials:,} "
              f"({100*(i+1)/n_trials:.1f}%)...")

    # ── Metadata ──────────────────────────────────────────────────────────────
    row = {
        'subject_id':  subject,
        'trial_id':    trial_id,
        'task':        task,
        'task_label':  TASK_LABEL[task],   # 1 = ITEM, 0 = RELATIONAL
    }

    # ── FEATURE 1–2: AOI dwell proportions ───────────────────────────────────
    obj_prop, sc_prop = compute_dwell_proportions(trial_df)
    row['obj_dwell_prop']   = obj_prop
    row['scene_dwell_prop'] = sc_prop

    # ── FEATURE 3–4: AOI fixation counts ─────────────────────────────────────
    obj_count, sc_count = compute_fixation_counts(trial_df)
    row['obj_fix_count']   = obj_count
    row['scene_fix_count'] = sc_count

    # ── FEATURES 5–10: Temporal dwell (THIRDS — primary) ─────────────────────
    # Object dwell in each third of the encoding window
    row['obj_dwell_early_ms']   = compute_temporal_dwell(
        trial_df, True, 0, EARLY_END)
    row['obj_dwell_middle_ms']  = compute_temporal_dwell(
        trial_df, True, EARLY_END + 1, MIDDLE_END)
    row['obj_dwell_late_ms']    = compute_temporal_dwell(
        trial_df, True, MIDDLE_END + 1, 4000)

    # Scene dwell in each third of the encoding window
    row['scene_dwell_early_ms']  = compute_temporal_dwell(
        trial_df, False, 0, EARLY_END)
    row['scene_dwell_middle_ms'] = compute_temporal_dwell(
        trial_df, False, EARLY_END + 1, MIDDLE_END)
    row['scene_dwell_late_ms']   = compute_temporal_dwell(
        trial_df, False, MIDDLE_END + 1, 4000)

    # ── FEATURES 5–8 (HALVES — secondary exploratory) ────────────────────────
    # These are stored separately but computed in the same loop for efficiency
    row['obj_dwell_first_half_ms']    = compute_temporal_dwell(
        trial_df, True, 0, HALF_POINT)
    row['obj_dwell_second_half_ms']   = compute_temporal_dwell(
        trial_df, True, HALF_POINT + 1, 4000)
    row['scene_dwell_first_half_ms']  = compute_temporal_dwell(
        trial_df, False, 0, HALF_POINT)
    row['scene_dwell_second_half_ms'] = compute_temporal_dwell(
        trial_df, False, HALF_POINT + 1, 4000)

    # ── FEATURE 11: Mean fixation duration ───────────────────────────────────
    row['mean_fix_duration_ms'] = compute_mean_fixation_duration(trial_df)

    # ── FEATURE 12: First fixation latency to object ─────────────────────────
    row['first_fix_latency_obj_ms'] = compute_first_fixation_latency(
        trial_df, True)

    # ── FEATURES 13–14: Transitions and entropy ──────────────────────────────
    transitions, entropy = compute_transitions_and_entropy(trial_df)
    row['obj_scene_transitions'] = transitions
    row['transition_entropy']    = entropy

    # ── FEATURES 15–16: Revisits ──────────────────────────────────────────────
    obj_revisits, scene_revisits = compute_revisits(trial_df)
    row['obj_revisits']   = obj_revisits
    row['scene_revisits'] = scene_revisits

    # ── FEATURE 17: Scanpath length (degrees) ────────────────────────────────
    row['scanpath_length_deg'] = compute_scanpath_length(trial_df)

    # ── FEATURE 18: Fixation dispersion (Ramey et al. 2020) ──────────────────
    row['fixation_dispersion'] = compute_fixation_dispersion(trial_df)

    # ── FEATURE 19: Mean saccade amplitude (degrees) ─────────────────────────
    row['saccade_amplitude_mean_deg'] = compute_saccade_amplitude(trial_df)

    records.append(row)

print()
print(f"Feature extraction complete: {len(records):,} trials processed.")
print()


# =============================================================================
# CELL 4 — BUILD FEATURE MATRICES, VALIDATE, AND SAVE
# =============================================================================

print("=" * 65)
print("BUILDING AND VALIDATING FEATURE MATRICES")
print("=" * 65)

# ── Build the full dataframe ──────────────────────────────────────────────────
df_features = pd.DataFrame(records)

# ── Define the two feature matrices ──────────────────────────────────────────

# Metadata columns — always included, never used as classifier features
METADATA_COLS = ['subject_id', 'trial_id', 'task', 'task_label']

# Primary feature set — temporal split into THIRDS
PRIMARY_FEATURE_COLS = [
    # AOI dwell
    'obj_dwell_prop', 'scene_dwell_prop',
    'obj_fix_count',  'scene_fix_count',
    # Temporal dwell — thirds
    'obj_dwell_early_ms',   'obj_dwell_middle_ms',   'obj_dwell_late_ms',
    'scene_dwell_early_ms', 'scene_dwell_middle_ms', 'scene_dwell_late_ms',
    # Duration
    'mean_fix_duration_ms', 'first_fix_latency_obj_ms',
    # Transitions
    'obj_scene_transitions', 'transition_entropy',
    'obj_revisits',          'scene_revisits',
    # Spatial
    'scanpath_length_deg', 'fixation_dispersion', 'saccade_amplitude_mean_deg',
]

# Secondary feature set — same but temporal split into HALVES
HALVES_FEATURE_COLS = [
    # AOI dwell (same)
    'obj_dwell_prop', 'scene_dwell_prop',
    'obj_fix_count',  'scene_fix_count',
    # Temporal dwell — halves
    'obj_dwell_first_half_ms',    'obj_dwell_second_half_ms',
    'scene_dwell_first_half_ms',  'scene_dwell_second_half_ms',
    # Duration (same)
    'mean_fix_duration_ms', 'first_fix_latency_obj_ms',
    # Transitions (same)
    'obj_scene_transitions', 'transition_entropy',
    'obj_revisits',          'scene_revisits',
    # Spatial (same)
    'scanpath_length_deg', 'fixation_dispersion', 'saccade_amplitude_mean_deg',
]

# Build the two output dataframes
df_thirds = df_features[METADATA_COLS + PRIMARY_FEATURE_COLS].copy()
df_halves = df_features[METADATA_COLS + HALVES_FEATURE_COLS].copy()

# ── Validation ────────────────────────────────────────────────────────────────
print("Feature matrix shape (thirds): ", df_thirds.shape)
print("Feature matrix shape (halves): ", df_halves.shape)
print()
print(f"Subjects:         {df_thirds['subject_id'].nunique()}")
print(f"ITEM trials:      {(df_thirds['task_label'] == 1).sum():,}")
print(f"RELATIONAL trials:{(df_thirds['task_label'] == 0).sum():,}")
print()

# NaN report — how many trials have missing values per feature?
print("NaN counts per feature (expected for some features by design):")
nan_counts = df_thirds[PRIMARY_FEATURE_COLS].isnull().sum()
nan_features = nan_counts[nan_counts > 0]
if len(nan_features) == 0:
    print("  No NaN values found.")
else:
    for feat, count in nan_features.items():
        pct = 100 * count / len(df_thirds)
        print(f"  {feat:<35} {count:>5} trials ({pct:.1f}%)")
print()

# Quick sanity check — do features differ between tasks in expected directions?
print("Feature means by task (sanity check — should match theory):")
print(f"  {'Feature':<35} {'ITEM':>10} {'RELATIONAL':>12} {'Direction':>12}")
print("  " + "-" * 72)

expected_direction = {
    'obj_dwell_prop':              'ITEM > REL',
    'scene_dwell_prop':            'REL > ITEM',
    'obj_fix_count':               'ITEM > REL',
    'scene_fix_count':             'REL > ITEM',
    'obj_scene_transitions':       'REL > ITEM',
    'transition_entropy':          'REL > ITEM',
    'scanpath_length_deg':         'REL > ITEM',
    'fixation_dispersion':         'REL > ITEM',
    'saccade_amplitude_mean_deg':  'REL > ITEM',
}

item_means = df_thirds[df_thirds['task_label']==1][PRIMARY_FEATURE_COLS].mean()
rel_means  = df_thirds[df_thirds['task_label']==0][PRIMARY_FEATURE_COLS].mean()

for feat, direction in expected_direction.items():
    item_val = item_means[feat]
    rel_val  = rel_means[feat]

    # Check if actual direction matches expected
    if 'ITEM > REL' in direction:
        matches = '✓' if item_val > rel_val else '✗ UNEXPECTED'
    else:
        matches = '✓' if rel_val > item_val else '✗ UNEXPECTED'

    print(f"  {feat:<35} {item_val:>10.3f} {rel_val:>12.3f} "
          f"  {direction} {matches}")

print()

# ── Save both feature matrices ────────────────────────────────────────────────
df_thirds.to_csv(OUTPUT_PATH_THIRDS, index=False)
df_halves.to_csv(OUTPUT_PATH_HALVES, index=False)

print("=" * 65)
print("SAVED")
print("=" * 65)
print(f"Primary (thirds): {OUTPUT_PATH_THIRDS}")
print(f"  {len(df_thirds):,} trials × {len(PRIMARY_FEATURE_COLS)} features")
print()
print(f"Secondary (halves): {OUTPUT_PATH_HALVES}")
print(f"  {len(df_halves):,} trials × {len(HALVES_FEATURE_COLS)} features")
print()
print("Download both files from the Colab Files panel.")
print("Send the full output above back for documentation.")
print()
print("=" * 65)
print("FEATURE EXTRACTION COMPLETE — READY FOR SANITY CHECK")
print("=" * 65)
