# Retrieval Phase Documentation
## Eye-Tracking Memory Task Classifier — Whitlock Lab

**Phase:** Retrieval Phase Classifier (Step 6 onwards)
**Builds on:** Encoding Phase Classifier (see DOCUMENTATION.md)
**Status:** Active — Pre-build planning complete, awaiting Q1 answer before Step 6
**Target:** Nature publication

---

## Table of Contents

1. [Phase Overview](#1-phase-overview)
2. [Retrieval Data Description](#2-retrieval-data-description)
3. [Design Decisions Log](#3-design-decisions-log)
4. [Feature Set — Final Specification](#4-feature-set--final-specification)
5. [Preprocessing Pipeline — Specification](#5-preprocessing-pipeline--specification)
6. [Classifier Architecture](#6-classifier-architecture)
7. [Encoding vs Retrieval Comparison](#7-encoding-vs-retrieval-comparison)
8. [Scripts](#8-scripts)
9. [Open Questions](#9-open-questions)
10. [Meeting Log](#10-meeting-log)

---

## 1. Phase Overview

The retrieval phase classifier is a separate binary classifier that predicts
whether a retrieval trial came from the **Item Memory task** or the
**Relational Memory task**, using eye movements recorded during the 6-second
test display.

This is the natural complement to the encoding classifier. Where the encoding
classifier asks "can we detect encoding strategy from how people look during
learning?", the retrieval classifier asks "can we detect retrieval strategy
from how people look during test?"

**The retrieval display is structurally different from encoding:**
- Encoding: one centrally-located object on a background scene
- Retrieval: three objects simultaneously on screen (one target + two foils)
  arranged in Top_Left, Top_Right, and Bottom_Middle positions

This means the feature set is different. AOI transitions between objects are
now necessary to complete the task, and foil behavior becomes relevant in a
way it could not be at encoding.

**Key scientific distinction between tasks at retrieval:**
- Item task foils are novel/unstudied objects — familiarity alone can guide
  correct selection without comparing objects against the scene
- Relational task foils are studied objects paired with different scenes —
  participants must actively discriminate the associate from familiar foils
  using the scene as a retrieval cue

This predicts more scene-directed fixations, more comparative scanning
between objects, and stronger target-orienting in the Relational task.

---

## 2. Retrieval Data Description

### File: `Item_Relational_Retrieval_Data.csv`

**Raw dimensions:** 96,731 rows × 27 columns  
**Subjects:** 84 (subjects 39 and 61 absent — confirmed no data exists)  
**Subject 2:** Present in retrieval data — will be excluded for consistency  
with encoding exclusion  

### Column Reference

| Column | Type | Description |
|---|---|---|
| `Subject` | Integer | Participant ID |
| `TargetObject` | String | Name of the target object in the display |
| `Trial` | Integer | Encoding trial number this test trial corresponds to |
| `x`, `y` | Float | Fixation centroid coordinates (pixels) |
| `Start` | Integer | Fixation onset time (ms) — see timing caveat below |
| `End` | Integer | Fixation offset time (ms) |
| `Duration` | Integer | Fixation duration (ms) |
| `Confidence` | Integer | Participant confidence rating (low/medium/high) |
| `TaskOrder` | String | Task order for this participant |
| `KeyPressed` | Integer | Participant's selection (1=object1, 2=object2, 3=object3) |
| `ResponseTime` | Integer | Response time from display onset (ms) |
| `Task` | String | `ITEM` or `RELATIONAL` |
| `object1/2/3` | String | Names of objects in Top_Left, Top_Right, Bottom_Middle positions |
| `Scene` | String | Name of the background scene |
| `CorrectTest` | Integer | Position of correct item (1=Top_Left, 2=Top_Right, 3=Bottom_Middle) |
| `Accuracy` | Integer | 1 = correct response, 0 = incorrect |
| `Top_Left` | Boolean | Fixation falls within top-left ROI |
| `Top_Right` | Boolean | Fixation falls within top-right ROI |
| `Bottom_Middle` | Boolean | Fixation falls within bottom-middle ROI |
| `Screen` | Boolean | Fixation is on the display |
| `LookROI` | String | Which object ROI is fixated (Top_Left/Top_Right/Bottom_Middle/NaN) |
| `objectviewed` | String | Name of the object being fixated |
| `TargetLocation` | String | **CODING ERROR — do not use. Use CorrectTest instead.** |
| `LookItem` | String | `Target` or `Foil` — which type of object is fixated |

### Critical Data Notes

**TargetLocation coding error (confirmed by Whitlock):**
The `TargetLocation` column contains a coding error — `Top_Right` is absent
and all trials where `CorrectTest = 2` are incorrectly labeled. Use
`CorrectTest` for all target identification. Whitlock confirmed this.

**Scene fixations encoding:**
There is no explicit `scene` value in `LookROI`. Scene/background fixations
are identified as: `LookROI = NaN AND Screen = True`.
Off-screen fixations are: `LookROI = NaN AND Screen = False` — these are
excluded from all features.

**Timing caveat:**
Although the column descriptor labels `Start`/`End` as "relative to onset
of test display," observed values begin at approximately 3001ms, consistent
with the pre-display sequence (0.5s fixation cross + 2s scene presentation
+ 0.5s fixation cross = 3000ms offset). All retrieval timing features are
computed after subtracting 3000ms to express fixation times relative to the
onset of the three-object test display.

### Retrieval Data Audit Results

| Metric | Value |
|---|---|
| Total fixation rows | 96,731 |
| Total subjects (raw) | 84 |
| ITEM trials (mean per subject) | 34.7 |
| RELATIONAL trials (mean per subject) | 34.9 |
| Fixations per trial — mean | 16.6 |
| Fixations per trial — min | 1 |
| Fixations per trial — max | 46 |
| Test display window | ~3001ms to ~9470ms (Start values) |
| Scene fixations total | 24,940 |
| Scene fixations — ITEM | 10,089 |
| Scene fixations — RELATIONAL | 14,851 |
| Off-screen fixations | 894 |
| Accuracy — ITEM | 90.7% (2641/2912) |
| Accuracy — RELATIONAL | 82.0% (2401/2929) |
| CorrectTest distribution | 1=1925, 2=1944, 3=1972 (balanced) |

Scene fixation counts by task confirm theoretical predictions — Relational
participants look at the scene more during test because it is an informative
retrieval cue in that task.

---

## 3. Design Decisions Log

### Decision R1 — Retrieval classifier is separate from encoding

**Source:** Whitlock (email, June 2026)

The retrieval phase classifier is a completely separate analysis from the
encoding classifier. The two are run independently and their AUCs are
compared. The encoding result (AUC = 0.861) is finalized and unaffected
by retrieval phase decisions.

**Rationale:** Encoding features capture how participants form a
representation. Retrieval features capture how they use it. These are
different scientific questions requiring different feature sets.

---

### Decision R2 — Feature set (15 features, confirmed by Whitlock)

**Source:** Whitlock (document, June 2026) + joint planning

All four of Whitlock's specified measures are included within a richer
15-feature set. See Section 4 for full specification.

**Key clarification from Whitlock:** "Rate of transitioning between faces"
in his document means inter-object transitions (not faces). Confirmed.

---

### Decision R3 — Target identification uses CorrectTest, not TargetLocation

**Source:** Confirmed by Whitlock (June 2026) after coding error flagged

`TargetLocation` has a coding error — Top_Right values are missing.
`CorrectTest` is correct and is used for all target identification.

---

### Decision R4 — Scene fixation definition

**Source:** Data audit (June 2026)

Scene/background fixations = `LookROI = NaN AND Screen = True`
Off-screen fixations = `LookROI = NaN AND Screen = False` → excluded from all
features and trial fixation counts.

---

### Decision R5 — Timing offset

**Source:** Data audit + Whitlock confirmation (June 2026)

All timing features computed after subtracting 3000ms from `Start`/`End`
values. This expresses fixation times relative to the onset of the
three-object test display rather than trial onset.

Formula: `display_time_ms = Start - 3000`

Documentation note for paper methods:
"Although the column descriptor labels Start/End as relative to test display
onset, observed values begin at approximately 3000ms, consistent with the
pre-display sequence. Retrieval timing features were therefore computed after
subtracting 3000ms."

---

### Decision R6 — Subject exclusions

**Source:** Whitlock (June 2026)

- Exclude **Subject 2** — consistent with encoding exclusion
- Exclude subjects with fewer than **23 valid trials per task** (65% of 36
  maximum, same threshold as encoding)
- Applied after trial-level exclusions (Decision R7)

---

### Decision R7 — Trial and fixation exclusions

**Source:** Whitlock (June 2026)

Applied in this order:
1. Exclude off-screen fixations: `Screen = False`
2. Identify valid on-screen fixations: object fixations + scene fixations
3. Exclude trials with fewer than **[N] valid on-screen fixations**

**Note:** Minimum fixation threshold for retrieval is pending Whitlock's
response. Encoding used ≥3 fixations. Given mean fixations per retrieval
trial is 16.6 (vs 10.1 at encoding), the same threshold of 3 is likely
appropriate. Will be confirmed and updated.

---

### Decision R8 — Positive class

**Source:** Consistent with encoding classifier

Item task = 1 (positive class), Relational task = 0.

---

### Decision R9 — Validation strategy

**Source:** Consistent with encoding classifier

Leave-One-Subject-Out (LOSO) cross-validation. Train on N-1 subjects,
test on 1, repeat for all subjects. Same fold-safe preprocessing
(imputation and scaling fit on training data only inside each fold).

---

### Decision R10 — Statistical testing

**Source:** Consistent with encoding classifier

- **Bootstrap CI:** Subject-level cluster bootstrap (2000 iterations)
- **Permutation test:** Within-subject label shuffling, 200 permutations,
  100 trees for null models, plus-one correction
- **Both AUCs reported:** Pooled and mean per-subject, to eliminate
  researcher degrees of freedom

---

### Decision R11 — Encoding vs retrieval comparison

**Source:** Joint planning (June 2026)

The encoding vs retrieval AUC comparison is an **optional sensitivity
analysis** run after both classifiers are individually complete.

Plan:
1. Identify common subject set passing both encoding and retrieval
   inclusion rules
2. Rerun encoding classifier on that common subset
3. Compare encoding AUC vs retrieval AUC on the same participants

This does not change or invalidate the main encoding result (83 subjects,
AUC = 0.861). It provides a cleaner apples-to-apples comparison for the
paper's discussion.

---

## 4. Feature Set — Final Specification

**Total features: 15**
All confirmed by Whitlock. Features marked ✅ are Whitlock's four
specified measures.

### Target/Foil Dwell (4)

| # | Feature | Definition |
|---|---|---|
| 1 | `target_dwell_prop` | Total Duration on target / total on-screen Duration |
| 2 | `foil_dwell_prop_mean` | Mean Duration on foils / total on-screen Duration |
| 3 | `target_fix_count` | Number of fixations on target (LookItem = Target) |
| 4 | `foil_fix_count_mean` | Mean fixations on each foil |

### First Fixation (2)

| # | Feature | Definition |
|---|---|---|
| 5 | `first_fix_is_target` | ✅ Binary: 1 if first valid on-screen fixation lands on target, else 0 |
| 6 | `time_to_first_target_ms` | ✅ Display-relative ms to first fixation on target; NaN if target never fixated |

### Transitions (3)

| # | Feature | Definition |
|---|---|---|
| 7 | `inter_object_transition_rate` | ✅ Transitions between distinct object AOIs / total valid object fixations |
| 8 | `target_foil_transition_count` | Switches between target and either foil |
| 9 | `scene_object_transition_count` | Switches between scene background and any object |

### Revisits (2)

| # | Feature | Definition |
|---|---|---|
| 10 | `target_revisits` | ✅ Returns to target AOI after leaving |
| 11 | `foil_revisits_mean` | Mean returns to each foil after leaving |

### Scene (2)

| # | Feature | Definition |
|---|---|---|
| 12 | `scene_dwell_prop` | Duration on scene / total on-screen Duration |
| 13 | `scene_fix_count` | Number of fixations on scene background |

### Spatial (2)

| # | Feature | Definition |
|---|---|---|
| 14 | `scanpath_length_deg` | Sum of inter-fixation distances (degrees of visual angle) |
| 15 | `saccade_amplitude_mean_deg` | Mean inter-fixation distance (degrees of visual angle) |

### NaN Handling

| Feature | NaN condition | Treatment |
|---|---|---|
| `time_to_first_target_ms` | Target never fixated in trial | NaN → median impute inside LOSO fold |
| `first_fix_is_target` | No valid on-screen fixations | NaN → median impute inside LOSO fold |
| All others | Should not produce NaN after exclusions | Flag if found |

---

## 5. Preprocessing Pipeline — Specification

Steps applied in this exact order. Every exclusion logged with counts.

### Step 1 — Load raw CSV
Load `Item_Relational_Retrieval_Data.csv`. Print basic audit.

### Step 2 — Exclude Subject 2
Remove all rows where `Subject == 2`. Consistent with encoding exclusion.

### Step 3 — Exclude off-screen fixations
Remove rows where `Screen == False`. These are fixations outside the display
and contribute no meaningful AOI information.

### Step 4 — Remove trials below minimum fixation threshold
Count valid on-screen fixations per trial. Remove trials below threshold.
**Threshold: Pending Whitlock confirmation (expected: ≥3 fixations).**

### Step 5 — Subject exclusion (65% threshold)
Exclude subjects with fewer than 23 valid trials in either task.

### Step 6 — Save cleaned data
Save to `retrieval_data_clean.csv`. Print full exclusion report.

---

## 6. Classifier Architecture

Identical to encoding classifier. Key parameters:

| Parameter | Value |
|---|---|
| Primary model | Random Forest (500 trees) |
| Baseline model | Logistic Regression |
| Validation | LOSO cross-validation |
| Positive class | Item = 1, Relational = 0 |
| Decision threshold | 0.5 |
| Preprocessing | Median imputation + StandardScaler, fit inside each fold |
| Bootstrap CI | Subject-level cluster bootstrap, 2000 iterations |
| Permutation test | 200 within-subject shuffles, 100 trees for null, plus-one correction |
| SHAP | TreeExplainer on all-subject model |
| transition_entropy | N/A — not a retrieval feature |

---

## 7. Encoding vs Retrieval Comparison

**Status:** Deferred — run after both classifiers are individually complete.

**Plan:**
1. Identify subjects present in both the encoding clean dataset (83 subjects)
   and the retrieval clean dataset (N TBD)
2. Rerun encoding LOSO on the common subject subset only
3. Compare:
   - Encoding AUC (common subset) vs Retrieval AUC (common subset)
4. Report in paper discussion as a sensitivity analysis

**Primary results remain:**
- Encoding: 83 subjects, AUC = 0.861 [0.839–0.881], p = .005
- Retrieval: TBD

---

## 8. Scripts

| Script | Purpose | Status |
|---|---|---|
| `step6_retrieval_preprocessing.py` | Load, clean, apply exclusions | ⏳ Not started |
| `step7_retrieval_feature_extraction.py` | Compute 15 features | ⏳ Not started |
| `step8_retrieval_sanity_check.py` | Verify feature directions | ⏳ Not started |
| `step9_retrieval_classifier.py` | LOSO, AUC, bootstrap CI, SHAP | ⏳ Not started |
| `step9b_retrieval_subject_diagnostic.py` | Per-subject AUC check | ⏳ Not started |
| `step9c_retrieval_permutation_test.py` | Permutation test | ⏳ Not started |
| `step10_retrieval_figures.py` | Publication figures | ⏳ Not started |

---

## 9. Open Questions

| # | Question | Asked | Status |
|---|---|---|---|
| R1 | Minimum fixation threshold for retrieval (≥3 expected) | Yes | ⏳ Awaiting Whitlock |

---

## 10. Meeting Log

### Whitlock document — June 2026 (retrieval measures)
Whitlock provided a document specifying four retrieval-specific eye movement
measures to include in the classifier. "Rate of transitioning between faces"
confirmed to mean inter-object transitions (not faces — typo). All four
measures incorporated within a 15-feature retrieval set.

### Planning session — June 2026
Full retrieval feature set locked (15 features). Exclusion rules confirmed.
Timing offset documented. CorrectTest confirmed as the correct column for
target identification. Encoding vs retrieval comparison deferred to after
both classifiers complete. One open question remaining: minimum fixation
threshold for retrieval (Q for Whitlock).
