# Project Documentation
## Eye-Tracking Memory Task Classifier

**Last Updated:** 9 June 2026  
**Status:** Active — Pre-build planning phase complete, awaiting final decisions before feature extraction  
**Target:** Nature publication  
**Collaborators:** Prof. Jonathan Whitlock (PI, cognitive psychology) | [Adi Singh] (ML & analysis)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Scientific Background](#2-scientific-background)
3. [Experiment Design](#3-experiment-design)
4. [Data Description](#4-data-description)
5. [Planning Phase — Questions & Decisions Log](#5-planning-phase--questions--decisions-log)
6. [Feature Set — Final Specification](#6-feature-set--final-specification)
7. [Preprocessing Pipeline — Final Specification](#7-preprocessing-pipeline--final-specification)
8. [Classifier Architecture — Final Specification](#8-classifier-architecture--final-specification)
9. [Outputs & Figures](#9-outputs--figures)
10. [Scripts Built So Far](#10-scripts-built-so-far)
11. [Open Questions](#11-open-questions)
12. [Meeting Log](#12-meeting-log)
13. [Key References](#13-key-references)

---

## 1. Project Overview

This project builds a **binary trial-level classifier** that predicts whether an encoding trial came from an **Item Memory task** or a **Relational Memory task**, using eye movement features extracted from fixation data recorded during encoding.

The core scientific question is whether the way a person moves their eyes during learning reflects the *kind* of memory representation they are forming — specifically, whether they are encoding an object's individual features in isolation (item memory) versus encoding how that object relates to its background scene (relational memory). If a classifier trained on eye movement features can reliably distinguish the two tasks, it provides direct evidence that these two fundamentally different encoding strategies produce detectably different gaze signatures.

The classifier will be evaluated using Leave-One-Subject-Out (LOSO) cross-validation, tested for statistical significance via permutation test, explained using SHAP feature importance, and reported with bootstrapped confidence intervals. The goal is a publication-quality, fully reproducible analysis pipeline targeting Nature.

---

## 2. Scientific Background

### The Core Distinction

Memory researchers distinguish between two fundamental types of encoding:

**Item memory** involves encoding the features of an individual object so it can be later identified or recognized on its own. The task focuses attention on the object itself — what it looks like, what its properties are.

**Relational memory** involves encoding the relationship between two things — in this case, how a specific object fits with a specific background scene. This requires processing both elements and integrating them, which demands that attention be distributed across the display rather than focused on a single region.

These two encoding strategies place fundamentally different demands on visual attention, and those differences should be detectable in eye movement patterns.

### Why Eye Movements Are the Right Measure

Eye movements are the primary mechanism by which humans sample visual information. Where you look determines what you encode. This means the gaze record is not just a *correlate* of memory encoding — it is, in a meaningful sense, the mechanism through which encoding happens. Prior work (Ryan et al., 2000; Hannula et al., 2010; Baym et al., 2014) has established that eye movement patterns during encoding are sensitive to the type of memory being formed, and that object-scene transitions in particular are a direct index of relational binding.

The Ramey, Henderson & Yonelinas (2020) paper — which Whitlock pointed us to for the fixation dispersion measure — provides strong evidence that the *spatial distribution* of fixations during encoding predicts subsequent memory strength, with more dispersed viewing associated with stronger familiarity. This work motivates several of our features, particularly fixation dispersion and scanpath length.

### Why a Classifier?

Traditional analyses in cognitive neuroscience test individual measures one at a time. A classifier approach offers three advantages for this project:

1. It asks whether the full *pattern* of eye movement features, taken together, can distinguish the tasks — a more powerful and ecologically valid question than testing features individually.
2. It produces a continuous probability score per trial, which maps naturally onto signal detection measures (AUC, hits, false alarms) that Whitlock specifically wanted.
3. SHAP explainability lets us identify *which specific features* are driving the classification, closing the loop back to the psychological theory.

---

## 3. Experiment Design

### Overview

Approximately 84 participants completed two encoding tasks while eye movements were recorded using an **EyeLink 1000** (SR Research, 1000 Hz). The order of the two tasks was counterbalanced across participants. Eye-tracker calibration was performed before each block.

### Item Task

**Encoding:** Participants were shown 36 scene-object pairs. The *same scene* was used on every trial — it served as a constant backdrop. A unique object was superimposed on that scene for 4 seconds per trial. Participants were instructed to judge whether the object would fit inside a shoebox (this orienting task encourages featural processing of the object without explicitly requiring memorization).

**Test (3-AFC):** After encoding, participants saw a test display with three objects superimposed on the scene — one studied object and two novel foils. They selected the studied object via button press and rated their confidence.

**Key structural note:** Because the scene repeats on every Item trial, the scene provides zero discriminative information at test. Participants must rely on the object's features alone. This means looking at the scene during encoding carries no strategic value for the Item task.

### Relational Task

**Encoding:** Participants completed three sequential study-test blocks. Each encoding block had 36 trials with *unique scene-object pairs* on every trial (108 encoding trials total). Participants were instructed to judge how well the object fit with its background scene — an orienting task that encourages relational processing.

**Test (3-AFC):** Each test display showed three *studied* objects on one studied scene. One object (the **associate**) had been paired with that scene during encoding; the other two had been paired with *different* scenes. Participants identified the associate.

**Key structural note:** Because all three objects in the test display were studied, the scene is the only cue that can guide the correct choice. Participants who successfully encoded the object-scene relationship can use the scene at test to identify the associate. This makes scene-looking during encoding strategically valuable in the Relational task.

### Why This Contrast Is Powerful

The design creates a theoretically clean contrast:

| | Item Task | Relational Task |
|---|---|---|
| Scene at encoding | Uninformative (same on every trial) | Informative (unique each trial) |
| Scene at test | Uninformative | Critical retrieval cue |
| Encoding instruction | Object features (shoebox judgment) | Object-scene relationship (fit judgment) |
| Foils at test | Unstudied (novel objects) | Studied (objects from other pairs) |

This means that scene-directed fixations during encoding and scene-to-object transitions are theoretically expected to be higher in the Relational task — and these are precisely the features most likely to drive classifier performance.

---

## 4. Data Description

### File: `Item_Relational_Encoding_Data.csv`

This is the encoding phase fixation dataset. Each row is one fixation.

| Column | Type | Description |
|---|---|---|
| `Subject` | Integer | Participant ID |
| `Trial` | Integer | Trial number within session |
| `x` | Float | X-coordinate of fixation centroid (pixels) |
| `y` | Float | Y-coordinate of fixation centroid (pixels) |
| `Start` | Integer | Fixation onset time (ms), relative to encoding display onset |
| `End` | Integer | Fixation offset time (ms) |
| `Duration` | Integer | Fixation duration (ms) |
| `TaskOrder` | String | Which task came first for this participant |
| `Task` | String | `ITEM` or `RELATIONAL` |
| `object` | String | Name of the object on this trial |
| `scene` | String | Name of the scene on this trial |
| `target` | Integer | `1` = this object is the correct answer at test; `0` = foil |
| `StudiedItem` | Boolean | `TRUE` = fixation lands on the object AOI; `FALSE` = fixation lands on the scene AOI |

### Key Data Facts

- **Total rows:** 121,282 fixations
- **Subjects:** 84 (subject numbers 39 and 61 are absent — under investigation)
- **Encoding window:** 0–4000ms per trial (4 seconds)
- **Screen resolution:** 1920 × 1080 pixels
- **Viewing distance:** To be confirmed by Whitlock
- **AOIs:** Pre-defined and encoded in the `StudiedItem` column — no coordinate-based AOI mapping required
- **Phase coverage:** This file contains encoding phase only. Test phase data is a separate file (arriving separately).

### Trial Counts (Before Any Exclusions)

| Task | Trials per subject | Total trials (84 subjects) |
|---|---|---|
| ITEM (all) | 36 | 3,024 |
| RELATIONAL (all) | 108 | 9,072 |
| ITEM (target==1 only) | 36 | 3,024 |
| RELATIONAL (target==1 only) | ~36 | ~3,024 |

**Note on class balance:** The raw data has 3× more Relational trials because of the task structure (see Section 3). We resolve this by filtering to `target==1` trials only (see Section 7).

### AOI Structure

During encoding, the display contains one scene and one object. The `StudiedItem` column is the AOI flag:

- `StudiedItem = TRUE` → fixation is on the **object**
- `StudiedItem = FALSE` → fixation is on the **scene**

This was pre-defined by the lab and requires no further coordinate processing on our end.


### File: `Item_Relational_Retrieval_Data.csv`

Test phase fixation dataset. One row per fixation during the test display window.

| Column | Type | Description |
|---|---|---|
| `Subject` | Integer | Participant ID |
| `TargetObject` | String | Name of the target object in the display |
| `Trial` | Integer | Encoding trial number this test trial corresponds to |
| `x`, `y` | Float | Fixation centroid coordinates (pixels) |
| `Start`, `End`, `Duration` | Integer | Fixation timing (ms), relative to test display onset |
| `Confidence` | Integer | Participant confidence rating (low/medium/high) |
| `TaskOrder` | String | Task order for this participant |
| `KeyPressed` | Integer | Participant's selection (1=object1, 2=object2, 3=object3) |
| `ResponseTime` | Integer | Response time from display onset (ms) |
| `Task` | String | `ITEM` or `RELATIONAL` |
| `object1/2/3` | String | Names of objects in Top_Left, Top_Right, Bottom_Middle positions |
| `Scene` | String | Name of the background scene |
| `CorrectTest` | Integer | Position of correct item (1=Top_Left, 2=Top_Right, 3=Bottom_Middle) |
| `Accuracy` | Integer | 1 = correct response, 0 = incorrect |
| `Top_Left`, `Top_Right`, `Bottom_Middle` | Boolean | Whether fixation falls in each object ROI |
| `Screen` | Boolean | Whether fixation is on the display at all |
| `LookROI` | String | Which object ROI is fixated (`Top_Left`, `Top_Right`, `Bottom_Middle`, or NaN) |
| `objectviewed` | String | Name of the object being fixated |
| `TargetLocation` | String | Position of target — **NOTE: coding error, use `CorrectTest` instead** |
| `LookItem` | String | `Target` or `Foil` — which item is being fixated |

#### Key Facts — Retrieval Data
- **Total rows:** 96,731 fixations
- **Subjects:** 84 (same as encoding)
- **Test window:** ~3000ms to ~9000ms (6-second display, offset by pre-display period)
- **Mean fixations per trial:** 16.6 (SD = 3.8)
- **Scene fixations:** Encoded as `LookROI = NaN` AND `Screen = True` — 26.7% of all fixations
- **Accuracy:** Item = 90.7%, Relational = 82.0%

#### Critical Data Note — TargetLocation Column
The `TargetLocation` column contains a coding error: `Top_Right` is absent, and all trials where `CorrectTest = 2` (Top_Right) are incorrectly labeled as `Bottom_Middle`. The `CorrectTest` column is correct and will be used in all analyses. Flagged to Whitlock.

#### Scene Fixations at Test
There is no explicit `scene` value in `LookROI`. Scene/background fixations are identified as: `LookROI = NaN` AND `Screen = True`. This combination means the fixation lands on the display but outside all three object ROIs — i.e., on the background scene. This is the operationalization used for all scene-related test phase features.

Scene fixation counts by task confirm theoretical predictions:
- Item task: 10,089 scene fixations
- Relational task: 14,851 scene fixations

Relational participants look at the scene more during test because the scene is an informative retrieval cue in that task.

### Display Parameters & Visual Angle Conversion

| Parameter | Value |
|---|---|
| Monitor size | 23.8 inches diagonal |
| Resolution | 1920 × 1080 pixels |
| Physical dimensions | 527.3mm × 296.7mm |
| Viewing distance | 783mm (midpoint: 765mm top, 800mm bottom) |
| Pixels per mm | 3.642 |
| Pixels per degree | 99.4 |
| Full screen width | ~19.3° |
| Full screen height | ~10.9° |

Conversion formula: `degrees = 2 × arctan(pixels / (2 × 783 × 3.642))`

---

## 5. Planning Phase — Questions & Decisions Log

Before writing a single line of feature extraction or classifier code, we spent several weeks clarifying every ambiguous design decision with Whitlock. The principle was: it is better to answer 20 questions upfront than to fix errors halfway through a Nature-level pipeline.

Every decision below was either confirmed by Whitlock or resolved through referenced literature.

---

### Decision 1 — What data to use

**Question:** Is this encoding-only, or are we combining encoding and test phase features?

**Answer (Whitlock):** Both phases, but as **separate classifiers**. Encoding and test phase will be analyzed independently and AUCs compared. This is scientifically motivated: encoding features capture how the participant is *forming* a representation; test features capture how they are *using* it. These are different questions.

**Implication:** We build two separate feature extraction pipelines and two separate classifiers. The encoding phase classifier comes first (current stage). Test phase data will arrive separately.

---

### Decision 2 — Handling the class imbalance

**Question:** There are 3× more Relational trials than Item trials (108 vs 36 per subject). How do we handle this?

**Answer (Whitlock):** Filter to **associate/target trials only** (`target == 1`). In the Relational task, only 1 of every 3 encoding trials corresponds to the object that will be the correct answer (associate) at test. The other 2 are foil encodings. By keeping only the associate trial, we:

1. Eliminate the class imbalance (36 Item vs ~36 Relational per subject)
2. Make the trials theoretically comparable — every trial in the analysis is one where the encoded object goes on to be the correct answer at test
3. Make the classifier question more precise: we are asking whether the encoding strategy for the *to-be-remembered* item differs between tasks

**Implementation:** `df[df['target'] == 1]` applied before any feature extraction.

---

### Decision 3 — Positive class

**Question:** Which task is the positive class in the binary classifier?

**Answer (Whitlock):** **Item task = 1 (positive class), Relational task = 0.**

---

### Decision 4 — Validation strategy

**Question:** How do we validate the classifier to ensure generalizability?

**Answer (Whitlock):** **Leave-One-Subject-Out (LOSO) cross-validation.** Train on 83 subjects, test on 1, repeat 84 times. Every subject serves as the held-out test set exactly once. This is the most conservative approach — it directly tests whether the classifier generalizes to completely new, unseen participants.

**Our recommendation (accepted by Whitlock):** Also report 10-fold stratified cross-validation by participant as a secondary check for computational comparison.

---

### Decision 5 — Statistical testing

**Question:** How do we demonstrate the AUC is above chance?

**Answer:** Two-part approach, confirmed by Whitlock:

1. **Permutation test** — shuffle task labels, rerun LOSO 1000 times, build null AUC distribution, compute p-value. This is the primary significance test.
2. **Bootstrap confidence intervals** — resample held-out predictions 2000 times, report 95% CI.

**Final reported format:** `AUC = 0.XX [95% CI: 0.XX–0.XX], permutation p < 0.001`

**Note:** Permutation test is computationally expensive (1000 × 84 model fits = 84,000 fits). This is the last step of the project, not the first.

---

### Decision 6 — Which AUC to report

**Question:** LOSO produces two AUC estimates — the overall pooled AUC (all held-out predictions combined) and the mean of the 84 per-subject AUCs. Which is primary?

**Answer (Whitlock):** **Report both.** Whitlock specifically cited researcher degrees of freedom as the reason — reporting only the more favorable metric would be selective. Reporting both is transparent and reviewers will respect it.

**Technical note:** These two values can differ, especially with class imbalance or heterogeneous subjects. The pooled AUC weights subjects by their trial count; the mean per-subject AUC weights each subject equally.

---

### Decision 7 — Explainability

**Question:** How do we show which features drive the classification?

**Answer:** **SHAP (SHapley Additive exPlanations)** using TreeExplainer, trained on a model fit to all 84 subjects. This was suggested by the student collaborator and accepted by Whitlock. Whitlock was sent the foundational SHAP papers (Lundberg & Lee 2017; Lundberg et al. 2020) for background reading.

SHAP values are computed on a model trained on all subjects (not the LOSO folds) because this maximizes training data and produces the most stable feature importance estimates. The slight theoretical impurity (model has seen all data) is acceptable here because SHAP is used for *interpretation*, not for the performance claim. The performance claim comes from LOSO.

---

### Decision 8 — Confusion matrix threshold

**Question:** With probabilistic outputs, the confusion matrix requires a decision threshold. What do we use?

**Answer:** **0.5.** With balanced classes (~36 Item vs ~36 Relational after the target==1 filter), 0.5 is appropriate and standard. Confirmed by Whitlock after explanation.

---

### Decision 9 — Subject exclusion

**Question:** Some subjects have fewer trials than expected. What is the exclusion threshold?

**Answer (Whitlock):** **65% of total possible trials.** With associate-only Relational trials, this means subjects with fewer than ~23 trials are excluded. If this removes too many subjects, the threshold can be softened — but the justification must be reported in the manuscript.

---

### Decision 10 — Minimum fixation threshold

**Question:** Some trials have only 1 fixation. Features like transition entropy and dispersion are undefined or meaningless for these. What is the exclusion threshold?

**Answer (Whitlock, two-step approach):**

**Step 1 — Remove all 1-fixation trials unconditionally.** These are definite exclusions regardless of task.

**Step 2 — Data-driven threshold from distribution.** After removing 1-fixation trials, plot the distribution of fixation counts and determine a principled cutoff. We built and sent Whitlock a fixation distribution plot (v2, with 1-fixation trials already excluded) to support this decision. 

Final decision (Whitlock, 5 June 2026): Remove trials with 1 or 2 fixations. Minimum threshold = 3 fixations. Trials with fewer than 3 fixations are excluded entirely.

Additionally: apply a **±3 SD filter on fixation duration** to remove individual outlier fixations (extremely short or extremely long). This is applied at the fixation level, not the trial level — the trial stays, only the bad fixation is dropped. *Awaiting confirmation from Whitlock on whether the SD is computed globally or per-subject.*

---

### Decision 11 — Transition entropy definition

**Question:** Transition entropy uses Shannon H = -Σ p·log₂(p). With two AOIs, there are four transition types. Which are included?

**Answer (Whitlock):** **Cross-AOI transitions only:** object→scene and scene→object. Same-AOI transitions (object→object, scene→scene) are excluded.

**Consequence:** With only two transition types, entropy is a measure of how *evenly split* the gaze transitions are between the two directions. Maximum entropy (H = 1.0 bit) means 50% object→scene and 50% scene→object. Minimum entropy (H = 0) means all transitions go in one direction only.

**Theoretical expectation:** Relational encoding should produce higher entropy because participants actively integrate the two regions, producing roughly equal proportions of both transition types. Item encoding should produce lower entropy because gaze predominantly stays on the object with occasional scene glances.

---

### Decision 12 — Fixation dispersion definition

**Question:** What exact implementation of fixation dispersion do we use?

**Answer:** Referenced to **Ramey, Henderson & Yonelinas (2020)**, page 6. The Ramey et al. measure is a three-step composite:

1. Submit all fixation (x, y) coordinates for a trial to a **k-means clustering algorithm**
2. Use a **silhouette algorithm** to identify the optimal number of clusters (minimum = 2)
3. **Dispersion score = number of clusters × mean Euclidean distance between cluster centroids**

Higher score = fixations spread across more distant regions of the display.

**Important implementation note:** The silhouette method requires a minimum of 2 clusters, which means trials with very few fixations produce unstable clustering. This is one reason the minimum fixation threshold matters — it directly affects the reliability of this feature.

---

### Decision 13 — Early/late temporal split

**Question:** The feature list includes early and late dwell time. What is the cutoff, and how many windows?

**Answer (Whitlock):** Split the 4000ms encoding window into **thirds**: 0–1333ms (early), 1334–2666ms (middle), 2667–4000ms (late). This gives 6 temporal dwell features instead of 4 (early/middle/late × object/scene).

**Additionally:** Whitlock wants us to also compute the **halves split** (0–2000ms, 2001–4000ms) as a secondary exploratory analysis. Both versions will be computed; the thirds version is primary.

**Scientific motivation for thirds:** The early window captures initial orienting behavior (where does attention go first?), the middle window captures sustained processing, and the late window captures consolidation or review. Thirds give a richer picture of how attention unfolds across the encoding episode.

---

### Decision 14 — Saccade amplitude

**Question:** Is saccade amplitude a 17th feature beyond the 16 in the feature document?

**Answer (Whitlock):** Yes, confirmed as feature #17. Whitlock acknowledged it was accidentally omitted from the document.

**Implementation:** Computed as the Euclidean distance in pixels between consecutive fixation centroids (x, y). This is a fixation-report-derived approximation of true saccade amplitude. Screen resolution is 1920×1080. Viewing distance is pending — will convert to degrees of visual angle once confirmed.

---

### Decision 15 — Revisit definition

**Question:** What counts as one "revisit" to an AOI?

**Answer:** Any return to an AOI after leaving counts as one revisit. So the sequence object→scene→object = 1 object revisit. The definition is straightforward and sequence-based.

### Decision 16 — ±3 SD duration filter removed from pipeline

**Background:** Earlier planning included a ±3 SD fixation duration filter to remove outlier fixations, based on a preprocessing approach Whitlock had used previously.

**Clarification (Whitlock, June 2026):** The ±3 SD approach was used in a different project where only the first fixation per trial was analyzed. It does not apply to this project where we are summarizing fixation behavior across the entire trial. The duration filter is removed from the preprocessing pipeline entirely.

**Updated pipeline:** Step 5 (duration outlier removal) is eliminated. Preprocessing goes directly from minimum fixation threshold to subject exclusions.

### Decision 17 — Subjects 39 and 61 confirmed absent

**Question:** Why are subject numbers 39 and 61 missing from the dataset?

**Answer (Whitlock, June 2026):** No data exists for these two subjects.
The dataset is definitively 84 subjects. No further investigation needed.

---

### Decision 18 — TargetLocation coding error confirmed

**Answer (Whitlock, June 2026):** Confirmed. CorrectTest column is correct
and will be used in all retrieval phase analyses. Whitlock is correcting
the original code. No impact on encoding phase pipeline.

---

### Decision 19 — Viewing distance and pixels-to-degrees conversion

**Final values confirmed:**
- Viewing distance: 783mm (midpoint of 765mm top / 800mm bottom)
- Monitor: 23.8 inches diagonal, 1920×1080 resolution
- Physical dimensions: 527.3mm × 296.7mm
- Pixels per mm: 3.642 (horizontal), 3.641 (vertical)
- Conversion: 1 degree of visual angle = 99.4 pixels
- Formula: degrees = 2 × arctan(pixels / (2 × 783 × 3.642))

All saccade amplitude values computed in pixels and converted to
degrees using this formula for reporting.

---

## 6. Feature Set — Final Specification

**Total features: 20** (17 primary + 3 from the temporal split expansion)

All features are computed at the **trial level**. The input to the classifier is one row per trial with 20 feature columns plus metadata.

### AOI-Based Dwell Features (8)

| # | Feature | Definition |
|---|---|---|
| 1 | Object dwell time proportion | Sum of Duration where StudiedItem=True / total Duration in trial |
| 2 | Scene dwell time proportion | Sum of Duration where StudiedItem=False / total Duration in trial |
| 3 | Object fixation count | Count of rows where StudiedItem=True |
| 4 | Scene fixation count | Count of rows where StudiedItem=False |
| 5 | Early object dwell time | Sum of Duration on object where Start < 1333ms |
| 6 | Middle object dwell time | Sum of Duration on object where 1333ms ≤ Start < 2667ms |
| 7 | Late object dwell time | Sum of Duration on object where Start ≥ 2667ms |
| 8 | Early scene dwell time | Sum of Duration on scene where Start < 1333ms |
| 9 | Middle scene dwell time | Sum of Duration on scene where 1333ms ≤ Start < 2667ms |
| 10 | Late scene dwell time | Sum of Duration on scene where Start ≥ 2667ms |

### Fixation Duration Features (2)

| # | Feature | Definition |
|---|---|---|
| 11 | Mean fixation duration | Mean of all Duration values in the trial |
| 12 | First fixation latency to object | Start time (ms) of the first fixation where StudiedItem=True |

### Sequence & Transition Features (4)

| # | Feature | Definition |
|---|---|---|
| 13 | Object→scene transition count | Number of cross-AOI switches (both directions) between object and scene |
| 14 | Transition entropy | Shannon H computed over {object→scene, scene→object} transition proportions |
| 15 | Number of object revisits | Number of times gaze returns to object after leaving (any return = 1 revisit) |
| 16 | Number of scene revisits | Number of times gaze returns to scene after leaving |

### Spatial Features (3)

| # | Feature | Definition |
|---|---|---|
| 17 | Scanpath length | Sum of Euclidean distances between consecutive fixation centroids (pixels) |
| 18 | Fixation dispersion | k-means + silhouette composite score: N_clusters × mean_centroid_distance (Ramey et al. 2020) |
| 19 | Saccade amplitude (mean) | Mean Euclidean distance between consecutive fixation centroids (pixels; approximate saccade amplitude) |

### Secondary Exploratory Feature Set

A parallel feature matrix will be computed using halves split (0–2000ms / 2001–4000ms) for features 5–10, per Whitlock's request for exploratory comparison.

---

## 7. Preprocessing Pipeline — Final Specification

The preprocessing pipeline runs in strict sequential order. Every step is logged and documented.

### Step 1 — Load raw CSV

Load `Item_Relational_Encoding_Data.csv`. Verify row count, subject count, column names. Log basic descriptive statistics.

### Step 2 — Filter to target trials only

Apply `df[df['target'] == 1]`. This resolves the 3:1 class imbalance and restricts analysis to theoretically comparable trials across tasks. Log the trial counts before and after.

### Step 3 — Remove 1-fixation trials

Any trial with exactly 1 fixation is excluded unconditionally. Confirmed by Whitlock. Log how many trials are removed per task.

### Step 4 — Data-driven fixation count threshold (PENDING)

After sending Whitlock the v2 fixation distribution plot (1-fixation trials already removed), we are awaiting his decision on the minimum fixation count threshold. Once received, this step will apply that threshold and log exclusions.

### Step 5 — Duration outlier removal (±3 SD)

Remove individual fixations with durations more than ±3 SD from the mean. This is a fixation-level exclusion — the trial is kept, the outlier fixation is dropped.

**Pending decision:** Whether the mean/SD is computed globally (across all fixations in the dataset) or per-subject (each subject's fixations judged against their own distribution). Per-subject is standard in the literature. *Awaiting Whitlock confirmation.*

### Step 6 — Subject exclusion

Apply the 65% trial threshold. Any subject with fewer than ~23 valid trials (after steps 2–5) is excluded from the analysis. Log which subjects are excluded and why.

### Step 7 — Feature extraction

For each remaining trial, compute all 20 features. Output: one row per trial with columns for Subject, Trial, Task, task_label (1=Item, 0=Relational), and all 20 features.

### Step 8 — Save feature matrix

Save to `data/processed/feature_matrix_encoding.csv`. This file is the input to the classifier. It is version-controlled and never overwritten — any changes regenerate it with a new version tag.

### Preprocessing Results (from step1_preprocessing.py run)

| Step | Action | Trials Before | Trials After | Removed |
|---|---|---|---|---|
| Raw data | Loaded | 11,983 | — | — |
| Step 1 | Target filter (target==1) | 11,983 | 6,002 | 5,981 (49.9%) |
| Step 2 | Fixation threshold (≥3) | 6,002 | 5,664 | 338 |
| Step 3 | Subject exclusion (≥23 trials) | 5,664 | 5,609 | 55 (Subject 2 excluded) |

**Final cleaned dataset:**
- Subjects: 83
- ITEM trials: 2,760
- RELATIONAL trials: 2,849
- Total fixation rows: 56,561
- Mean fixations per trial: ITEM = 9.0, RELATIONAL = 11.1
- Min fixations per trial: 3, Max: 28

---

## 8. Classifier Architecture — Final Specification

### Model

**Primary:** Random Forest (scikit-learn `RandomForestClassifier`)
- 500 trees
- `class_weight='balanced'` as a safety measure even with balanced classes
- `min_samples_leaf=5` to prevent overfitting on small per-subject trial counts
- Random seed fixed at 42 for reproducibility

**Baseline comparison:** Logistic Regression — simpler, interpretable, serves as a sanity check

### Validation

**Leave-One-Subject-Out (LOSO)**
- Train on 83 subjects, test on 1
- Repeat 84 times (once per subject)
- Pool all held-out predictions across folds for overall AUC
- Also compute per-subject AUC for mean ± SD reporting

**Pipeline within each fold:**
1. Impute missing feature values (median imputation — some features may be NaN for edge-case trials)
2. StandardScaler normalization
3. Fit Random Forest
4. Predict probabilities on held-out subject
5. Store true labels and predicted probabilities

### Evaluation Metrics

| Metric | Method |
|---|---|
| Overall AUC | Computed from pooled held-out predictions across all 84 folds |
| Mean per-subject AUC | Mean ± SD of the 84 individual fold AUCs |
| 95% CI on AUC | Bootstrap resampling of pooled predictions (2000 iterations) |
| p-value | Permutation test: shuffle labels, rerun LOSO 1000 times, compare null distribution to observed AUC |
| Confusion matrix | Hard labels at 0.5 threshold from pooled predictions |

### SHAP Explainability

- Train a final Random Forest on all 84 subjects (no held-out set)
- Apply `shap.TreeExplainer` to compute SHAP values for every trial
- Report global mean |SHAP| bar chart and beeswarm plot
- This identifies which features are driving classification and whether they align with the theoretical predictions from Section 2

**Per-subject AUC reporting decision (confirmed):**
AUC is computed and reported for all 83 subjects individually.
The overall pooled AUC (computed from all held-out predictions
combined) is the primary metric. Mean per-subject AUC ± SD is
reported as supporting evidence. The minimum trial count of 23
trials per subject is noted in the methods to contextualise the
reliability of individual AUC estimates. No subjects are excluded
from per-subject AUC reporting.

**Updated evaluation metrics (confirmed):**

| Metric | Method |
|---|---|
| Overall AUC | Pooled held-out predictions across all 83 LOSO folds |
| Mean per-subject AUC | Mean ± SD of 83 individual fold AUCs |
| 95% CI on AUC | Bootstrap resampling of pooled predictions (2000 iterations) |
| p-value | Permutation test (deferred to final step) |
| Confusion matrix | Hard labels at 0.5 threshold |
| Accuracy | (TP + TN) / total trials |
| Sensitivity | TP / (TP + FN) — Item trials correctly identified |
| Specificity | TN / (TN + FP) — Relational trials correctly identified |

**Note on p-values from Step 3 sanity check:**
Trial-level t-tests in Step 3 are for internal pre-classifier validation
only and will not be reported as results in the paper. Because trials
are nested within subjects, trial-level tests inflate degrees of freedom
and produce overly optimistic p-values. The classifier-based validation
(LOSO AUC + permutation test) is the appropriate statistical framework
for the paper.

**Saved predictions file:**
After LOSO, a complete trial-level predictions CSV is saved containing
subject_id, trial_id, true_label, predicted_prob and predicted_class
for both Random Forest and Logistic Regression. This file is the single
source of truth for all downstream figures and statistics.

---

## 9. Outputs & Figures

Three publication-quality figures, all at 300 DPI:

**Figure 1 — ROC Curve**
Two panels: left panel shows the ROC curve with AUC and 95% CI in the legend; right panel shows the permutation null distribution with the observed AUC marked. Demonstrates both the magnitude and the statistical significance of classification performance.

**Figure 2 — Confusion Matrix**
2×2 matrix showing Item vs Relational, with raw counts and proportions in each cell. Threshold at 0.5.

**Figure 3 — SHAP Feature Importance**
Two panels: left shows mean |SHAP| bar chart (global feature importance ranking); right shows beeswarm plot (direction and distribution of each feature's contribution). This is the figure that tells the psychological story — which gaze behaviors most distinguish the two encoding strategies.

---

## 10. Scripts Built So Far

### `fixation_distribution_analysis.py` (v1)

**Purpose:** Exploratory data audit. Plots the distribution of fixation counts per trial (Item vs Relational) to inform the minimum fixation threshold decision.

**Status:** Completed and run. Output sent to Whitlock.

**Key output:** Histogram and retention curve showing that Item trials peak at 8–10 fixations and Relational trials peak at 12–15 fixations. 92 Item trials (3.1%) and 47 Relational trials (1.6%) have exactly 1 fixation.

---

### `fixation_distribution_analysis_v2.py`

**Purpose:** Same as v1 but with 1-fixation trials removed before plotting. This is the version Whitlock needs to set the minimum fixation threshold.

**Status:** Completed and run. Output sent to Whitlock. Awaiting threshold decision.

**Key changes from v1:** Added Step 2c (remove fixation_count == 1 trials), updated title, updated annotation box to show post-exclusion counts and cost of further thresholds.

### `step1_preprocessing.py`

**Purpose:** Preprocessing pipeline for the raw encoding fixation data.
Applies all confirmed exclusion decisions in sequence and outputs a
clean, analysis-ready CSV for feature extraction.

**Steps encoded:**
1. Load raw CSV and audit baseline counts
2. Filter to target==1 trials (class imbalance resolution)
3. Remove trials with fewer than 3 fixations
4. Exclude subjects with fewer than 23 valid trials (65% threshold)
5. Save cleaned data to encoding_data_clean.csv

**Status:** Complete. Exclusion report verified. Output: encoding_data_clean.csv
(83 subjects, 5,609 trials, 56,561 fixation rows). Ready for feature extraction.
---------------------------------------------------------------------------------
### `step2_feature_extraction.py`

**Purpose:** Computes all 20 trial-level eye movement features from the
cleaned fixation data. Outputs two feature matrices — one using the
primary thirds temporal split and one using the secondary halves split.

**Features computed:** obj_dwell_prop, scene_dwell_prop, obj_fix_count,
scene_fix_count, temporal dwell (thirds and halves), mean_fix_duration_ms,
first_fix_latency_obj_ms, obj_scene_transitions, transition_entropy,
obj_revisits, scene_revisits, scanpath_length_deg, fixation_dispersion,
saccade_amplitude_mean_deg.

**Key implementation notes:**
- Fixations sorted by Start time before any sequence-based computation
- Dispersion uses Ramey et al. 2020 k-means + silhouette method
- Saccade amplitude and scanpath length converted to degrees of visual angle
- NaN handled gracefully for trials with zero AOI fixations
- Built-in sanity check compares feature means by task against theoretical predictions

**Status:** Complete. All 5,609 trials processed successfully.
Outputs: feature_matrix_encoding.csv (19 features) and
feature_matrix_encoding_halves.csv (17 features).

### Feature Extraction Results

**Output dimensions:**
- Primary (thirds): 5,609 trials × 19 features (+ 4 metadata columns)
- Secondary (halves): 5,609 trials × 17 features (+ 4 metadata columns)

**NaN values by feature:**
| Feature | NaN count | % | Reason |
|---|---|---|---|
| first_fix_latency_obj_ms | 6 | 0.1% | Trials with zero object fixations |
| transition_entropy | 1,789 | 31.9% | Trials with zero cross-AOI transitions (mostly Item task — participants fixated only the object) |

**Sanity check — feature means by task:**
| Feature | ITEM | RELATIONAL | Expected | Result |
|---|---|---|---|---|
| obj_dwell_prop | 0.914 | 0.640 | ITEM > REL | ✓ |
| scene_dwell_prop | 0.086 | 0.360 | REL > ITEM | ✓ |
| obj_fix_count | 7.926 | 6.395 | ITEM > REL | ✓ |
| scene_fix_count | 1.123 | 4.692 | REL > ITEM | ✓ |
| obj_scene_transitions | 1.123 | 3.215 | REL > ITEM | ✓ |
| transition_entropy | 0.744 | 0.861 | REL > ITEM | ✓ |
| scanpath_length_deg | 17.666 | 40.942 | REL > ITEM | ✓ |
| fixation_dispersion | 543.094 | 1212.146 | REL > ITEM | ✓ |
| saccade_amplitude_mean_deg | 2.176 | 4.290 | REL > ITEM | ✓ |

All 9 theoretical predictions confirmed. Effect sizes are large across
all spatial and scene-directed features, consistent with the task design.
The transition_entropy NaN rate (31.9%) reflects a genuine behavioral
pattern — Item task participants frequently never switch away from the
object — and will be noted in the methods section.

### `step3_sanity_check.py`

**Purpose:** Pre-classifier quality gate. Verifies the feature matrix
is behaving as expected before any classifier is trained. Checks
NaN rates, univariate Item vs Relational comparisons with Cohen's d
effect sizes, per-subject consistency (% of subjects individually
showing the expected direction), and feature correlations.

**Outputs:**
- sanity_check_distributions.png — violin plots per feature
- sanity_check_correlations.png — full correlation heatmap
- sanity_check_consistency.png — per-subject consistency bar chart
- Printed statistical report with t-tests and Cohen's d

**Status:** Complete. All checks passed. Cleared to proceed to classifier.

### Sanity Check Results

**Check 1 — NaN audit:** Two features with expected NaN values only.
No unexpected missing data.

**Check 2 — Univariate comparisons:**
- Direction correct: 19/19 features
- Significant at p < 0.001: 19/19 features
- Large effects (|d| > 0.8): 12/19 features
- Top features by |d|: scene_fix_count (1.33), obj_dwell_prop (1.32),
  scene_dwell_prop (1.32), scanpath_length_deg (1.20),
  obj_scene_transitions (1.17)

**Check 3 — Per-subject consistency:**
- All 19 features above 80% threshold
- 4 features at 100%: scene_fix_count, obj_scene_transitions,
  scene_revisits, scanpath_length_deg
- Lowest: transition_entropy at 79.5% (expected — high NaN rate)

**Check 4 — High correlations (|r| > 0.90):**
- obj_dwell_prop ←→ scene_dwell_prop: r = -1.00 (mathematical complement)
- obj_dwell_prop ←→ scene_fix_count: r = -0.91 (shared AOI variance)
- scene_dwell_prop ←→ scene_fix_count: r = +0.91 (shared AOI variance)
- obj_scene_transitions ←→ obj_revisits: r = +0.95 (structurally related)
- obj_scene_transitions ←→ scene_revisits: r = +0.92 (structurally related)

All correlations are expected and require no action. Will be noted
in the methods section. Random Forest handles correlated features naturally.
---

### Scripts Planned (Not Yet Built)

- `preprocessing.py` — full preprocessing pipeline (Steps 1–6 above)
- `feature_extraction.py` — compute all 20 features per trial, output feature matrix
- `classifier.py` — LOSO cross-validation, AUC, bootstrap CI, permutation test
- `plots.py` — ROC curve, confusion matrix, SHAP figures
- `run_pipeline.py` — end-to-end entry point

---

## 11. Open Questions

The following decisions have been raised with Whitlock and are pending response:

| # | Question | Asked | Status |
|---|---|---|---|
| 1 | Minimum fixation count threshold (pending v2 plot review) | Yes | Closed |
| 2 | ±3 SD duration filter: global mean or per-subject mean? | Yes | Closed |
| 3 | Viewing distance for pixels→degrees conversion | Yes | ✅ 783mm (midpoint) — monitor dimensions pending |
| 4 | Test phase eye movement data file | Yes | ✅ Arriving soon |
| 5 | Subject count: why are subjects 39 and 61 missing? | Yes | ✅ Confirmed absent — 84 subjects total 
| 6 | TargetLocation coding error flagged to Whitlock | Yes | ✅ Confirmed — use CorrectTest |
| 7 | Monitor physical dimensions | Yes | ✅ 23.8 inches — conversion fully resolved |
---

## 12. Meeting Log

### Meeting 1 — Initial scoping meeting (May 2026)
**Participants:** Whitlock, student collaborator  
**Outcomes:** Confirmed project scope (binary classifier, item vs relational), confirmed target journal (Nature), agreed on Python as analysis environment, confirmed timeline (August–September 2025), established collaborator roles (Whitlock = psychological theory and interpretation; student = ML pipeline and analysis).

### Meeting 2 — Feature and design meeting (May 2026)
**Participants:** Whitlock, student collaborator  
**Outcomes:** Confirmed feature set (16 measures from document + saccade amplitude = 17), confirmed LOSO as validation strategy, confirmed statistical tests (permutation + bootstrap CI), confirmed positive class (Item = 1), confirmed behavioral data (accuracy and confidence) excluded as features, agreed on three figures (ROC, confusion matrix, SHAP), confirmed encoding and test phase as separate classifiers.

### Email exchange — June 2026
**Topic:** Resolution of all pre-build design questions  
**Outcomes:** Early/mid/late thirds split confirmed, transition entropy cross-AOI only confirmed, fixation dispersion referenced to Ramey et al. 2020, class imbalance resolved via target==1 filter, subject exclusion threshold set at 65%, 1-fixation trial removal confirmed, ±3 SD duration filter introduced (global vs per-subject pending), both AUCs to be reported, 0.5 confusion matrix threshold confirmed.

### Email exchange — June 2026 (retrieval data delivery)
**Topic:** Retrieval data delivery + final preprocessing decisions
**Outcomes:**
- Retrieval data (Item_Relational_Retrieval_Data.csv) received and audited
- Minimum fixation threshold confirmed: remove trials with 1 or 2 fixations (keep ≥ 3)
- ±3 SD duration filter removed from pipeline — does not apply to this project
- Viewing distance still pending (Whitlock will measure next lab visit)
- TargetLocation column coding error identified and flagged to Whitlock — use CorrectTest instead

### Email exchange — 6 June 2026 (final pre-build confirmations)
**Outcomes:**
- TargetLocation coding error confirmed — CorrectTest is correct
- Subjects 39 and 61 confirmed absent — dataset is definitively 84 subjects
- Viewing distance confirmed: 783mm (midpoint of 765mm and 800mm measurements)
- Monitor physical dimensions requested — needed for pixels-to-degrees conversion
- All other questions closed

### Step 1 preprocessing run — June 2026
Preprocessing script executed successfully in Google Colab.
Exclusion report verified. All numbers consistent with expectations.
One subject excluded (Subject 2, 22 ITEM trials — 1 below the 23-trial threshold).
Clean dataset saved as encoding_data_clean.csv.

### Step 2 feature extraction run — June 2026
Feature extraction script executed successfully in Google Colab.
5,609 trials processed. All 9 sanity checks passed with large effect sizes.
Two features have expected NaN values (first_fix_latency_obj_ms: 6 trials,
transition_entropy: 1,789 trials). Both outputs verified and saved.

### Step 3 sanity check run — June 2026
All 19 features passed direction check, all significant at p < 0.001,
12/19 large effects. Per-subject consistency exceptional — 4 features
at 100%, lowest at 79.5%. Five high-correlation pairs identified,
all structurally expected. Feature matrix cleared for classifier.

---

## 13. Key References

**Paradigm:**
- Baym, C. L., et al. (2014). Cited in experiment procedure — original paradigm reference.

**Eye movements and relational memory:**
- Ryan, J. D., Althoff, R. R., Whitlow, S., & Cohen, N. J. (2000). Amnesia is a deficit in relational memory. *Psychological Science, 11*, 454–461.
- Hannula, D. E., et al. (2010). Worth a glance: Using eye movements to investigate the cognitive neuroscience of memory. *Frontiers in Human Neuroscience, 4*, 166.

**Fixation dispersion measure:**
- Ramey, M. M., Henderson, J. M., & Yonelinas, A. P. (2020). The spatial distribution of attention predicts familiarity strength during encoding and retrieval. *Journal of Experimental Psychology: General, 149*(11), 2046–2062.

**SHAP explainability:**
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS.*
- Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence.*

**Classifier for cognitive state from eye movements:**
- Henderson, J. M., Shinkareva, S. V., Wang, J., Luke, S. G., & Olejarczyk, J. (2013). Predicting cognitive state from eye movements. *PLOS ONE.*
- Kardan, O., et al. (2015). Classifying mental states from eye movements during scene viewing. *Journal of Experimental Psychology: Human Perception and Performance, 41*, 1502–1514.
