# Gaze-Based Classification of Memory Encoding Strategy
### Eye-Tracking Machine Learning Classifier | Whitlock Lab

---

## Overview

This repository contains the full analysis pipeline for a binary machine learning classifier that predicts **memory encoding strategy** from eye movement data. Specifically, the classifier determines whether a given encoding trial came from an **Item Memory task** — where participants encoded individual object features — or a **Relational Memory task** — where participants encoded how an object related to its background scene.

The central hypothesis is that these two encoding strategies produce detectably different gaze signatures. Item encoding should produce focal, object-directed scanning, while relational encoding should produce distributed scanning with frequent transitions between the object and scene. A classifier trained on eye movement features should be able to exploit these differences to reliably distinguish the two tasks — and the features driving that classification should map onto the psychological theory.

This is a collaboration between **Prof. Jonathan Whitlock** (cognitive psychology, PI) and **[Your Name]** (ML and analysis). The work is targeting publication in **Nature**.

---

## Scientific Background

### The Memory Distinction

Human memory is not a single system. A foundational distinction in memory research separates **item memory** — the ability to recognize or recall individual objects or facts — from **relational memory** — the ability to remember how items are connected to each other or to their context. These two forms of memory are thought to rely on partially distinct neural systems, with the hippocampus playing a particularly important role in relational binding (Cohen & Eichenbaum, 1993; Eichenbaum et al., 1994).

Critically, the *process* of forming these two types of memory is different. Encoding an object's features requires attention focused on that object. Encoding a relationship between an object and a scene requires distributed attention — the eyes must move between the two elements, compare them, and integrate them into a unified representation.

### Why Eye Movements

Eye movements are the primary mechanism by which humans sample visual information from the world. Where you look determines what you encode. This means the gaze record is not merely a correlate of cognitive processing — it is the channel through which visual encoding happens (Henderson, 2003). Prior work has established that:

- The number of fixations during encoding predicts subsequent memory strength (Loftus, 1972; Kafkas & Montaldi, 2011)
- Object-to-scene gaze transitions during encoding are a direct behavioral index of relational binding (Ryan et al., 2000; Baym et al., 2014)
- The spatial dispersion of fixations during encoding predicts subsequent familiarity strength (Ramey et al., 2020)
- Saccade amplitude and fixation patterns differ systematically as a function of task demands (Castelhano et al., 2009; Henderson et al., 2013)

The present project extends this literature by asking whether a machine learning classifier — trained on a battery of theoretically motivated eye movement features — can reliably decode *which kind of memory* a participant was forming on a given trial.

### The Classification Problem

This is a **binary classification** problem. The two classes are:

| Class | Label | Description |
|---|---|---|
| Item task | 1 (positive) | Participant was encoding object features in isolation |
| Relational task | 0 (negative) | Participant was encoding object-scene relationship |

The classifier operates at the **trial level**: one feature vector per encoding trial, one binary prediction per trial. Features are extracted from raw fixation data recorded during the encoding window (0–4000ms). The classifier sees only eye movements — no behavioral outcomes (accuracy, confidence) are included as features.

---

## Experiment

### Participants

Approximately 84 participants completed the experiment. Eye movements were recorded continuously using an **SR Research EyeLink 1000** eye-tracker (1000 Hz sampling rate). All participants had normal or corrected-to-normal vision. Task order was counterbalanced across participants.

### Task Design

Participants completed both tasks in a single session. Each task had a distinct encoding instruction designed to orient processing toward either object features (Item) or object-scene relationships (Relational).

#### Item Task
- **Encoding:** 36 trials. The *same background scene* appeared on every trial. On each trial, a unique object was superimposed on the scene for **4 seconds**. Participants judged whether the object would fit inside a shoebox (item-oriented encoding instruction).
- **Test:** 3-alternative forced choice. Three objects (1 studied, 2 novel foils) were shown on the scene. Participants selected the studied object and rated their confidence.
- **Key property:** Because the scene repeats, it provides no discriminative information at test. Successful recognition depends entirely on memory for the object's features.

#### Relational Task
- **Encoding:** 108 trials across 3 study-test blocks (36 trials per block). Every trial used a *unique scene-object pair*. Participants judged how well the object fit with the background scene (relational encoding instruction).
- **Test:** 3-alternative forced choice. Three *studied* objects were shown on one studied scene. One object (the **associate**) had been paired with that scene during encoding; the other two were studied but paired with different scenes. Participants identified the associate.
- **Key property:** Because all three objects at test were studied, the scene is the only cue that can guide correct selection. Successful performance requires memory for the specific object-scene pairing.

### Apparatus

- **Eye-tracker:** SR Research EyeLink 1000, tower mount
- **Sampling rate:** 1000 Hz
- **Screen resolution:** 1920 × 1080 pixels
- **Viewing distance:** TBC (Whitlock, personal communication)
- **Stimulus presentation:** SR Research Experiment Builder

### Areas of Interest (AOIs)

The encoding display contains two regions of interest:

| AOI | Description | Column in data |
|---|---|---|
| Object | The object superimposed on the scene | `StudiedItem = TRUE` |
| Scene | The background scene | `StudiedItem = FALSE` |

AOIs are pre-defined and encoded directly in the dataset via the `StudiedItem` boolean column. No coordinate-based AOI mapping is required.

---

## Data

### Raw Data Format

The primary dataset is `Item_Relational_Encoding_Data.csv` — a fixation-level report with one row per fixation.

| Column | Type | Description |
|---|---|---|
| `Subject` | Integer | Participant ID |
| `Trial` | Integer | Trial number within session |
| `x` | Float | Fixation centroid x-coordinate (pixels) |
| `y` | Float | Fixation centroid y-coordinate (pixels) |
| `Start` | Integer | Fixation onset time (ms), relative to encoding display onset |
| `End` | Integer | Fixation offset time (ms) |
| `Duration` | Integer | Fixation duration (ms) |
| `TaskOrder` | String | Order in which tasks were completed |
| `Task` | String | `ITEM` or `RELATIONAL` |
| `object` | String | Name of the object stimulus |
| `scene` | String | Name of the scene stimulus |
| `target` | Integer | `1` = target/associate trial; `0` = foil trial |
| `StudiedItem` | Boolean | `TRUE` = fixation on object AOI; `FALSE` = fixation on scene AOI |

### Trial Selection

The raw data contains 3x more Relational trials than Item trials due to the task structure. To resolve this imbalance and ensure theoretical comparability across tasks, the analysis is restricted to **target trials only** (`target == 1`):

- **Item task:** `target` is always 1 (every encoded object goes to test)
- **Relational task:** `target == 1` marks the associate trial — the encoded object that will be the correct answer at test

This yields approximately **36 trials per task per subject**, producing a balanced classification problem where every trial in the analysis represents an encoding event where the encoded object goes on to be the correct retrieval target.

---

## Feature Set

All features are computed at the **trial level** from raw fixation data. The final feature matrix has one row per trial.

### AOI Dwell Features

| Feature | Definition |
|---|---|
| Object dwell proportion | Total Duration on object / total Duration in trial |
| Scene dwell proportion | Total Duration on scene / total Duration in trial |
| Object fixation count | Number of fixations on object |
| Scene fixation count | Number of fixations on scene |

### Temporal Dwell Features (Thirds Split — Primary)

The 4000ms encoding window is divided into thirds: **early** (0–1333ms), **middle** (1334–2666ms), **late** (2667–4000ms).

| Feature | Definition |
|---|---|
| Early object dwell time | Duration on object in 0–1333ms window |
| Middle object dwell time | Duration on object in 1334–2666ms window |
| Late object dwell time | Duration on object in 2667–4000ms window |
| Early scene dwell time | Duration on scene in 0–1333ms window |
| Middle scene dwell time | Duration on scene in 1334–2666ms window |
| Late scene dwell time | Duration on scene in 2667–4000ms window |

*A halves split (0–2000ms / 2001–4000ms) will also be computed as a secondary exploratory analysis.*

### Fixation Duration Features

| Feature | Definition |
|---|---|
| Mean fixation duration | Mean of all fixation durations in trial (ms) |
| First fixation latency to object | Onset time (ms) of the first fixation on the object AOI |

### Transition and Sequence Features

| Feature | Definition |
|---|---|
| Object-scene transition count | Number of cross-AOI gaze switches (both directions) |
| Transition entropy | Shannon H computed over {object to scene, scene to object} transition proportions. Higher values = more diverse, less stereotyped scanning. |
| Object revisit count | Number of returns to object AOI after leaving (any return = 1 revisit) |
| Scene revisit count | Number of returns to scene AOI after leaving |

### Spatial Features

| Feature | Definition |
|---|---|
| Scanpath length | Sum of Euclidean distances between consecutive fixation centroids (pixels) |
| Fixation dispersion | k-means + silhouette composite: N_clusters x mean centroid distance. Quantifies spatial spread of fixation clusters across the display (Ramey et al., 2020). |
| Saccade amplitude (mean) | Mean Euclidean distance between consecutive fixation centroids (pixels) |

---

## Pipeline

```
Raw fixation data (CSV)
        |
        v
+-------------------------+
|      PREPROCESSING      |
|  - Filter target==1     |
|  - Remove 1-fix trials  |
|  - +-3 SD duration      |
|    filter               |
|  - Subject exclusions   |
|    (>=65% trials)       |
+----------+--------------+
           |
           v
+-------------------------+
|   FEATURE EXTRACTION    |
|  - 20 features          |
|  - Trial-level          |
|  - Both tasks           |
+----------+--------------+
           |
           v
+-------------------------+
|       CLASSIFIER        |
|  - Random Forest        |
|  - LOSO (84 folds)      |
|  - Pooled AUC +         |
|    per-subject AUC      |
+----------+--------------+
           |
           v
+-------------------------+
|   STATISTICAL TESTS     |
|  - Bootstrap 95% CI     |
|  - Permutation test     |
|    (1000 shuffles)      |
+----------+--------------+
           |
           v
+-------------------------+
|     EXPLAINABILITY      |
|  - SHAP values          |
|  - TreeExplainer        |
|  - Global importance    |
+-------------------------+
```

### Validation Strategy

**Leave-One-Subject-Out (LOSO) cross-validation** is the primary validation method. In each fold, the model is trained on all trials from 83 subjects and tested on all trials from the 1 held-out subject. This is repeated 84 times. LOSO directly tests whether the classifier generalizes to completely unseen individuals — the strongest form of generalizability evidence for a dataset of this structure.

### Statistical Testing

Classification performance is reported as:

> *AUC = X.XX [95% CI: X.XX–X.XX], permutation p < 0.001*

- **AUC:** Computed from pooled held-out predictions across all 84 LOSO folds
- **95% CI:** Bootstrapped by resampling held-out predictions 2000 times
- **p-value:** Permutation test — task labels shuffled 1000 times, LOSO rerun each time, observed AUC compared to null distribution

Both the overall pooled AUC and the mean per-subject AUC (± SD) are reported to eliminate researcher degrees of freedom in metric selection.

---

## Repository Structure

```
eye-tracking-memory-classifier/
|
+-- data/
|   +-- raw/                    # Original EyeLink CSVs -- never modified
|   +-- processed/              # Feature matrices, cleaned data
|
+-- notebooks/                  # Exploratory analysis notebooks
|
+-- src/
|   +-- preprocessing/          # Data loading, cleaning, exclusions
|   +-- features/               # Trial-level feature extraction
|   +-- classifier/             # LOSO, AUC, permutation test, bootstrap CI
|   +-- visualization/          # ROC curve, confusion matrix, SHAP plots
|
+-- figures/                    # Publication-quality outputs (300 DPI)
+-- results/                    # Saved model outputs, AUC scores (JSON)
+-- docs/                       # Meeting notes, decisions log
|
+-- run_pipeline.py             # End-to-end pipeline entry point
+-- requirements.txt            # Python dependencies
+-- DOCUMENTATION.md            # Full project documentation and decisions log
+-- README.md                   # This file
```

---

## Outputs

| Figure | Description |
|---|---|
| ROC curve | AUC + 95% CI with permutation null distribution as second panel |
| Confusion matrix | Raw counts + proportions, Item vs Relational, threshold = 0.5 |
| SHAP importance | Mean absolute SHAP bar chart + beeswarm plot across all 20 features |

---

## Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/[username]/eye-tracking-memory-classifier
cd eye-tracking-memory-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place raw data in data/raw/
# (Data not included -- unpublished)

# 4. Run full pipeline
python run_pipeline.py --data data/processed/feature_matrix_encoding.csv
```

---

## Current Status

| Stage | Status |
|---|---|
| Experiment design review | Complete |
| Data audit | Complete |
| Pre-build planning | Complete |
| Fixation distribution analysis | Complete |
| Preprocessing pipeline | Pending final threshold decisions from Whitlock |
| Feature extraction | Not started |
| Classifier (LOSO) | Not started |
| Statistical testing | Not started |
| SHAP explainability | Not started |
| Test phase classifier | Awaiting test phase data |
| Figures | Not started |

---

## Dependencies

```
numpy >= 1.24.0
pandas >= 2.0.0
scipy >= 1.10.0
scikit-learn >= 1.3.0
shap >= 0.44.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
joblib >= 1.3.0
```

---

## Reproducibility

- All random seeds fixed at 42
- Raw data never modified — all preprocessing produces new files in `data/processed/`
- Every exclusion decision is logged with counts and justification in `DOCUMENTATION.md`
- All design decisions documented with source and rationale before any code was written
- Results saved with timestamps — no outputs are overwritten

---

## License

This repository is private and unpublished. All data, code, and documentation are confidential pending publication.

---

*Eye-tracking data collected using EyeLink 1000 (SR Research Ltd.). Analysis conducted in Python 3.10+.*
