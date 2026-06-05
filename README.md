# 👁️ Eye-Tracking Memory Task Classifier

> Can the way you move your eyes during learning predict *what kind* of memory you're forming?

This project builds a binary machine learning classifier that predicts whether a trial came from an **Item Memory task** or a **Relational Memory task** — using only eye movement features recorded during encoding. It is part of an ongoing research collaboration targeting publication in **Nature**.

---

## 🧠 The Science

When people learn, the nature of what they're encoding shapes *how* their eyes move. Two fundamentally different encoding strategies produce detectably different gaze patterns:

| Item Memory | Relational Memory |
|---|---|
| Encode object features in isolation | Encode how an object relates to its scene |
| Eyes stay on the object | Eyes move between object and scene |
| Shorter saccades, focal scanning | Longer saccades, exploratory scanning |
| Fewer object↔scene transitions | More object↔scene transitions |

This project asks: **can a classifier learn to distinguish these strategies from eye movements alone?**

---

## 📁 Repository Structure

```
eye-tracking-memory-classifier/
├── data/
│   ├── raw/              # Original EyeLink CSVs — never modified
│   └── processed/        # Trial-level feature matrices
├── notebooks/            # Exploratory analysis
├── src/
│   ├── preprocessing/    # Data loading, cleaning, quality checks
│   ├── features/         # Trial-level feature extraction
│   ├── classifier/       # LOSO cross-validation, AUC, permutation test
│   └── visualization/    # ROC curve, confusion matrix, SHAP plots
├── figures/              # Publication-quality outputs (300 DPI)
├── results/              # Saved model outputs and AUC scores
├── docs/                 # Meeting notes, decisions log, references
├── run_pipeline.py       # End-to-end pipeline entry point
└── requirements.txt
```

---

## 🔬 Experiment Design

Participants (~84) completed two encoding tasks while eye movements were recorded with an **EyeLink 1000**:

- **Item Task** — 36 trials. One repeated scene, one unique object per trial. Participants judged whether the object would fit in a shoebox (item-focused encoding).
- **Relational Task** — 108 trials. Unique scene on every trial. Participants judged how well the object fit with the scene (relational encoding).

At test, participants selected the correct object from a 3-alternative forced-choice display.

---

## 📊 Feature Set (Encoding Phase)

All features are extracted at the **trial level** from raw fixation data:

| Feature | Description |
|---|---|
| Object dwell time proportion | % of total dwell time spent on the object |
| Scene dwell time proportion | % of total dwell time spent on the scene |
| Object fixation count | Number of fixations landing on the object |
| Scene fixation count | Number of fixations landing on the scene |
| Mean fixation duration | Average duration of all fixations in trial |
| Scanpath length | Total Euclidean distance traveled across fixations |
| Object→scene transition count | Number of gaze switches between object and scene |
| Transition entropy | Shannon entropy over AOI transition distribution |
| First fixation latency to object | Time (ms) to first fixation on the object |
| Number of object revisits | Returns to object AOI after leaving |
| Number of scene revisits | Returns to scene AOI after leaving |
| Fixation dispersion | Spatial spread of fixation coordinates |
| Early object dwell time | Object dwell in first half of encoding window |
| Late object dwell time | Object dwell in second half of encoding window |
| Early scene dwell time | Scene dwell in first half of encoding window |
| Late scene dwell time | Scene dwell in second half of encoding window |

---

## ⚙️ Classifier Pipeline

```
Raw fixation data
       ↓
Feature extraction (trial-level)
       ↓
LOSO cross-validation (84 folds)
       ↓
Random Forest classifier
       ↓
Pooled held-out predictions
       ↓
AUC + 95% bootstrap CI + permutation test
       ↓
SHAP feature importance
```

**Validation:** Leave-One-Subject-Out (LOSO) — train on 83 subjects, test on 1, repeat 84 times. Ensures zero participant-level data leakage and tests generalizability to completely unseen individuals.

**Statistical testing:** Permutation test (null AUC distribution, p-value) + bootstrapped 95% confidence intervals.

**Interpretability:** SHAP values (TreeExplainer) computed on a model trained across all subjects, revealing which features drive classification and whether they align with theoretical predictions.

---

## 🚀 Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python run_pipeline.py --data data/processed/feature_matrix.csv

# Encoding features only
python run_pipeline.py --data data/processed/feature_matrix.csv --phase encoding_only
```

---

## 👥 Collaboration

**PI:** Prof. Jonathan Whitlock  
**ML & Analysis:** Adi Singh  
**Institution:** Mississippi State University
**Status:** Active — targeting Nature publication

---

## ⚠️ Data & Reproducibility

Raw data is not included in this repository (unpublished). The full pipeline is reproducible from the raw EyeLink CSV files given the preprocessing steps documented in `src/preprocessing/`. All random seeds are fixed. All exclusion decisions are logged in `docs/decisions_log.md`.

---

*Eye-tracking data collected using EyeLink 1000 (SR Research). Analysis in Python 3.10+.*
