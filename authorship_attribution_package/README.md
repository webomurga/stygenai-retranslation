# Authorship Attribution Analysis Package

Runs binary classification, sample-size analysis, and multiclass
classification on pre-computed feature-vector Excel files to perform
**translatorship attribution** for literary translations.

## Directory Structure

```
authorship_attribution_package/
├── run_analysis.py       # Main entry point
├── requirements.txt
├── README.md
├── analysis/             # Analysis sub-package
│   ├── config.py         # Translator registry & hyperparameters
│   ├── data_loader.py    # Load xlsx + keyword files
│   ├── feature_selection.py
│   ├── classifiers.py    # SVM, KNN, Random Forest (repeated CV)
│   ├── burrows_delta.py  # Burrows' Delta 1-NN
│   ├── sample_size.py    # Sample-size robustness analysis
│   └── reporting.py      # CSV output + matplotlib plots
├── data/                 # Input data (xlsx + keyword files)
│   ├── vectors_ref_files.xlsx
│   ├── vectors_ny_files.xlsx   # Yeğinobalı
│   ├── vectors_gs_files.xlsx   # Suveren
│   ├── vectors_bd_files.xlsx   # Dişbudak
│   ├── vectors_sg_files.xlsx   # Gürses
│   ├── vectors_yt_files.xlsx   # Taluy
│   ├── ny_pos.txt / ny_neg.txt
│   ├── gs_pos.txt / gs_neg.txt
│   ├── bd_pos.txt / bd_neg.txt
│   ├── sg_pos.txt / sg_neg.txt
│   └── yt_pos.txt / yt_neg.txt
└── output/               # Results written here
```

## Requirements

```
Python 3.10+
pandas, numpy, scikit-learn, matplotlib, openpyxl, scipy
```

```bash
pip install -r requirements.txt
```

## Usage

### Run all analyses

```bash
python run_analysis.py --all \
    --data-path ./data \
    --output-path ./output
```

### Run specific analyses

```bash
# Binary classification only
python run_analysis.py --binary

# Sample-size robustness analysis
python run_analysis.py --sample-size

# Multiclass (all translators vs. reference)
python run_analysis.py --multiclass
```

### Restrict to a subset of translators

```bash
python run_analysis.py --binary --translators yeginobali suveren disbudak
```

### Select feature sets for binary classification

```bash
python run_analysis.py --binary --feature-set stylistic   # morph+stylistic only
python run_analysis.py --binary --feature-set all         # + keywords (default)
```

## Outputs

| File | Description |
|------|-------------|
| `binary_classification_results.csv` | Mean F1 ± SD for each classifier / feature configuration |
| `multiclass_multiclass_best_results.csv` | Multiclass experiment summary |
| `sample_size_analysis.png` | Individual per-translator sample-size curves |
| `combined_sample_size.png` | All translators on one plot |

## Data Format

### Vector files (`vectors_XX_files.xlsx`)

Produced by the **Feature Extraction Package**.  Expected columns:

| Columns | Content |
|---------|---------|
| 1–2 | `book year`, `book title` (metadata) |
| 3–6 | Morphological features (avg/median morphemes per sentence/word) |
| 7–18 | Stylistic features (TTR, word/sentence statistics, punctuation) |
| 19+ | Keyword frequency features (one column per focus word) |

### Keyword files (`XX_pos.txt`, `XX_neg.txt`)

One keyword per line.  Positive = over-used by translator; negative = under-used.

## Adding a New Translator

1. Add a new entry to the `TRANSLATORS` dict in `analysis/config.py`:

```python
TRANSLATORS = {
    ...
    "newtranslator": {"prefix": "XX", "display": "New Translator Name"},
}
```

2. Place `vectors_XX_files.xlsx`, `XX_pos.txt`, and `XX_neg.txt` in `./data/`.

3. Run:

```bash
python run_analysis.py --all --translators newtranslator
```

## Methodology

### Cross-Validation Protocol

- **Repeated stratified 5-fold CV** × 30 repetitions
- All preprocessing (scaling, MFW selection) fit on training folds only
- Results: **mean ± SD** across 30 × 5-fold experiments

### Classifiers

| Classifier | Notes |
|------------|-------|
| SVM (RBF) | C=10, gamma=scale; StandardScaler per fold |
| KNN (k=3) | Min-max normalised vectors |
| Random Forest | 100 trees |
| Burrows' Delta | Z-score per fold; 1-NN with Manhattan distance |

### Robust Threshold (Sample-Size Analysis)

Minimum sample size where **both** mean F1 ≥ 85% **and** (mean − SD) ≥ 85%.
This ensures stability rather than just a lucky peak.

## Configuration

Edit `analysis/config.py` to change CV parameters:

```python
CONFIG = {
    "n_repeats":        30,
    "n_folds":          5,
    "svm_C":            10,
    "robust_threshold": 0.85,
    "mfw_counts":       [100, 200, 500],
}
```
