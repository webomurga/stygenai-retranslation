"""
Configuration for the Authorship Attribution Analysis package.

Edit this file to add, remove, or rename translators.
"""

# ---------------------------------------------------------------------------
# Translator registry
# ---------------------------------------------------------------------------
# Keys    : internal name (used in code / output filenames)
# prefix  : the short code that appears in filenames (e.g. "ny" → vectors_ny_files.xlsx,
#            ny_pos.txt, ny_neg.txt)
# display : human-readable name used in printed reports and plots
# ---------------------------------------------------------------------------
TRANSLATORS: dict[str, dict[str, str]] = {
#    "yeginobali": {"prefix": "ny", "display": "Yeğinobalı"},
#    "suveren":    {"prefix": "gs", "display": "Suveren"},
#    "disbudak":   {"prefix": "bd", "display": "Dişbudak"},
#    "gurses":     {"prefix": "sg", "display": "Gürses"},
#    "taluy":      {"prefix": "yt", "display": "Taluy"},
#    "ezber":    {"prefix": "se", "display": "Ezber"},
#    "gulbahar":    {"prefix": "sg", "display": "Gülbahar"},
#    "burian":    {"prefix": "kb", "display": "Burian"},
#    "erten":    {"prefix": "ae", "display": "Erten"},
#    "unalan":    {"prefix": "nu", "display": "Ünalan"},
#    "gurbuz":    {"prefix": "fg", "display": "Gürbüz"},
#    "bitlisoglu":    {"prefix": "bb", "display": "Bitlisoğlu"},
### FOR MAIN PAPER
#    "human":    {"prefix": "human", "display": "Human"},
#    "llm":    {"prefix": "llm", "display": "LLM"},
#    "machine":    {"prefix": "machine", "display": "NMT"},
### FOR NMT_HUMAN ABLATION
    "erten": {"prefix": "ae", "display": "Erten"},
    "disbudak":   {"prefix": "bd", "display": "Dişbudak"},
    "kina":     {"prefix": "kk", "display": "Kına"},
    "machine":    {"prefix": "machine", "display": "NMT"},
    "pelit":      {"prefix": "mp", "display": "Pelit"},
    "yeginobali": {"prefix": "ny", "display": "Yeğinobalı"},
    "cakmakci":    {"prefix": "oc", "display": "Çakmakçı"},


}

# Reference corpus entry — treated as class 0 in all experiments
REFERENCE: dict[str, str] = {"prefix": "ref", "display": "Reference"}

# ---------------------------------------------------------------------------
# Cross-validation & classifier settings
# ---------------------------------------------------------------------------
CONFIG: dict[str, object] = {
    "random_seed":       0,
    "n_repeats":         30,      # Repetitions of stratified k-fold CV
    "n_folds":           2,       # Folds per CV repetition
    "svm_C":             10,      # SVM regularisation parameter
    "svm_gamma":         "scale", # SVM kernel coefficient
    "robust_threshold":  0.85,    # F1 threshold used in sample-size analysis
    "mfw_counts":        [100, 200, 500],  # MFW counts for Burrows' Delta
}

# ---------------------------------------------------------------------------
# Feature layout constants
# ---------------------------------------------------------------------------
# The first 6 columns in all xlsx files are metadata / morphological features:
#   book year, book title,
#   average morphemes per sentence, median morphemes per sentence,
#   average morphemes per word,     median morphemes per word
# The next 12 columns are stylistic features.
# Everything after column index 17 (0-based) is keyword features.
N_METADATA_COLS = 2          # Columns to skip when building feature matrix
N_STYLISTIC_FEATURES = 17    # Total morphological + stylistic features

STYLISTIC_FEATURE_NAMES: list[str] = [
    "average morphemes per sentence",
    "median morphemes per sentence",
    "average morphemes per word",
    "median morphemes per word",
    "TTR",
    "Number of unique words",
    "Number of unique words t=10",
    "Mean word length",
    "Standard deviation of word lengths",
    "Reduplications (Normalized)",
    "Ellipsis (Normalized)",
    "Questions (Normalized)",
    "Exclamations (Normalized)",
    "Mean sentence length",
    "Standard deviation of sentence lengths",
    "Median of sentence lengths ",
    "Mode of sentence lengths ",
]
