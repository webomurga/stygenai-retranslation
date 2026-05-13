from __future__ import annotations
"""
Burrows' Delta authorship attribution method.

Classic 1-nearest-neighbour classifier using z-score normalised
word-frequency vectors and Manhattan (city-block) distance.

The cross-validation protocol is identical to that used for the ML classifiers
so results are directly comparable.
"""

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from .config import CONFIG


def _zscore_normalize(X_train: np.ndarray, X_test: np.ndarray):
    """Z-score normalise using training-set statistics only."""
    means = X_train.mean(axis=0)
    stds  = X_train.std(axis=0)
    stds[stds == 0] = 1.0           # Avoid division by zero
    return (X_train - means) / stds, (X_test - means) / stds


def _top_mfw_indices(X_train: np.ndarray, n_mfw: int) -> np.ndarray:
    """Return the column indices of the *n_mfw* most frequent words."""
    word_freqs = X_train.sum(axis=0)
    return np.argsort(word_freqs)[-n_mfw:]


def _predict_1nn_manhattan(X_train, y_train, X_test):
    """1-NN classification with Manhattan distance."""
    predictions = []
    for test_vec in X_test:
        distances = np.sum(np.abs(X_train - test_vec), axis=1)
        predictions.append(y_train[np.argmin(distances)])
    return np.array(predictions)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_burrows_delta_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_mfw: int = 100,
    n_repeats: int | None = None,
    n_folds: int | None = None,
) -> tuple[float, float]:
    """Burrows' Delta with repeated stratified k-fold cross-validation.

    At each fold:
    1. Z-score normalisation is fit on the training set only.
    2. The *n_mfw* most common words (by training-set frequency) are selected.
    3. 1-NN classification with Manhattan distance is performed.

    Parameters
    ----------
    X : ndarray, shape (n_books, n_features)
        Raw frequency matrix (not pre-normalised).
    y : ndarray, shape (n_books,)
        Integer or string class labels.
    n_mfw : int
        Number of most-frequent words to use.
    n_repeats, n_folds : override CONFIG defaults.

    Returns
    -------
    mean_f1, std_f1 : float
    """
    n_repeats = n_repeats or CONFIG["n_repeats"]
    n_folds   = n_folds   or CONFIG["n_folds"]

    repeat_scores: list[float] = []

    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rep)
        fold_scores: list[float] = []

        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            X_tr_z, X_te_z = _zscore_normalize(X_tr, X_te)

            top_idx = _top_mfw_indices(X_tr, n_mfw)
            X_tr_mfw = X_tr_z[:, top_idx]
            X_te_mfw = X_te_z[:, top_idx]

            y_pred = _predict_1nn_manhattan(X_tr_mfw, y_tr, X_te_mfw)
            fold_scores.append(f1_score(y_te, y_pred, average="macro"))

        repeat_scores.append(float(np.mean(fold_scores)))

    return float(np.mean(repeat_scores)), float(np.std(repeat_scores))
