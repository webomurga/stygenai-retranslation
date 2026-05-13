from __future__ import annotations
"""
ML classifiers: SVM, KNN, Random Forest — cross-validated.

All functions return ``(mean_f1, std_f1)`` tuples so they can be
compared uniformly against Burrows' Delta results.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import CONFIG


# ---------------------------------------------------------------------------
# Single-fold helpers
# ---------------------------------------------------------------------------

def _scale(X_train, X_test):
    """Fit a StandardScaler on train, apply to both splits."""
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)


# ---------------------------------------------------------------------------
# Repeated stratified k-fold CV — core engine
# ---------------------------------------------------------------------------

def run_repeated_cv(
    X: np.ndarray,
    y: np.ndarray,
    make_clf,
    n_repeats: int | None = None,
    n_folds: int | None = None,
    random_seed: int | None = None,
    scale: bool = True,
) -> tuple[float, float]:
    """Repeated stratified k-fold cross-validation.

    Parameters
    ----------
    X, y : feature matrix and labels
    make_clf : callable → sklearn estimator (called fresh each fold)
    n_repeats, n_folds, random_seed : override CONFIG if given
    scale : apply StandardScaler per fold

    Returns
    -------
    mean_f1, std_f1 : float
    """
    n_repeats = n_repeats or CONFIG["n_repeats"]
    n_folds   = n_folds   or CONFIG["n_folds"]
    seed      = random_seed if random_seed is not None else CONFIG["random_seed"]

    np.random.seed(seed)
    repeat_scores: list[float] = []

    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rep)
        fold_scores: list[float] = []

        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            if scale:
                X_tr, X_te = _scale(X_tr, X_te)

            clf = make_clf()
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            fold_scores.append(f1_score(y_te, y_pred, average="macro"))

        repeat_scores.append(float(np.mean(fold_scores)))

    return float(np.mean(repeat_scores)), float(np.std(repeat_scores))


# ---------------------------------------------------------------------------
# Public classifier functions
# ---------------------------------------------------------------------------

def run_svm_cv(
    X: np.ndarray,
    y: np.ndarray,
    C: float | None = None,
    gamma: str | float | None = None,
    **kwargs,
) -> tuple[float, float]:
    """SVM (RBF kernel) with repeated stratified CV.

    Returns (mean_macro_f1, std_macro_f1).
    """
    c     = C     if C     is not None else CONFIG["svm_C"]
    gamma = gamma if gamma is not None else CONFIG["svm_gamma"]

    def make_clf():
        return SVC(kernel="rbf", C=c, gamma=gamma, random_state=0)

    return run_repeated_cv(X, y, make_clf, scale=True, **kwargs)


def run_knn_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_neighbors: int = 3,
    **kwargs,
) -> tuple[float, float]:
    """k-NN classifier with repeated stratified CV.

    Returns (mean_macro_f1, std_macro_f1).
    """
    def make_clf():
        return KNeighborsClassifier(n_neighbors=n_neighbors)

    return run_repeated_cv(X, y, make_clf, scale=False, **kwargs)


def run_rf_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    **kwargs,
) -> tuple[float, float]:
    """Random Forest with repeated stratified CV.

    Returns (mean_macro_f1, std_macro_f1).
    """
    def make_clf():
        return RandomForestClassifier(
            n_estimators=n_estimators, random_state=0
        )

    return run_repeated_cv(X, y, make_clf, scale=False, **kwargs)
