from __future__ import annotations
"""
Feature selection helpers.

Provides utilities for extracting stylistic and keyword feature indices
from the combined feature-name list stored in the xlsx files.
"""

from .config import N_STYLISTIC_FEATURES


def stylistic_indices(feature_names: list[str]) -> list[int]:
    """Return the column indices of the 17 stylistic/morphological features."""
    return list(range(min(N_STYLISTIC_FEATURES, len(feature_names))))


def keyword_indices(feature_names: list[str], keywords: list[str]) -> list[int]:
    """Return column indices of features matching the given keyword list."""
    kw_set = set(keywords)
    return [i for i, name in enumerate(feature_names) if name in kw_set]


def select_features(
    X,
    feature_names: list[str],
    mode: str = "all",
    keywords: list[str] | None = None,
):
    """Slice a feature matrix to the requested feature subset.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    feature_names : list of str  — must align with X columns
    mode : one of
        ``"all"``        — all features
        ``"stylistic"``  — only the 17 morpho-stylistic features
        ``"keywords"``   — only keyword frequency features
        ``"stylistic+keywords"``  — stylistic + specific keywords
    keywords :
        Required when *mode* is ``"keywords"`` or ``"stylistic+keywords"``.

    Returns
    -------
    X_sub : ndarray
    selected_names : list[str]
    """
    if mode == "all":
        return X, feature_names

    sty_idx = stylistic_indices(feature_names)

    if mode == "stylistic":
        idx = sty_idx
    elif mode == "keywords":
        if not keywords:
            raise ValueError("keywords must be provided for mode='keywords'")
        idx = keyword_indices(feature_names, keywords)
    elif mode in ("stylistic+keywords", "stylistic+kw"):
        if not keywords:
            raise ValueError("keywords must be provided for mode='stylistic+keywords'")
        kw_idx = keyword_indices(feature_names, keywords)
        seen: set[int] = set()
        idx = []
        for i in sty_idx + kw_idx:
            if i not in seen:
                seen.add(i)
                idx.append(i)
    else:
        raise ValueError(f"Unknown feature mode: {mode!r}")

    if not idx:
        raise ValueError(
            f"Feature selection mode={mode!r} produced an empty feature set. "
            "Check that keyword files are present and non-empty."
        )

    selected_names = [feature_names[i] for i in idx]
    return X[:, idx], selected_names
