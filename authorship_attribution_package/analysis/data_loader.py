from __future__ import annotations
"""
Data loading utilities for the Authorship Attribution Analysis package.

Handles:
 - Loading feature-vector xlsx files produced by the Feature Extraction Package
 - Loading per-translator keyword (positive / negative) txt files
"""

import os

import numpy as np
import pandas as pd

from .config import TRANSLATORS, REFERENCE, N_METADATA_COLS


# ---------------------------------------------------------------------------
# Excel vector loading
# ---------------------------------------------------------------------------

def load_vectors(data_dir: str, prefix: str) -> tuple[list, list, list]:
    """Load a feature-vector xlsx file for one translator group.

    Parameters
    ----------
    data_dir :
        Directory containing the ``vectors_<prefix>_files.xlsx`` file.
    prefix :
        Short code for the translator (e.g. ``"ny"``).

    Returns
    -------
    X : ndarray, shape (n_books, n_features)
    feature_names : list of str
    book_labels : list of str  — ``"YYYY - Title"`` strings
    """
    path = os.path.join(data_dir, f"vectors_{prefix}_files.xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Vector file not found: {path}\n"
            f"Run the Feature Extraction Package first, or check your --data-path."
        )

    df = pd.read_excel(path)
    feature_names = list(df.columns[N_METADATA_COLS:])
    X = df.iloc[:, N_METADATA_COLS:].values.astype(float)
    book_labels = [
        f"{row.iloc[0]} - {row.iloc[1]}" for _, row in df.iterrows()
    ]
    return X, feature_names, book_labels


def load_all_vectors(
    data_dir: str,
    translators: list | None = None,
) -> dict:
    """Load vectors for all translators and the reference corpus.

    Parameters
    ----------
    data_dir :
        Directory that contains the xlsx files.
    translators :
        Subset of translator keys to load.  If ``None``, loads all translators
        defined in ``TRANSLATORS``.

    Returns
    -------
    dict  key -> {"X": ndarray, "feature_names": list, "labels": list,
                  "prefix": str, "display": str}
    """
    if translators is None:
        translators = list(TRANSLATORS.keys())

    data: dict = {}

    # Reference corpus first
    X, feat, labs = load_vectors(data_dir, REFERENCE["prefix"])
    data["ref"] = {
        "X": X,
        "feature_names": feat,
        "labels": labs,
        "prefix": REFERENCE["prefix"],
        "display": REFERENCE["display"],
    }

    # Translator corpora
    for name in translators:
        info = TRANSLATORS[name]
        try:
            X, feat, labs = load_vectors(data_dir, info["prefix"])
            data[name] = {
                "X": X,
                "feature_names": feat,
                "labels": labs,
                "prefix": info["prefix"],
                "display": info["display"],
            }
        except FileNotFoundError as exc:
            print(f"[WARN] {exc} — skipping '{name}'.")

    return data


# ---------------------------------------------------------------------------
# Keyword file loading
# ---------------------------------------------------------------------------

def load_translator_keywords(data_dir: str, translator: str) -> list:
    """Load positive + negative keywords for a single translator.

    Parameters
    ----------
    data_dir :
        Directory containing ``XX_pos.txt`` / ``XX_neg.txt``.
    translator :
        Translator key (e.g. ``"yeginobali"``).

    Returns
    -------
    Deduplicated list of keywords (words, not filtered).
    """
    prefix = TRANSLATORS[translator]["prefix"]
    words: list = []
    for suffix in ("_pos.txt", "_neg.txt"):
        path = os.path.join(data_dir, f"{prefix}{suffix}")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                words.extend(w.strip() for w in fh if w.strip())
    return words


def load_all_keywords(data_dir: str, translators: list | None = None) -> list:
    """Load and merge keywords for all (or given) translators.

    Returns a deduplicated sorted list.
    """
    if translators is None:
        translators = list(TRANSLATORS.keys())

    all_words: set = set()
    for name in translators:
        all_words.update(load_translator_keywords(data_dir, name))
    return sorted(all_words)
