from __future__ import annotations
"""
Sample-size analysis.

For each translator, tests how classification accuracy changes as the number of
translator books available grows from a minimum up to the full corpus size.
Identifies the smallest sample size at which performance reaches the robust
threshold defined in CONFIG.
"""

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import CONFIG, TRANSLATORS


def run_sample_size_analysis(
    ref_X: np.ndarray,
    translator_data: dict,   # name → {"X": ..., "display": ...}
    feature_names: list[str],
    keywords_per_translator: dict[str, list[str]],
    n_repeats: int | None = None,
    n_folds: int | None = None,
    random_seed: int | None = None,
) -> tuple[dict, dict]:
    """Run sample-size analysis for all translators.

    For each translator, samples between 5 and max_corpus_size books (without
    replacement) and measures SVM (stylistic + translator-specific keywords)
    F1. Identifies the robust threshold described in the paper.

    Parameters
    ----------
    ref_X :
        Feature matrix for the reference corpus (n_ref × n_features).
    translator_data :
        Dict mapping translator name → {"X": ndarray, "display": str}.
    feature_names :
        List of feature names aligned with columns of X matrices.
    keywords_per_translator :
        Dict mapping translator name → list of keywords.
    n_repeats, n_folds, random_seed :
        CV parameters (defaults from CONFIG).

    Returns
    -------
    all_results :
        Dict  translator → { sample_size → (mean_f1, std_f1) }.
    robust_thresholds :
        Dict  translator → (first_robust_size, mean_f1, std_f1).
    """
    n_repeats   = n_repeats   or CONFIG["n_repeats"]
    n_folds     = n_folds     or CONFIG["n_folds"]
    seed        = random_seed if random_seed is not None else CONFIG["random_seed"]
    threshold   = CONFIG["robust_threshold"]

    np.random.seed(seed)

    # Stylistic feature indices (first N_STYLISTIC_FEATURES columns)
    from .feature_selection import stylistic_indices, keyword_indices
    sty_idx = stylistic_indices(feature_names)

    all_results: dict = {}
    robust_thresholds: dict = {}

    for name, info in translator_data.items():
        display = info["display"]
        trans_X = info["X"]

        print(f"\n  [{display}] n={len(trans_X)} chunks")

        kw_idx  = keyword_indices(feature_names, keywords_per_translator.get(name, []))
        feat_idx = list(dict.fromkeys(sty_idx + kw_idx))   # Preserve insertion order, dedup

        max_size = len(trans_X)
        results: dict[int, tuple[float, float]] = {}

        for sample_size in range(5, max_size + 1):
            repeat_scores: list[float] = []

            for iteration in range(n_repeats):
                sampled_idx = np.random.choice(
                    max_size, size=sample_size, replace=False
                )
                X_trans_sample = trans_X[sampled_idx]

                X_all = np.vstack([X_trans_sample, ref_X])
                y_all = np.array([1] * sample_size + [0] * len(ref_X))

                # Select features
                X_feat = X_all[:, feat_idx]

                fold_scores: list[float] = []
                skf = StratifiedKFold(
                    n_splits=n_folds, shuffle=True, random_state=iteration
                )
                for tr_idx, te_idx in skf.split(X_feat, y_all):
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_feat[tr_idx])
                    X_te = scaler.transform(X_feat[te_idx])

                    clf = SVC(
                        kernel="rbf",
                        C=CONFIG["svm_C"],
                        gamma=CONFIG["svm_gamma"],
                        random_state=0,
                    )
                    clf.fit(X_tr, y_all[tr_idx])
                    y_pred = clf.predict(X_te)
                    fold_scores.append(
                        f1_score(y_all[te_idx], y_pred, average="macro")
                    )

                repeat_scores.append(float(np.mean(fold_scores)))

            mean_f1 = float(np.mean(repeat_scores))
            std_f1  = float(np.std(repeat_scores))
            results[sample_size] = (mean_f1, std_f1)

            # Detect robust threshold: mean ≥ t AND lower-bound ≥ t
            if (
                name not in robust_thresholds
                and mean_f1 >= threshold
                and (mean_f1 - std_f1) >= threshold
            ):
                robust_thresholds[name] = (sample_size, mean_f1, std_f1)
                print(
                    f"    Robust {threshold*100:.0f}% reached at "
                    f"{sample_size} chunks: {mean_f1*100:.1f} ± {std_f1*100:.1f}%"
                )

        if name not in robust_thresholds:
            print(
                f"    Never reached robust {threshold*100:.0f}% threshold"
            )

        all_results[name] = results

    return all_results, robust_thresholds
