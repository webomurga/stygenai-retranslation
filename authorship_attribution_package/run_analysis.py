#!/usr/bin/env python3
"""
Authorship Attribution Analysis
================================
Main entry point.  Runs binary classification, sample-size analysis, and/or
multiclass classification experiments on the pre-computed feature-vector
Excel files.

Usage
-----
::

    python run_analysis.py --all \
        --data-path ./data \
        --output-path ./output

Run specific experiments::

    python run_analysis.py --binary
    python run_analysis.py --sample-size
    python run_analysis.py --multiclass

Restrict to a subset of translators::

    python run_analysis.py --binary --translators yeginobali suveren disbudak

Select feature set::

    python run_analysis.py --binary --feature-set stylistic
    python run_analysis.py --binary --feature-set all   (default)

See ``python run_analysis.py --help`` for the full option list.
"""

import argparse
import os
import sys
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------
from analysis.config import TRANSLATORS, CONFIG
from analysis.data_loader import load_all_vectors, load_translator_keywords, load_all_keywords
from analysis.feature_selection import (
    stylistic_indices, keyword_indices, select_features
)
from analysis.classifiers import run_svm_cv, run_knn_cv, run_rf_cv
from analysis.burrows_delta import run_burrows_delta_cv
from analysis.sample_size import run_sample_size_analysis
from analysis.reporting import (
    save_binary_results,
    save_multiclass_results,
    plot_sample_size,
    print_results_table,
)

from analysis.visualizations import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_tsne_clusters, 
    plot_stylistic_violins,
    plot_radar_chart
)


# ---------------------------------------------------------------------------
# Helper — build binary dataset
# ---------------------------------------------------------------------------

def _build_binary(ref_X, trans_X):
    """Stack reference (label 0) and translator (label 1) feature matrices."""
    X = np.vstack([trans_X, ref_X])
    y = np.array([1] * len(trans_X) + [0] * len(ref_X))
    return X, y


# ---------------------------------------------------------------------------
# Binary classification
# ---------------------------------------------------------------------------

def run_binary(data, data_dir, output_dir, translators, feature_set):
    """Binary classification: one translator vs. reference corpus."""
    print("\n" + "=" * 70)
    print("BINARY CLASSIFICATION")
    print("=" * 70)

    ref_X          = data["ref"]["X"]
    feature_names  = data["ref"]["feature_names"]
    all_kw         = load_all_keywords(data_dir, translators)

    results: dict = {}

    for name in translators:
        if name not in data:
            print(f"\n  [SKIP] '{name}' data not found.")
            continue

        display   = data[name]["display"]
        trans_X   = data[name]["X"]
        trans_kw  = load_translator_keywords(data_dir, name)

        print(f"\n--- {display} (n={len(trans_X)}) vs Reference (n={len(ref_X)}) ---")

        X_full, y_full = _build_binary(ref_X, trans_X)

        results[name] = {}

        # ------- SVM experiments -------
        svm_configs = {
            "SVM: Stylistic + All KW":      ("stylistic+keywords", all_kw),
            "SVM: Stylistic + Filtered KW": ("stylistic+keywords", trans_kw),
            "SVM: Only All KW":             ("keywords",           all_kw),
            "SVM: Only Filtered KW":        ("keywords",           trans_kw),
            "SVM: Only Stylistic":          ("stylistic",          None),
        }

        for label, (mode, kw) in svm_configs.items():
            try:
                X_sub, _ = select_features(X_full, feature_names, mode, kw)
                mean, std = run_svm_cv(X_sub, y_full)
                results[name][label] = (mean, std)
                print(f"  {label:<42} {mean*100:>6.2f} ± {std*100:.2f}%")
            except ValueError as exc:
                print(f"  {label:<42} SKIP — {exc}")

        # ------- KNN (k=3 on all features) -------,
        try:
            X_sub, _ = select_features(X_full, feature_names, "all")
            mean, std = run_knn_cv(X_sub, y_full)
            results[name]["KNN (k=3, all features)"] = (mean, std)
            print(f"  {'KNN (k=3, all features)':<42} {mean*100:>6.2f} ± {std*100:.2f}%")
        except Exception as exc:
            print(f"  KNN: SKIP — {exc}")

        # ------- Burrows' Delta -------
        for n_mfw in CONFIG["mfw_counts"]:
            label = f"Burrows' Delta (MFW={n_mfw})"
            mean, std = run_burrows_delta_cv(X_full, y_full, n_mfw=n_mfw)
            results[name][label] = (mean, std)
            print(f"  {label:<42} {mean*100:>6.2f} ± {std*100:.2f}%")

    save_binary_results(results, output_dir)
    return results


# ---------------------------------------------------------------------------
# Sample-size analysis
# ---------------------------------------------------------------------------

def run_sample_size(data, data_dir, output_dir, translators):
    """Sample-size analysis: how many books are needed for robust attribution?"""
    print("\n" + "=" * 70)
    print("SAMPLE SIZE ANALYSIS")
    print("=" * 70)

    ref_X         = data["ref"]["X"]
    feature_names = data["ref"]["feature_names"]

    translator_data = {
        name: {"X": data[name]["X"], "display": data[name]["display"]}
        for name in translators
        if name in data
    }

    keywords_per_translator = {
        name: load_translator_keywords(data_dir, name)
        for name in translator_data
    }

    all_results, robust_thresholds = run_sample_size_analysis(
        ref_X=ref_X,
        translator_data=translator_data,
        feature_names=feature_names,
        keywords_per_translator=keywords_per_translator,
    )

    plot_sample_size(all_results, robust_thresholds, translator_data, output_dir)
    return all_results, robust_thresholds


# ---------------------------------------------------------------------------
# Multiclass classification
# ---------------------------------------------------------------------------

def run_multiclass(data, data_dir, output_dir, translators):
    """Multiclass classification: distinguish reference vs. all translators."""
    print("\n" + "=" * 70)
    print("MULTICLASS CLASSIFICATION")
    print("=" * 70)

    ref_X         = data["ref"]["X"]
    feature_names = data["ref"]["feature_names"]
    all_kw        = load_all_keywords(data_dir, translators)

    # Build joint dataset
    X_parts = [ref_X]
    y_parts = [np.zeros(len(ref_X), dtype=int)]
    class_labels = {0: "Reference"}

    for class_id, name in enumerate(translators, start=1):
        if name not in data:
            continue
        X_parts.append(data[name]["X"])
        y_parts.append(np.full(len(data[name]["X"]), class_id, dtype=int))
        class_labels[class_id] = data[name]["display"]

    X_all = np.vstack(X_parts)
    y_all = np.concatenate(y_parts)

    print(f"\nDataset: {len(X_all)} books, {len(class_labels)} classes")
    for cid, cname in class_labels.items():
        n = int((y_all == cid).sum())
        print(f"  Class {cid}: {cname} — {n} books")

    # Feature subsets to test
    sty_idx = stylistic_indices(feature_names)
    kw_idx  = keyword_indices(feature_names, all_kw)
    all_idx = list(dict.fromkeys(sty_idx + kw_idx))

    all_multiclass_results = {}

    experiments = [
        ("SVM — Stylistic + All Keywords", X_all[:, all_idx], "svm"),
        ("Random Forest — All Features",   X_all,             "rf"),
        ("KNN (k=3) — All Features",       X_all,             "knn"),
    ]

    for label, X_sub, clf_type in experiments:
        print(f"\n  {label}")
        if clf_type == "svm":
            mean, std = run_svm_cv(X_sub, y_all)
        elif clf_type == "rf":
            mean, std = run_rf_cv(X_sub, y_all)
        else:
            mean, std = run_knn_cv(X_sub, y_all)

        all_multiclass_results[label] = (mean, std)
        print(f"    Macro F1: {mean*100:.2f} ± {std*100:.2f}%")

    # Best result — detailed per-class breakdown re-run
    best_label = max(all_multiclass_results, key=lambda k: all_multiclass_results[k][0])
    macro_mean, macro_std = all_multiclass_results[best_label]
    print(f"\n  Best method: {best_label}")

    # Save
    save_multiclass_results(all_multiclass_results, output_dir)
    print_results_table(all_multiclass_results, "MULTICLASS SUMMARY")

    # -----------------------------------------------------------------------
    # NEW VISUALIZATION GENERATION BLOCK
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    # Filter out Reference (class 0) for visualizations to match the paper
    viz_mask = y_all != 0
    X_viz = X_all[viz_mask]
    y_viz = y_all[viz_mask]

    # Map remaining classes to match visualizations.py: CLASS_NAMES = ["Human", "NMT", "LLM"]
    # 0 = Human, 1 = NMT, 2 = LLM
    def map_label_to_int(y_int):
        name = class_labels[y_int].lower()
        if 'machine' in name or 'nmt' in name: return 1
        if 'llm' in name: return 2
        return 0

    y_viz_int = np.array([map_label_to_int(y) for y in y_viz])

    # Train a dedicated Random Forest for visualization purposes (feature importance & conf matrix)
    rf_viz = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_viz.fit(X_viz, y_viz_int)
    y_pred_int = cross_val_predict(rf_viz, X_viz, y_viz_int, cv=5)

    try:
        # 1. Confusion Matrix
        plot_confusion_matrix(y_viz_int, y_pred_int, output_path=os.path.join(output_dir, "confusion_matrix.png"))

        # 2. Feature Importance
        top_features = plot_feature_importance(rf_viz, feature_names, top_n=15, output_path=os.path.join(output_dir, "feature_importance.png"))

        # 3. t-SNE Clusters
        plot_tsne_clusters(X_viz, y_viz_int, output_path=os.path.join(output_dir, "tsne_clusters.png"))

        # 4. Violin Plots
        plot_stylistic_violins(X_viz, y_viz_int, feature_names, features_to_plot=top_features, output_path=os.path.join(output_dir, "violin_plots.png"))

        # 5. Radar Chart
        radar_categories = {
            'Morphological Density': ['Average morphemes per sentence', 'Average morphemes per word'],
            'Lexical Richness': ['TTR', 'Number of unique words t=10'],
            'Length Metrics': ['Mean sentence length', 'Mean word length'],
            'Punctuation': ['Ellipsis (Normalized)', 'Questions (Normalized)', 'Exclamations (Normalized)']
        }
        plot_radar_chart(X_viz, y_viz_int, feature_names, category_mapping=radar_categories, output_path=os.path.join(output_dir, "radar_chart.png"))
        
        print("\nAll visualizations generated successfully in the output directory.")
    except Exception as e:
        print(f"\nError generating visualizations: {e}")

    return all_multiclass_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_analysis",
        description=(
            "Authorship attribution analysis for translated literary corpora.\n"
            "Runs binary classification, sample-size analysis, and/or multiclass\n"
            "classification on pre-computed feature-vector Excel files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Paths
    parser.add_argument(
        "--data-path", default="./data", metavar="DIR",
        help="Directory containing xlsx and keyword files. Default: ./data",
    )
    parser.add_argument(
        "--output-path", default="./output", metavar="DIR",
        help="Directory for results and plots. Default: ./output",
    )

    # Analyses
    parser.add_argument(
        "--all", action="store_true",
        help="Run all three analyses.",
    )
    parser.add_argument(
        "--binary", action="store_true",
        help="Run binary classification (one translator vs. reference).",
    )
    parser.add_argument(
        "--sample-size", action="store_true",
        help="Run sample-size robustness analysis.",
    )
    parser.add_argument(
        "--multiclass", action="store_true",
        help="Run multiclass classification (all translators simultaneously).",
    )

    # Scope
    parser.add_argument(
        "--translators", nargs="+", metavar="NAME",
        default=list(TRANSLATORS.keys()),
        help=(
            "Translator(s) to include. Choose from: "
            f"{', '.join(TRANSLATORS)}. Default: all."
        ),
    )
    parser.add_argument(
        "--feature-set", choices=["all", "stylistic", "keywords"],
        default="all",
        help="Feature subset for binary SVM. Default: all.",
    )

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # Validate
    for name in args.translators:
        if name not in TRANSLATORS:
            parser.error(
                f"Unknown translator '{name}'. "
                f"Valid names: {', '.join(TRANSLATORS)}"
            )

    run_any = args.all or args.binary or args.sample_size or args.multiclass
    if not run_any:
        parser.print_help()
        print(
            "\nNo analysis selected. Use --all, --binary, --sample-size, "
            "or --multiclass."
        )
        sys.exit(0)

    data_dir   = os.path.abspath(args.data_path)
    output_dir = os.path.abspath(args.output_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nData path   : {data_dir}")
    print(f"Output path : {output_dir}")
    print(f"Translators : {', '.join(args.translators)}")

    # Load data once
    print("\nLoading feature vectors …")
    data = load_all_vectors(data_dir, args.translators)
    print(f"  Loaded {len(data)} groups (ref + {len(data)-1} translators)")

    # Run selected analyses
    if args.all or args.binary:
        run_binary(data, data_dir, output_dir, args.translators, args.feature_set)

    if args.all or args.sample_size:
        run_sample_size(data, data_dir, output_dir, args.translators)

    if args.all or args.multiclass:
        run_multiclass(data, data_dir, output_dir, args.translators)

    print(f"\nAll done.  Results written to: {output_dir}\n")


if __name__ == "__main__":
    main()