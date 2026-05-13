"""
Reporting: save results to CSV + Excel, generate matplotlib plots.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import TRANSLATORS, CONFIG


# ---------------------------------------------------------------------------
# Save binary classification results
# ---------------------------------------------------------------------------

def save_binary_results(results: dict, output_dir: str) -> str:
    """Save binary classification results to CSV.

    Parameters
    ----------
    results :
        Dict  translator_name → { method_name → (mean_f1, std_f1) }.
    output_dir :
        Where to write the CSV.

    Returns
    -------
    Path to written file.
    """
    rows = []
    for translator, methods in results.items():
        display = TRANSLATORS.get(translator, {}).get("display", translator)
        for method, (mean_f1, std_f1) in methods.items():
            rows.append({
                "Translator": display,
                "Method":     method,
                "Mean F1 (%)":   f"{mean_f1 * 100:.2f}",
                "SD (%)":        f"{std_f1  * 100:.2f}",
            })

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "binary_classification_results.csv")
    df.to_csv(path, index=False)
    print(f"  → Saved {path}")
    return path


# ---------------------------------------------------------------------------
# Save multiclass results
# ---------------------------------------------------------------------------

def save_multiclass_results(
    results: dict,          # method_name → (mean_f1, std_f1)
    output_dir: str,
) -> str:
    """Save multiclass classification results to CSV."""
    rows = []
    for method, (m, s) in results.items():
        rows.append({"Method": method, "Mean Macro F1 (%)": f"{m*100:.2f}", "SD (%)": f"{s*100:.2f}"})

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "multiclass_classification_results.csv")
    df.to_csv(path, index=False)
    print(f"  → Saved {path}")
    return path


# ---------------------------------------------------------------------------
# Sample-size plots
# ---------------------------------------------------------------------------

_COLORS = {
#    "yeginobali": "#2E86AB",
#    "suveren":    "#A23B72",
#    "disbudak":   "#F18F01",
#    "gurses":     "#3BB273",
#    "taluy":      "#C85250",
##   FOR MAIN PAPER
#    "human":    "#F18F01",
#    "llm":      "#2E86AB",
#    "machine":  "#3BB273",
##   FOR NMT_HUMAN ABLATION
    "erten":    "#A23B72",
    "disbudak":   "#F18F01",
    "kina":    "#D16BA5",
    "machine":     "#3BB273",
    "pelit":      "#C85250",
    "yeginobali": "#2E86AB",
    "cakmakci": "#1B998B",
}


def _translator_color(name: str) -> str:
    return _COLORS.get(name, "#888888")


def plot_sample_size(
    all_results: dict,
    robust_thresholds: dict,
    translator_data: dict,
    output_dir: str,
) -> None:
    """Generate individual and combined sample-size analysis plots.

    Parameters
    ----------
    all_results :
        Dict  translator → { sample_size → (mean_f1, std_f1) }.
    robust_thresholds :
        Dict  translator → (size, mean_f1, std_f1).
    translator_data :
        Dict  translator → {"display": str, ...}.
    output_dir :
        Directory to save plots.
    """
    translators = list(all_results.keys())
    threshold = CONFIG["robust_threshold"] * 100

    # --- Individual subplots ---
    n = len(translators)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, translators):
        display = translator_data[name]["display"]
        data = all_results[name]
        sizes = sorted(data)
        means = [data[s][0] * 100 for s in sizes]
        stds  = [data[s][1] * 100 for s in sizes]
        color = _translator_color(name)

        ax.errorbar(sizes, means, yerr=stds, fmt="o-", color=color,
                    capsize=3, markersize=4, linewidth=1.5, alpha=0.85)
        ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1.5,
                   label=f"{threshold:.0f}% threshold")

        if name in robust_thresholds:
            thresh_size = robust_thresholds[name][0]
            ax.axvline(x=thresh_size, color="green", linestyle=":", linewidth=1.5, alpha=0.75)
            ax.annotate(f"{thresh_size} chunks", xy=(thresh_size, threshold - 8),
                        fontsize=9, color="green", ha="center", fontweight="bold")

        ax.set_xlabel("Number of Translation Chunks", fontsize=11)
        ax.set_ylabel("Macro F1 Score (%)", fontsize=11)
        ax.set_title(f"{display} (n={len(all_results[name])+4})", fontsize=12, fontweight="bold")
        ax.set_ylim(40, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)

    individual_path = os.path.join(output_dir, "sample_size_analysis.png")
    fig.savefig(individual_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved {individual_path}")

    # --- Combined plot ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    for name in translators:
        display = translator_data[name]["display"]
        data = all_results[name]
        sizes = sorted(data)
        means = [data[s][0] * 100 for s in sizes]
        stds  = [data[s][1] * 100 for s in sizes]

        ax2.errorbar(sizes, means, yerr=stds, fmt="o-",
                     color=_translator_color(name),
                     capsize=2, markersize=3, linewidth=1.5, alpha=0.85,
                     label=display)

    ax2.axhline(y=threshold, color="red", linestyle="--", linewidth=2,
                label=f"{threshold:.0f}% threshold")
    ax2.set_xlabel("Number of Translation Chunks", fontsize=12)
    ax2.set_ylabel("Macro F1 Score (%)", fontsize=12)
    ax2.set_title("Sample Size Analysis: All Translators", fontsize=14, fontweight="bold")
    ax2.set_ylim(40, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right")

    combined_path = os.path.join(output_dir, "combined_sample_size.png")
    fig2.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  → Saved {combined_path}")


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_results_table(results: dict, title: str = "Results") -> None:
    """Pretty-print a  method → (mean_f1, std_f1) dict."""
    print(f"\n{'='*65}")
    print(title)
    print(f"{'='*65}")
    print(f"  {'Method':<42} {'Mean F1':>8}  {'SD':>6}")
    print(f"  {'-'*42} {'-'*8}  {'-'*6}")
    for method, (mean, std) in sorted(results.items(), key=lambda x: -x[1][0]):
        print(f"  {method:<42} {mean*100:>7.2f}%  {std*100:>5.2f}%")
