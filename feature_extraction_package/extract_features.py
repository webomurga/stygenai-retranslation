#!/usr/bin/env python3
"""
Feature Extraction for Translatorship Attribution
===================================================
Extracts stylistic and morphological feature vectors from a corpus of text
files and their corresponding morphological analyses, then writes the results
to Excel (.xlsx) files — one per translator group.

Input layout
------------
The corpus directory is expected to contain parallel folder pairs:

    <prefix>_txt/      — plain-text books (UTF-8, one sentence per line)
    <prefix>_morph/    — morphologically analysed counterparts

Each folder pair shares the same <prefix> (e.g. ``ny``, ``ref``, ``bd``).
The script discovers these pairs automatically.

Keyword files
-------------
Optional keyword files following the naming convention

    <prefix>_pos.txt   — words over-used by this translator (positive keyness)
    <prefix>_neg.txt   — words under-used by this translator (negative keyness)

are merged (after applying ``kwd_filter.txt`` if present) to form the set of
*focus-word* features appended after the stylistic features.

Morphological file format
-------------------------
Each ``.txt`` file in the ``*_morph/`` folders is the output of the Turkish
morphological analyser used in the project.  Expected line format::

    <S> <S>+BSTag          ← sentence boundary open
    surface  analysis      ← one word per line: surface<TAB>analysis
    </S> </S>+ESTag        ← sentence boundary close

Usage
-----
::

    python extract_features.py [OPTIONS]

Options
-------
--corpus-dir DIR      Root directory that contains the *_txt/*_morph folder
                      pairs.  Defaults to the current working directory.
--output-dir DIR      Directory where xlsx files are written.
                      Defaults to the current working directory.
--keyword-dir DIR     Directory that contains the XX_pos/neg.txt and
                      kwd_filter.txt files.  Defaults to --corpus-dir.
--dry-run             Print discovered folders and keyword files, then exit
                      without writing any output.
"""

import argparse
import glob
import os
import re
import statistics
import string
from collections import Counter, defaultdict

import pandas as pd


# ---------------------------------------------------------------------------
# Morphological feature extraction
# ---------------------------------------------------------------------------

def morph_stylistic_vector(morph_file: str) -> tuple[list, list]:
    """Extract morphological features from a morphologically analysed file.

    Returns
    -------
    vector : list
        [book_year, book_title,
         avg_morph_per_sent, median_morph_per_sent,
         avg_morph_per_word,  median_morph_per_word]
    index : list
        Corresponding feature names.
    """
    sent_morph_counts: list[int] = []
    word_morph_counts: list[int] = []

    with open(morph_file, "r", encoding="utf-8") as fh:
        current_sentence: list[int] = []
        for line in fh:
            line = line.strip()

            if line == "<S> <S>+BSTag":
                current_sentence = []
                continue

            if line == "</S> </S>+ESTag":
                if current_sentence:
                    sent_morph_counts.append(sum(current_sentence))
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            # Parse the analysis field: remove brackets/parens/+/- then split
            analysis = re.sub(r"\[|\]|\([^()]*\)|\+|-", "", parts[1]).strip()
            # Drop trailing character (morpheme separator)
            if analysis:
                analysis = analysis[:-1]
            morpheme_tokens = analysis.split()
            n_morphemes = len(morpheme_tokens)

            word_morph_counts.append(n_morphemes)
            current_sentence.append(n_morphemes)

    # Filter out sentence counts that are suspiciously large (data artefacts)
    sent_morph_counts = [c for c in sent_morph_counts if c < 200]

    if not sent_morph_counts or not word_morph_counts:
        raise ValueError(f"No morpheme data found in {morph_file}")

    avg_morph_per_sent = sum(sent_morph_counts) / len(sent_morph_counts)
    avg_morph_per_word = sum(word_morph_counts) / len(word_morph_counts)

    # Derive year and title from filename convention: YYYY - Title.txt
    basename = os.path.basename(morph_file)
    name_no_ext = os.path.splitext(basename)[0]
    parts_name = name_no_ext.split("-", 1)
    book_year = parts_name[0].strip() if len(parts_name) >= 1 else ""
    book_title = parts_name[1].strip() if len(parts_name) >= 2 else name_no_ext

    vector = [
        book_year,
        book_title,
        avg_morph_per_sent,
        statistics.median(sent_morph_counts),
        avg_morph_per_word,
        statistics.median(word_morph_counts),
    ]
    index = [
        "book year",
        "book title",
        "average morphemes per sentence",
        "median morphemes per sentence",
        "average morphemes per word",
        "median morphemes per word",
    ]
    return vector, index


# ---------------------------------------------------------------------------
# Stylistic feature extraction
# ---------------------------------------------------------------------------

def stylistic_vector(book_text: str, focus_words: list[str]) -> tuple[list, list]:
    """Extract stylistic features from raw book text.

    Parameters
    ----------
    book_text :
        Full text of the book (UTF-8).  Expected to have one sentence per line.
    focus_words :
        List of keywords whose normalised frequencies become extra features.

    Returns
    -------
    vector : list
        Feature values.
    index : list
        Corresponding feature names.
    """
    all_words = book_text.split()
    book_len = len(all_words)
    if book_len == 0:
        raise ValueError("Empty book text")

    table = str.maketrans("", "", string.punctuation)
    stripped = [w.translate(table).lower() for w in all_words]

    word_counter = Counter(stripped)
    word_len_list = [len(w) for w in stripped]
    sentence_len_list = [len(s.split()) for s in book_text.splitlines() if s.strip()]

    vector: list = []
    index: list = []

    def add(name, value):
        index.append(name)
        vector.append(value)

    # --- Type-Token Ratio ---
    add("TTR", len(word_counter) / book_len)

    # --- Vocabulary richness ---
    add("Number of unique words", len(word_counter))
    add("Number of unique words t=10",
        sum(1 for cnt in word_counter.values() if cnt >= 10))

    # --- Word length ---
    add("Mean word length", statistics.mean(word_len_list))
    add("Standard deviation of word lengths", statistics.stdev(word_len_list))

    # --- Reduplications (simple adjacent-word repetition, normalised) ---
    plain = " ".join(stripped)
    pattern = r"(\b\w+\b)\s+\1"
    n_reduplications = len(re.findall(pattern, plain)) / book_len
    add("Reduplications (Normalized)", n_reduplications)

    # --- Punctuation markers ---
    add("Ellipsis (Normalized)",
        (book_text.count("...") + book_text.count("…")) / book_len)
    add("Questions (Normalized)", book_text.count("?") / book_len)
    add("Exclamations (Normalized)", book_text.count("!") / book_len)

    # --- Sentence length ---
    if len(sentence_len_list) < 2:
        sentence_len_list = [1]  # Degenerate fallback
    add("Mean sentence length", statistics.mean(sentence_len_list))
    add("Standard deviation of sentence lengths",
        statistics.stdev(sentence_len_list) if len(sentence_len_list) > 1 else 0)
    add("Median of sentence lengths", statistics.median(sentence_len_list))
    add("Mode of sentence lengths", statistics.mode(sentence_len_list))

    # --- Keyword frequencies (normalised) ---
    for word in focus_words:
        add(word, word_counter[word] / book_len)

    return vector, index


# ---------------------------------------------------------------------------
# Keyword / focus-word loading
# ---------------------------------------------------------------------------

def load_focus_words(keyword_dir: str, prefixes: list[str]) -> list[str]:
    """Load and merge keyword files for all discovered prefixes.

    Parameters
    ----------
    keyword_dir :
        Directory containing ``XX_pos.txt`` / ``XX_neg.txt`` files.
    prefixes :
        List of translator prefixes (e.g. ``["ny", "gs", "bd"]``).

    Returns
    -------
    Deduplicated list of focus words (after applying ``kwd_filter.txt``).
    """
    filter_path = os.path.join(keyword_dir, "kwd_filter.txt")
    kwd_filter: set[str] = set()
    if os.path.exists(filter_path):
        with open(filter_path, "r", encoding="utf-8") as fh:
            kwd_filter = set(fh.read().splitlines())

    focus_words: set[str] = set()
    for prefix in prefixes:
        for suffix in ("_pos.txt", "_neg.txt"):
            kwd_file = os.path.join(keyword_dir, f"{prefix}{suffix}")
            if os.path.exists(kwd_file):
                with open(kwd_file, "r", encoding="utf-8") as fh:
                    for word in fh.read().splitlines():
                        word = word.strip()
                        if word and word not in kwd_filter:
                            focus_words.add(word)
    return sorted(focus_words)  # Deterministic order


# ---------------------------------------------------------------------------
# Corpus discovery
# ---------------------------------------------------------------------------

def discover_translator_groups(corpus_dir: str) -> dict[str, dict[str, str]]:
    """Discover translator groups from parallel *_txt / *_morph folder pairs.

    Returns
    -------
    dict mapping  prefix  →  {"txt": path, "morph": path}
    """
    groups: dict[str, dict[str, str]] = {}

    for entry in sorted(os.scandir(corpus_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        name = entry.name
        if name.endswith("_txt"):
            prefix = name[:-4]
            morph_dir = os.path.join(corpus_dir, f"{prefix}_morph")
            if os.path.isdir(morph_dir):
                groups[prefix] = {
                    "txt": entry.path,
                    "morph": morph_dir,
                }

    return groups


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_group(
    prefix: str,
    txt_dir: str,
    morph_dir: str,
    focus_words: list[str],
    output_dir: str,
) -> None:
    """Extract feature vectors for all books in one translator group.

    Writes the result to ``<output_dir>/vectors_<prefix>_files.xlsx``.
    """
    txt_files = sorted(glob.glob(os.path.join(txt_dir, "*.txt")))
    if not txt_files:
        print(f"  [WARN] No .txt files found in {txt_dir} — skipping.")
        return

    rows: list[list] = []
    columns: list[str] | None = None
    n_ok = 0
    n_err = 0

    for txt_path in txt_files:
        # Derive the corresponding morph file path
        morph_path = os.path.join(morph_dir, os.path.basename(txt_path))

        try:
            with open(txt_path, "r", encoding="utf-8") as fh:
                book_text = fh.read()

            morph_vec, morph_idx = morph_stylistic_vector(morph_path)
            style_vec, style_idx = stylistic_vector(book_text, focus_words)

            full_vec = morph_vec + style_vec
            full_idx = morph_idx + style_idx

            if columns is None:
                columns = full_idx

            rows.append(full_vec)
            n_ok += 1

        except FileNotFoundError as exc:
            print(f"  [WARN] Missing file — {exc}; skipping {os.path.basename(txt_path)}")
            n_err += 1
        except Exception as exc:
            print(f"  [WARN] Error processing {os.path.basename(txt_path)}: {exc}; skipping")
            n_err += 1

    if not rows:
        print(f"  [WARN] No vectors extracted for prefix '{prefix}'.")
        return

    out_path = os.path.join(output_dir, f"vectors_{prefix}_files.xlsx")
    pd.DataFrame(rows, columns=columns).to_excel(out_path, index=False, sheet_name="Sheet")
    print(f"  → Wrote {n_ok} vectors to {out_path}"
          + (f"  ({n_err} errors)" if n_err else ""))


def run_extraction(
    corpus_dir: str,
    output_dir: str,
    keyword_dir: str,
    dry_run: bool = False,
) -> None:
    """Orchestrate full feature extraction for all discovered translator groups."""

    print(f"\nCorpus directory : {os.path.abspath(corpus_dir)}")
    print(f"Output directory  : {os.path.abspath(output_dir)}")
    print(f"Keyword directory : {os.path.abspath(keyword_dir)}")

    groups = discover_translator_groups(corpus_dir)

    if not groups:
        print("\n[ERROR] No *_txt / *_morph folder pairs found.")
        return

    print(f"\nDiscovered {len(groups)} translator group(s):")
    for prefix, paths in groups.items():
        n_txt = len(glob.glob(os.path.join(paths["txt"], "*.txt")))
        n_morph = len(glob.glob(os.path.join(paths["morph"], "*.txt")))
        print(f"  [{prefix}]  txt={n_txt} files,  morph={n_morph} files")

    focus_words = load_focus_words(keyword_dir, list(groups.keys()))
    print(f"\nLoaded {len(focus_words)} focus words from keyword files")

    if dry_run:
        print("\n[DRY RUN] Exiting without writing output.")
        return

    os.makedirs(output_dir, exist_ok=True)

    print("\nExtracting features ...\n")
    for prefix, paths in groups.items():
        print(f"Processing [{prefix}] ...")
        extract_group(
            prefix=prefix,
            txt_dir=paths["txt"],
            morph_dir=paths["morph"],
            focus_words=focus_words,
            output_dir=output_dir,
        )

    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract stylistic + morphological feature vectors from a corpus "
            "of translator text files and write them to Excel (.xlsx) files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus-dir",
        default="./corpus_data/",
        metavar="DIR",
        help=(
            "Root directory containing the *_txt / *_morph folder pairs. "
            "Defaults to the current working directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="./output/",
        metavar="DIR",
        help=(
            "Directory where the output xlsx files are written. "
            "Defaults to the current working directory."
        ),
    )
    parser.add_argument(
        "--keyword-dir",
        default="./keywords_data/",
        metavar="DIR",
        help=(
            "Directory containing XX_pos.txt, XX_neg.txt, and kwd_filter.txt. "
            "Defaults to --corpus-dir."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovered folders and exit without writing any output.",
    )

    args = parser.parse_args()

    keyword_dir = args.keyword_dir if args.keyword_dir else args.corpus_dir

    run_extraction(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        keyword_dir=keyword_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
