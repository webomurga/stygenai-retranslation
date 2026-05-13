#!/usr/bin/env python3
"""
Unified aligned chunker:
- Aligns raw + morph at sentence level
- Writes outputs into SEPARATE folders
- Keeps shared segmentation logic ONLY during chunking
"""

import os
import re
import argparse


# -----------------------------
# CONFIG
# -----------------------------
SENT_CHUNK_SIZE = 25
SENT_OVERLAP = 5


# -----------------------------
# SENTENCE SPLITTING (RAW)
# -----------------------------
def split_sentences(text: str):
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


# -----------------------------
# MORPH PARSING
# -----------------------------
def extract_morph_sentences(lines):
    sentences = []
    current = []
    inside = False

    for line in lines:
        line = line.strip()

        if line.startswith("<S>"):
            inside = True
            current = [line]
            continue

        if line.startswith("</S>"):
            current.append(line)
            sentences.append(current)
            inside = False
            continue

        if inside:
            current.append(line)

    return sentences


# -----------------------------
# ALIGNMENT + CHUNKING
# -----------------------------
def chunk_aligned(raw_text, morph_sentences):
    raw_sents = split_sentences(raw_text)

    # safety alignment
    min_len = min(len(raw_sents), len(morph_sentences))
    raw_sents = raw_sents[:min_len]
    morph_sentences = morph_sentences[:min_len]

    raw_chunks = []
    morph_chunks = []

    start = 0
    n = min_len

    while start < n:
        end = min(start + SENT_CHUNK_SIZE, n)

        raw_chunks.append(raw_sents[start:end])
        morph_chunks.append(morph_sentences[start:end])

        start += SENT_CHUNK_SIZE - SENT_OVERLAP

    return raw_chunks, morph_chunks


# -----------------------------
# WRITERS (SEPARATE OUTPUTS)
# -----------------------------
def write_raw_chunks(chunks, out_dir, base):
    os.makedirs(out_dir, exist_ok=True)

    for i, chunk in enumerate(chunks):
        path = os.path.join(out_dir, f"{base}_chunk{i}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(" ".join(chunk))


def write_morph_chunks(chunks, out_dir, base):
    os.makedirs(out_dir, exist_ok=True)

    for i, chunk in enumerate(chunks):
        path = os.path.join(out_dir, f"{base}_chunk{i}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for sentence in chunk:
                for line in sentence:
                    f.write(line + "\n")


# -----------------------------
# PROCESS SINGLE PAIR
# -----------------------------
def process_file(raw_path, morph_path, raw_out, morph_out):

    base = os.path.splitext(os.path.basename(raw_path))[0]

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    with open(morph_path, "r", encoding="utf-8") as f:
        morph_lines = f.read().splitlines()

    morph_sentences = extract_morph_sentences(morph_lines)

    raw_chunks, morph_chunks = chunk_aligned(raw_text, morph_sentences)

    write_raw_chunks(raw_chunks, raw_out, base)
    write_morph_chunks(morph_chunks, morph_out, base)

    print(f"[OK] {base}: {len(raw_chunks)} aligned chunks")


# -----------------------------
# MAIN
# -----------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--raw", required=True, help="RAW input folder")
    parser.add_argument("--morph", required=True, help="MORPH input folder")

    args = parser.parse_args()

    raw_files = sorted([
        f for f in os.listdir(args.raw)
        if f.endswith(".txt")
    ])

    print("\nProcessing aligned corpus...\n")

    for raw_file in raw_files:

        raw_path = os.path.join(args.raw, raw_file)
        morph_path = os.path.join(args.morph, raw_file)

        if not os.path.exists(morph_path):
            print(f"[SKIP] missing morph: {raw_file}")
            continue

        process_file(
            raw_path,
            morph_path,
            raw_out=args.raw,
            morph_out=args.morph
        )

    print("\nDone.")


if __name__ == "__main__":
    main()