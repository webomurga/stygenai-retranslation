# Feature Extraction Package

Extracts stylistic and morphological feature vectors from a corpus of
translated books and writes one Excel (`.xlsx`) file per translator group.
These `.xlsx` files are the input for the **Authorship Attribution Analysis
Package**.

This package contains two main utilities:

- `chunk_text.py` — creates aligned sentence chunks from raw and
  morphological corpora
- `extract_features.py` — extracts stylistic and morphological feature vectors
  from aligned corpora

## Directory Structure

```
feature_extraction_package/
├── chunk_text.py         # aligned corpus chunker
├── extract_features.py   # feature extraction
├── requirements.txt
└── README.md
```

Your corpus must be organised with **parallel folder pairs** in the same root
directory:

```
<corpus_dir>/
├── ref_txt/           # reference corpus — plain text (one sentence per line)
├── ref_morph/         # reference corpus — morphological analysis
├── ny_txt/            # Yeğinobalı — plain text
├── ny_morph/          # Yeğinobalı — morphological analysis
├── gs_txt/
├── gs_morph/
├── ...
├── ny_pos.txt         # positive keywords (over-used) for prefix ny
├── ny_neg.txt         # negative keywords (under-used) for prefix ny
├── gs_pos.txt
├── gs_neg.txt
├── ...
└── kwd_filter.txt     # (optional) stopword list for keyword filtering
```

Folder pairs (`<prefix>_txt/` ↔ `<prefix>_morph/`) are discovered
automatically — no configuration needed.

## Preprocessing with `chunk_text.py`

Before feature extraction, large corpus files can optionally be segmented into
aligned sentence chunks using `chunk_text.py`.

The script:

- splits raw text into sentences
- aligns raw and morphological sentences
- creates overlapping chunks
- preserves sentence alignment between corpora

### Default Chunking Parameters

| Parameter | Value |
|---|---|
| Sentences per chunk | `25` |
| Sentence overlap | `5` |

Example overlap:

```text
Chunk 0 → sentences 1–25
Chunk 1 → sentences 21–45
Chunk 2 → sentences 41–65
```

### Usage

```bash
python chunk_text.py \
    --raw /path/to/ny_txt \
    --morph /path/to/ny_morph
```

### Output

Chunked files are written into the same folders:

```text
ny_txt/
├── Book_chunk0.txt
├── Book_chunk1.txt

ny_morph/
├── Book_chunk0.txt
├── Book_chunk1.txt
```

Files without matching morphological counterparts are skipped automatically.

## Morphological File Format

Each `.txt` file in a `*_morph/` folder should be the output of the Turkish
morphological analyser used in the project.  Sentence boundaries mark each
sentence:

```
<S> <S>+BSTag
word1  analysis1
word2  analysis2
</S> </S>+ESTag
<S> <S>+BSTag
...
```

## Requirements

```
Python 3.10+
pandas
openpyxl
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Typical Workflow

### 1. (Optional) Chunk large aligned corpora

```bash
python chunk_text.py --raw ny_txt --morph ny_morph
```

### 2. Extract feature vectors

```bash
python extract_features.py --corpus-dir /path/to/corpus
```

## Usage

### Basic — run with defaults (corpus dir = current dir)

```bash
python extract_features.py
```

### Specify corpus and output directories

```bash
python extract_features.py \
    --corpus-dir /path/to/corpus \
    --output-dir /path/to/output
```

### Keyword files in a different location

```bash
python extract_features.py \
    --corpus-dir /path/to/corpus \
    --keyword-dir /path/to/keywords \
    --output-dir /path/to/output
```

### Dry run (discover folders without writing anything)

```bash
python extract_features.py --corpus-dir /path/to/corpus --dry-run
```

## Outputs

One Excel file per translator group:

```
vectors_ref_files.xlsx
vectors_ny_files.xlsx
vectors_gs_files.xlsx
...
```

### Column layout

| Columns | Content |
|---------|---------|
| `book year`, `book title` | Metadata derived from filename (`YYYY - Title.txt`) |
| `average morphemes per sentence` | Morphological — from `*_morph/` files |
| `median morphemes per sentence` | Morphological |
| `average morphemes per word` | Morphological |
| `median morphemes per word` | Morphological |
| `TTR` | Type-Token Ratio |
| `Number of unique words` | Vocabulary size |
| `Number of unique words t=10` | Words appearing ≥10 times |
| `Mean word length` | Characters per word (mean) |
| `Standard deviation of word lengths` | |
| `Reduplications (Normalized)` | Adjacent repeated words / total words |
| `Ellipsis (Normalized)` | `...` and `…` / total words |
| `Questions (Normalized)` | `?` count / total words |
| `Exclamations (Normalized)` | `!` count / total words |
| `Mean sentence length` | Words per sentence (mean) |
| `Standard deviation of sentence lengths` | |
| `Median of sentence lengths` | |
| `Mode of sentence lengths` | |
| `<keyword>` × N | Normalised frequency of each focus word |

## Adding a New Translator

1. Create `<prefix>_txt/` and `<prefix>_morph/` in your corpus directory.
2. Optionally add `<prefix>_pos.txt` and `<prefix>_neg.txt` keyword files.
3. Run the script — the new group is detected automatically.
