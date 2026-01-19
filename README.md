# MinHash Dedup (Jaccard Similarity) — JSONL Near-Duplicate Removal

A fast, scalable **near-duplicate removal** tool for large **JSONL datasets**, based on:

- **MinHash** signatures (Jaccard similarity estimation)
- **LSH (Locality Sensitive Hashing)** for candidate retrieval
- explicit **Jaccard verification** (MinHash-estimated) to reduce LSH false positives
- clustering with **Union-Find (DSU)**

Ideal for dataset cleaning before:
- **LLM fine-tuning / SFT**
- **RAG corpora**
- OCR / web-scraped text pipelines

---

## Features

✅ Near-duplicate detection (not only exact duplicates)  
✅ Char shingles (robust for OCR/noisy text)  
✅ Word shingles (better for clean text)  
✅ Jaccard similarity statistics for audit / tuning  
✅ Optional full similarity auditing (`--all-similarities`)  
✅ JSON report: clusters + kept/removed + similarity pairs  

---

## Installation

Python 3.10+ recommended.

```bash
pip install datasketch tqdm
```

---

## Input format

Input must be a **JSONL** file (one JSON object per line), e.g.:

```json
{"text":"Hello world"}
{"text":"Hello world!"}
{"text":"Completely different content"}
```

By default, the script reads the `text` field (configurable).

---

## Usage

### 1) Basic deduplication

```bash
python minhash_dedup.py -i dataset.jsonl -o dedup.jsonl -r report.json
```

### 2) Full similarity analysis (audit/debug mode)

```bash
python minhash_dedup.py -i dataset.jsonl -o dedup.jsonl -r report.json --all-similarities
```

### 3) Word shingles

```bash
python minhash_dedup.py -i dataset.jsonl -o dedup.jsonl -r report.json \
  --shingle-mode word --shingle-size 3 --threshold 0.85
```

---

## Threshold tuning

- **Higher threshold = stricter dedup** (keeps more data)
- **Lower threshold = more aggressive dedup** (removes more data)

Examples:

```bash
# Conservative: only very close duplicates
python minhash_dedup.py -i dataset.jsonl -o dedup.jsonl -r report.json --threshold 0.92

# Aggressive: removes more near-duplicates
python minhash_dedup.py -i dataset.jsonl -o dedup.jsonl -r report.json --threshold 0.75
```

> Note: Similarity here is **MinHash-estimated Jaccard**, not exact Jaccard.

---

## Recommended presets

### OCR / noisy text
```bash
--shingle-mode char --shingle-size 5 --strip-punct --threshold 0.85
```

### Clean instruction / Q&A datasets
```bash
--shingle-mode word --shingle-size 3 --threshold 0.88
```

### Higher accuracy
Increase `num_perm` (slower, more RAM):
```bash
--num-perm 256
```

---

## Outputs

### `dedup.jsonl`
Deduplicated JSONL dataset (keeps the first occurrence per cluster by default).

### `report.json`
Contains:
- config used
- global stats
- top clusters
- similarity pairs (if enabled)

Example:

```json
{
  "stats": {
    "total": 10000,
    "kept": 8200,
    "removed": 1800,
    "clusters": 8200,
    "largest_cluster": 12,
    "duplicate_pairs": 2400,
    "avg_similarity": 0.9123,
    "min_similarity": 0.85,
    "max_similarity": 1.0
  }
}
```

---

## How it works (high-level)

1. Normalize text (`lowercase`, optional punctuation removal)
2. Create shingles (character or word n-grams)
3. Compute MinHash signatures
4. Retrieve candidate duplicates via LSH
5. Compute MinHash-Jaccard similarity estimate for candidates
6. Union near-duplicates into clusters
7. Keep one representative per cluster

This provides a strong trade-off between:
- **speed** (LSH candidate lookup)
- **quality** (Jaccard verification)

---

## Limitations

- Jaccard similarity is **estimated**, not exact (due to MinHash).
- For exact Jaccard verification, you must store shingles and compute `|A∩B| / |A∪B|` (slower / more memory).
- For multi-million scale datasets, consider batching or Spark.

---

## License

MIT (or your preferred license)
