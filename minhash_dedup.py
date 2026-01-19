#!/usr/bin/env python3
"""
MinHash + LSH deduplication for JSONL datasets.

- Input:  JSONL (one JSON object per line)
- Output: deduplicated JSONL + JSON report (clusters/stats/similarities)

Default behavior:
- Uses one text field (configurable) to compute MinHash signatures
- Groups near-duplicates using LSH for candidate generation
- Verifies candidates by computing actual Jaccard similarity estimates
- Keeps the first occurrence in each cluster (configurable strategy possible)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Tuple

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

# Default random seed for reproducibility
DEFAULT_SEED = 42


# ----------------------------
# Text normalization & shingling
# ----------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_text(text: str | None, *, lowercase: bool = True, strip_punct: bool = False) -> str:
    """Basic text normalization."""
    if text is None:
        return ""
    if lowercase:
        text = text.lower()
    if strip_punct:
        text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def iter_shingles(text: str, *, mode: str, n: int) -> Generator[str, None, None]:
    """
    Generate shingles from text.
    mode:
      - "char": character n-grams
      - "word": word n-grams
    """
    if not text:
        return
    if mode == "char":
        if len(text) <= n:
            yield text
            return
        for i in range(len(text) - n + 1):
            yield text[i:i + n]
    elif mode == "word":
        words = text.split()
        if len(words) <= n:
            yield " ".join(words)
            return
        for i in range(len(words) - n + 1):
            yield " ".join(words[i:i + n])
    else:
        raise ValueError(f"Unknown shingle mode: {mode}")


def build_minhash(
    text: str,
    *,
    num_perm: int,
    shingle_mode: str,
    shingle_size: int,
    lowercase: bool,
    strip_punct: bool,
    seed: int = DEFAULT_SEED,
) -> MinHash:
    """Build a MinHash signature for the given text."""
    text = normalize_text(text, lowercase=lowercase, strip_punct=strip_punct)
    mh = MinHash(num_perm=num_perm, seed=seed)
    for sh in iter_shingles(text, mode=shingle_mode, n=shingle_size):
        mh.update(sh.encode("utf-8", errors="ignore"))
    return mh


# ----------------------------
# Union-Find for clustering
# ----------------------------

class UnionFind:
    """Disjoint Set Union with path compression and union by rank."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find root with path compression."""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        """Union by rank. Returns True if a merge happened."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
        return True


# ----------------------------
# Similarity pair storage
# ----------------------------

@dataclass
class SimilarityPair:
    """Represents a pair of documents with their Jaccard similarity."""
    idx_a: int
    idx_b: int
    similarity: float

    def to_dict(self) -> Dict[str, Any]:
        return {"idx_a": self.idx_a, "idx_b": self.idx_b, "similarity": round(self.similarity, 4)}


# ----------------------------
# Main dedup logic
# ----------------------------

@dataclass
class DedupStats:
    total: int
    kept: int
    removed: int
    clusters: int
    largest_cluster: int
    duplicate_pairs: int  # Number of pairs above threshold
    avg_similarity: float  # Average similarity among duplicate pairs
    min_similarity: float  # Minimum similarity among duplicate pairs
    max_similarity: float  # Maximum similarity among duplicate pairs


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no}: {e}") from e
    return rows


def get_text(row: Dict[str, Any], text_field: str) -> str:
    """Extract text from a row, handling None and non-string values."""
    v = row.get(text_field, "")
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    # If field is not a string, stringify safely
    return json.dumps(v, ensure_ascii=False)


def validate_threshold(threshold: float) -> None:
    """Validate that threshold is in valid range."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")


def deduplicate(
    rows: List[Dict[str, Any]],
    *,
    text_field: str,
    threshold: float,
    num_perm: int,
    shingle_mode: str,
    shingle_size: int,
    lowercase: bool,
    strip_punct: bool,
    seed: int = DEFAULT_SEED,
    compute_all_similarities: bool = False,
) -> Tuple[List[int], Dict[str, Any], DedupStats]:
    """
    Deduplicate rows using MinHash + LSH with explicit Jaccard verification.

    Args:
        rows: List of JSON objects
        text_field: Field name containing text to compare
        threshold: Minimum Jaccard similarity to consider as duplicate (0-1)
        num_perm: Number of MinHash permutations (higher = more accurate)
        shingle_mode: "char" or "word" n-grams
        shingle_size: Size of n-grams
        lowercase: Whether to lowercase text
        strip_punct: Whether to remove punctuation
        seed: Random seed for reproducibility
        compute_all_similarities: If True, compute similarities for all LSH candidates
                                  (useful for analysis, but slower)

    Returns:
        kept_indices: indices of rows to keep
        report: dict with clusters, similarities & mapping
        stats: summary stats
    """
    validate_threshold(threshold)

    n = len(rows)
    if n == 0:
        report = {"clusters": [], "kept_indices": [], "removed_indices": [], "similarity_pairs": []}
        stats = DedupStats(
            total=0, kept=0, removed=0, clusters=0, largest_cluster=0,
            duplicate_pairs=0, avg_similarity=0.0, min_similarity=0.0, max_similarity=0.0
        )
        return [], report, stats

    # Build LSH index
    # Note: LSH threshold is used for candidate generation (may have false positives)
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    minhashes: List[MinHash] = []

    # Create MinHashes and insert into LSH
    for i, row in enumerate(tqdm(rows, desc="Building MinHashes", unit="doc")):
        text = get_text(row, text_field)
        mh = build_minhash(
            text,
            num_perm=num_perm,
            shingle_mode=shingle_mode,
            shingle_size=shingle_size,
            lowercase=lowercase,
            strip_punct=strip_punct,
            seed=seed,
        )
        minhashes.append(mh)
        lsh.insert(str(i), mh)

    # Cluster using Union-Find, but VERIFY similarities first
    uf = UnionFind(n)
    similarity_pairs: List[SimilarityPair] = []
    seen_pairs: set[Tuple[int, int]] = set()  # Avoid computing same pair twice

    for i in tqdm(range(n), desc="Querying LSH & computing similarities", unit="doc"):
        # Find candidates for doc i
        candidates = lsh.query(minhashes[i])

        for c in candidates:
            j = int(c)
            if j <= i:  # Skip self and already-seen pairs (we only process i < j)
                continue

            pair_key = (i, j)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # IMPORTANT: Compute actual Jaccard similarity estimate
            jaccard_sim = minhashes[i].jaccard(minhashes[j])

            # Store similarity info (for report/analysis)
            if compute_all_similarities or jaccard_sim >= threshold:
                similarity_pairs.append(SimilarityPair(idx_a=i, idx_b=j, similarity=jaccard_sim))

            # Only union if similarity is ABOVE threshold (filter LSH false positives)
            if jaccard_sim >= threshold:
                uf.union(i, j)

    # Build clusters
    clusters_map: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        clusters_map[uf.find(i)].append(i)

    clusters: List[List[int]] = list(clusters_map.values())
    clusters.sort(key=len, reverse=True)

    # Keep first index in each cluster (deterministic: smallest index)
    kept_indices: List[int] = []
    removed_indices: List[int] = []
    for cluster in clusters:
        cluster.sort()
        kept_indices.append(cluster[0])
        removed_indices.extend(cluster[1:])

    kept_indices.sort()
    removed_indices.sort()

    # Compute similarity statistics
    # Filter to only pairs that caused merges (above threshold)
    valid_pairs = [p for p in similarity_pairs if p.similarity >= threshold]
    similarities = [p.similarity for p in valid_pairs]

    largest = max((len(c) for c in clusters), default=0)
    stats = DedupStats(
        total=n,
        kept=len(kept_indices),
        removed=len(removed_indices),
        clusters=len(clusters),
        largest_cluster=largest,
        duplicate_pairs=len(valid_pairs),
        avg_similarity=sum(similarities) / len(similarities) if similarities else 0.0,
        min_similarity=min(similarities) if similarities else 0.0,
        max_similarity=max(similarities) if similarities else 0.0,
    )

    # Sort similarity pairs by similarity (descending) for the report
    similarity_pairs.sort(key=lambda p: p.similarity, reverse=True)

    report = {
        "config": {
            "text_field": text_field,
            "threshold": threshold,
            "num_perm": num_perm,
            "shingle_mode": shingle_mode,
            "shingle_size": shingle_size,
            "lowercase": lowercase,
            "strip_punct": strip_punct,
            "seed": seed,
        },
        "stats": stats.__dict__,
        "clusters": clusters[:200],  # limit in report to avoid huge files
        "kept_indices": kept_indices,
        "removed_indices": removed_indices[:2000],  # limit
        # Include top similarity pairs in report (most similar first)
        "similarity_pairs": [p.to_dict() for p in similarity_pairs[:1000]],
    }

    return kept_indices, report, stats


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write a list of dictionaries to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description="MinHash+LSH JSONL deduplication with Jaccard similarity estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic deduplication with default settings
  %(prog)s -i data.jsonl -o dedup.jsonl -r report.json

  # Stricter threshold (more aggressive dedup)
  %(prog)s -i data.jsonl -o dedup.jsonl -r report.json --threshold 0.7

  # Word-level shingling (better for longer texts)
  %(prog)s -i data.jsonl -o dedup.jsonl -r report.json --shingle-mode word --shingle-size 3

  # Higher accuracy (more permutations)
  %(prog)s -i data.jsonl -o dedup.jsonl -r report.json --num-perm 256
        """,
    )
    p.add_argument("--input", "-i", required=True, help="Input JSONL file")
    p.add_argument("--output", "-o", required=True, help="Output deduplicated JSONL file")
    p.add_argument("--report", "-r", required=True, help="Output JSON report file")
    p.add_argument("--text-field", default="text", help="Field to use as text (default: text)")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Jaccard similarity threshold for duplicates (0-1, default: 0.85)",
    )
    p.add_argument(
        "--num-perm",
        type=int,
        default=128,
        help="MinHash permutations - higher is more accurate but slower (default: 128)",
    )
    p.add_argument(
        "--shingle-mode",
        choices=["char", "word"],
        default="char",
        help="Shingle mode: 'char' for character n-grams, 'word' for word n-grams (default: char)",
    )
    p.add_argument(
        "--shingle-size",
        type=int,
        default=5,
        help="n-gram size (default: 5 for char mode, try 3 for word mode)",
    )
    p.add_argument("--no-lowercase", action="store_true", help="Disable lowercasing")
    p.add_argument("--strip-punct", action="store_true", help="Remove punctuation during normalization")
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    p.add_argument(
        "--all-similarities",
        action="store_true",
        help="Store all LSH candidate similarities in report (slower, larger report)",
    )
    args = p.parse_args()

    print(f"Loading {args.input}...")
    rows = load_jsonl(args.input)
    print(f"Loaded {len(rows)} documents.")

    kept_indices, report, stats = deduplicate(
        rows,
        text_field=args.text_field,
        threshold=args.threshold,
        num_perm=args.num_perm,
        shingle_mode=args.shingle_mode,
        shingle_size=args.shingle_size,
        lowercase=not args.no_lowercase,
        strip_punct=args.strip_punct,
        seed=args.seed,
        compute_all_similarities=args.all_similarities,
    )

    dedup_rows = [rows[i] for i in kept_indices]

    write_jsonl(args.output, dedup_rows)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print("DEDUPLICATION COMPLETE")
    print("=" * 50)
    print(f"Total documents:       {stats.total:,}")
    print(f"Kept:                  {stats.kept:,} ({100 * stats.kept / stats.total:.1f}%)" if stats.total else "Kept: 0")
    print(f"Removed:               {stats.removed:,} ({100 * stats.removed / stats.total:.1f}%)" if stats.total else "Removed: 0")
    print(f"Clusters:              {stats.clusters:,}")
    print(f"Largest cluster:       {stats.largest_cluster:,}")
    print("-" * 50)
    print("SIMILARITY STATISTICS")
    print("-" * 50)
    print(f"Duplicate pairs found: {stats.duplicate_pairs:,}")
    if stats.duplicate_pairs > 0:
        print(f"Average similarity:    {stats.avg_similarity:.4f}")
        print(f"Min similarity:        {stats.min_similarity:.4f}")
        print(f"Max similarity:        {stats.max_similarity:.4f}")
    print("-" * 50)
    print(f"Output file:           {args.output}")
    print(f"Report file:           {args.report}")
    print("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)