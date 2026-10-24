"""Content hashing, duplicate detection, and leakage checks (PRD §9.6).

Exact duplicates: SHA-256 of case-folded, whitespace-collapsed text.
Near duplicates: Jaccard similarity of word-token sets (punctuation stripped)
>= ``NEAR_DUPLICATE_THRESHOLD`` for texts with at least ``MIN_TOKENS`` tokens.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

NEAR_DUPLICATE_THRESHOLD = 0.9
MIN_TOKENS = 4
NEAR_DUP_METHOD = (
    f"word-token-set Jaccard >= {NEAR_DUPLICATE_THRESHOLD} (lowercased, punctuation stripped, "
    f"texts with >= {MIN_TOKENS} tokens), candidate pairs blocked on shared tokens"
)

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9']+")


def normalize_text(text: str) -> str:
    return _WS.sub(" ", text.casefold()).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN.findall(text.casefold()))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class DedupResult:
    kept: list[int] = field(default_factory=list)
    duplicates: list[tuple[int, int]] = field(default_factory=list)  # (dropped_idx, kept_idx)


def dedupe_exact(texts: Sequence[str]) -> DedupResult:
    """Keep the first occurrence of each normalized text."""
    seen: dict[str, int] = {}
    res = DedupResult()
    for i, t in enumerate(texts):
        h = content_hash(t)
        if h in seen:
            res.duplicates.append((i, seen[h]))
        else:
            seen[h] = i
            res.kept.append(i)
    return res


def find_near_duplicates(
    left: Sequence[str],
    right: Sequence[str],
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> list[tuple[int, int, float]]:
    """Return (left_idx, right_idx, similarity) pairs with similarity >= threshold."""
    left_toks = [tokens(t) for t in left]
    right_toks = [tokens(t) for t in right]
    index: dict[str, list[int]] = defaultdict(list)
    for j, toks in enumerate(right_toks):
        if len(toks) >= MIN_TOKENS:
            for tok in toks:
                index[tok].append(j)
    out = []
    for i, toks in enumerate(left_toks):
        if len(toks) < MIN_TOKENS:
            continue
        # Jaccard >= t requires sharing at least t*|A| tokens; count shared tokens per candidate.
        counts: dict[int, int] = defaultdict(int)
        for tok in toks:
            for j in index.get(tok, ()):
                counts[j] += 1
        need = threshold * len(toks)
        for j, shared in counts.items():
            if shared >= need * 0.999:
                sim = jaccard(toks, right_toks[j])
                if sim >= threshold:
                    out.append((i, j, sim))
    return out


@dataclass
class LeakageReport:
    shared_source_ids: list[str]
    exact_overlaps: list[tuple[str, str]]
    near_overlaps: list[tuple[str, str, float]]

    @property
    def clean(self) -> bool:
        return not (self.shared_source_ids or self.exact_overlaps or self.near_overlaps)

    def to_dict(self) -> dict:
        return {
            "shared_source_ids": self.shared_source_ids,
            "exact_overlaps": [list(p) for p in self.exact_overlaps],
            "near_overlaps": [[a, b, round(s, 4)] for a, b, s in self.near_overlaps],
            "near_duplicate_method": NEAR_DUP_METHOD,
            "clean": self.clean,
        }


def check_leakage(
    dev: Iterable[tuple[str, str, str]],
    holdout: Iterable[tuple[str, str, str]],
) -> LeakageReport:
    """Each item is (case_id, source_record_id, text). Reports holdout items overlapping dev."""
    dev, holdout = list(dev), list(holdout)
    dev_src = {s for _, s, _ in dev if s}
    shared = sorted({s for _, s, _ in holdout if s and s in dev_src})
    dev_hash = {content_hash(t): cid for cid, _, t in dev}
    exact = [
        (cid, dev_hash[content_hash(t)]) for cid, _, t in holdout if content_hash(t) in dev_hash
    ]
    near_pairs = find_near_duplicates([t for _, _, t in holdout], [t for _, _, t in dev])
    exact_ids = {c for c, _ in exact}
    near = [
        (holdout[i][0], dev[j][0], s) for i, j, s in near_pairs if holdout[i][0] not in exact_ids
    ]
    return LeakageReport(shared, exact, near)
