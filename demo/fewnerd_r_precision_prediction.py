"""Compute R-precision for FewNERD using fine-type queries.

This script embeds both entity mentions and fine-grained FewNERD types with the
same LLM+MLP pipeline and then evaluates retrieval quality by treating each
fine type as a query.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple

import faiss
import numpy as np

from embedding_utils import embed_texts
from fewnerd_fine_types import FINE_TYPES


def load_entities(path: str) -> tuple[Dict[str, str], Dict[str, Set[str]]]:
    """Load entity texts and build fine-type mappings."""
    with open(path, "r", encoding="utf8") as f:
        records = {rec["id"]: rec for rec in json.load(f)}

    tid_to_text: Dict[str, str] = {}
    fine_type_to_ids: Dict[str, Set[str]] = defaultdict(set)
    for sid, rec in records.items():
        for idx, ent in enumerate(rec.get("gold", [])):
            tid = f"{sid}-{idx}"
            tid_to_text[tid] = ent["text"]
            fine_type_to_ids[ent["label"]].add(tid)
    return tid_to_text, fine_type_to_ids


def r_precision(
    embeddings: Dict[str, List[Sequence[float]]],
    fine_type_embeddings: Dict[str, Sequence[float]],
    fine_type_to_ids: Dict[str, Set[str]],
) -> Tuple[Dict[str, float], float]:
    """Compute R-precision for each fine type given query embeddings."""

    # Build FAISS index of all document embeddings.
    all_vecs: List[np.ndarray] = []
    index_to_tid: List[str] = []
    for tid, vecs in embeddings.items():
        for vec in vecs:
            all_vecs.append(np.asarray(vec, dtype="float32"))
            index_to_tid.append(tid)
    if not all_vecs:
        return {}, 0.0

    emb_mat = np.stack(all_vecs)
    faiss.normalize_L2(emb_mat)
    index = faiss.IndexFlatIP(emb_mat.shape[1])
    index.add(emb_mat)

    results: Dict[str, float] = {}

    for ft, qvec in fine_type_embeddings.items():
        relevant = fine_type_to_ids.get(ft, set())
        r = len(relevant)
        if r <= 0:
            continue

        q = np.asarray(qvec, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(q)
        max_k = min(4 * len(index_to_tid), index.ntotal)
        D, I = index.search(q, max_k)

        ranking: List[str] = []
        seen = set()
        for idx in I[0]:
            tid = index_to_tid[idx]
            if tid not in seen:
                ranking.append(tid)
                seen.add(tid)
            if len(ranking) >= r:
                break

        hits = sum(1 for tid in ranking[:r] if tid in relevant)
        results[ft] = hits / r

    macro = sum(results.values()) / len(results) if results else 0.0
    return results, macro


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entities", default="fewnerd_entities.json", help="Original entity JSON"
    )
    args = parser.parse_args()

    tid_to_text, fine_type_to_ids = load_entities(args.entities)

    # Embed entity mentions (documents)
    doc_embs = embed_texts(tid_to_text)
    doc_embs = {tid: [emb.numpy()] for tid, emb in doc_embs.items()}

    # Embed fine-type queries
    query_texts = {ft: ft for ft in FINE_TYPES}
    query_embs = {ft: emb.numpy() for ft, emb in embed_texts(query_texts).items()}

    scores, macro = r_precision(doc_embs, query_embs, fine_type_to_ids)

    print("R-precision per fine type:")
    for ft, sc in sorted(scores.items()):
        print(f"{ft}: {sc:.4f}")
    print(f"Average R-precision: {macro:.4f}")


if __name__ == "__main__":
    main()
