"""Compute R-Precision for FewNERD fine types using FAISS.

This script evaluates how well fine-type query embeddings retrieve the
corresponding sentences (text IDs) that contain gold entities of that
fine type. It follows the multi-vector R-precision logic: the index is
built over all entity embeddings (multiple per sentence). Retrieval is
performed with FAISS, then de-duplicated to unique text IDs.

Key points:
- Loads sentence-level gold labels from ``fewnerd_entities.json``.
- Loads entity embeddings from ``fewnerd_embeddings.pth`` (see
  ``forward_llm_mlp_fewnerd.py``).
- Generates a query embedding per fine type (using ``embedding_utils``),
  or loads types from a provided JSON list.
- For each fine type ``t``, retrieves top ``k`` unique text IDs where
  ``k`` equals the number of relevant text IDs for ``t``. Because the
  index contains multiple vectors per text, we search up to ``4*k`` and
  then de-duplicate by text ID.

Example:
    python fewnerd_r_precision_prediction.py \
        --entities fewnerd_entities.json \
        --embeddings fewnerd_embeddings.pth \
        --fine_types fewnerd_fine_types.json

Outputs a compact table with per-type R-precision and overall macro
average.
"""

from __future__ import annotations

import argparse
import os
import json
from typing import Iterable, Set, Dict, Sequence, List, Tuple

import numpy as np
import torch

import faiss

from embedding_utils import load_llm_and_mlp, embed_entities_dataset
from tqdm.auto import tqdm as _tqdm


def _to_float32(x: Iterable[float] | np.ndarray) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x
    else:
        arr = np.asarray(list(x))
    return arr.astype("float32", copy=False)


def _l2_normalize(mat: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + eps
    return mat / norms


def _build_fine_type_to_ids(
    entities: List[dict],
    allowed_fine_types: Set[str] | None = None,
) -> Dict[str, Set[str]]:
    """Map each fine type to the set of sentence IDs containing it in gold.

    The entities JSON contains records with fields:
      - id: str (sentence UUID)
      - gold: list of {text, start, end, label}
    We treat ``label`` verbatim as the fine type string.
    """
    mapping: Dict[str, Set[str]] = {}
    for rec in _tqdm(entities, desc="Building fine_type->ids: records", dynamic_ncols=True):
        tid = str(rec["id"])  # sentence/text id
        for g in rec.get("gold", []):
            ft = g.get("label")
            if not isinstance(ft, str):
                continue
            if allowed_fine_types is not None and ft not in allowed_fine_types:
                continue
            if ft not in mapping:
                mapping[ft] = set()
            mapping[ft].add(tid)
    # Ensure all allowed types appear (possibly empty)
    if allowed_fine_types is not None:
        for ft in allowed_fine_types:
            mapping.setdefault(ft, set())
    return mapping


def _flatten_embeddings(embeds: Dict[str, Sequence[Sequence[float]]]) -> Tuple[np.ndarray, List[str]]:
    """Flatten mapping text_id -> list[embedding] into matrix and id map.

    Returns:
      - matrix: float32 array of shape (N_vectors, dim)
      - id_map: list of text_id per row in matrix
    """
    rows: List[np.ndarray] = []
    id_map: List[str] = []
    dim: int | None = None

    for tid, vecs in _tqdm(list(embeds.items()), desc="Flatten embeddings: texts", dynamic_ncols=True):
        for v in vecs:
            arr = _to_float32(v)
            if dim is None:
                dim = arr.shape[0]
            elif arr.shape[0] != dim:
                raise ValueError(f"Inconsistent embedding dim: got {arr.shape[0]}, expected {dim}")
            rows.append(arr)
            id_map.append(tid)

    if not rows:
        raise ValueError("No embeddings found in the provided .pth file")

    mat = np.stack(rows, axis=0)
    return mat, id_map



def compute_r_precision(
    embeddings_by_tid: Dict[str, Sequence[Sequence[float]]],
    fine_type_texts: Dict[str, str],
    fine_type_to_ids: Dict[str, Set[str]],
) -> Dict[str, float]:
    """Compute R-precision per fine type.

    - Builds an index over all entity embeddings (multiple per text ID).
    - Embeds each fine type text into a query vector.
    - For each fine type f with relevant set size k, retrieves up to 4*k
      vectors, de-duplicates to unique text IDs, takes the top-k unique IDs
      and computes |hits|/k.
    Returns mapping fine_type -> R-precision.
    """
    # Build base matrix and id map
    base_mat, id_map = _flatten_embeddings(embeddings_by_tid)
    base_mat = _l2_normalize(base_mat)

    dim = base_mat.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(base_mat)

    # Embed fine type texts using the same path as dataset embeddings
    print("embedding fine types")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model, mlp = load_llm_and_mlp(device)

    sentences = [
        {"id": k, "sentence": v, "predicted": [{"start": 0, "end": len(v)}]}
        for k, v in fine_type_texts.items()
    ]
    records = embed_entities_dataset(
        sentences=sentences,
        tokenizer=tokenizer,
        model=model,
        mlp=mlp,
        device=device,
        batch_size=32,
    )
    # Convert to normalized numpy in fixed order
    fine_types = list(fine_type_texts.keys())
    query_mat = np.stack([_to_float32(records[ft][0]) for ft in fine_types], axis=0)
    query_mat = _l2_normalize(query_mat)

    # Perform a single batched search with conservative top-k (max 4*k)
    k_per_type = [max(1, len(fine_type_to_ids.get(ft, set()))) for ft in fine_types]
    max_k = max(k_per_type) if k_per_type else 1
    D, I = index.search(query_mat, 4*max_k)

    # Compute per-type R-precision with de-dup by text_id
    r_precisions: Dict[str, float] = {}
    for row, ft in enumerate(_tqdm(fine_types, desc="Compute R-precision per type", dynamic_ncols=True)):
        relevant = fine_type_to_ids.get(ft, set())
        k = max(1, len(relevant))

        # Deduplicate by text_id while preserving order
        seen: Set[str] = set()
        ranking: List[str] = []
        limit = min(I.shape[1], 4 * k)
        for col in range(limit):
            idx = int(I[row, col])
            if idx < 0 or idx >= len(id_map):
                continue
            tid = id_map[idx]
            if tid not in seen:
                seen.add(tid)
                ranking.append(tid)
                if len(ranking) >= k:
                    break

        retrieved_k = ranking[:k]
        hits = len(set(retrieved_k) & relevant)
        r_precisions[ft] = hits / float(k) if k else 0.0

    return r_precisions


def _load_fine_types(path: str | None, entities: List[dict], name_map_path: str | None = None) -> Dict[str, str]:
    """Load list of fine types and return mapping type->text to embed.

    If ``path`` is provided, it should contain a JSON array of string labels
    (e.g., ``demo/fewnerd_fine_types.json``). Otherwise, derives the set of
    labels from the entities file.
    """
    if path:
        with open(path, "r", encoding="utf8") as f:
            types_list = json.load(f)
        if not isinstance(types_list, list):
            raise ValueError("fine_types file must be a JSON list of strings")
        fine_types = [str(x) for x in types_list]
    else:
        # Derive from entities' gold labels
        s: Set[str] = set()
        for rec in _tqdm(entities, desc="Deriving fine types: records", dynamic_ncols=True):
            for g in rec.get("gold", []):
                lab = g.get("label")
                if isinstance(lab, str):
                    s.add(lab)
        fine_types = sorted(s)

    # Map each fine type to a human-readable query text using the mapping JSON.
    mapping_path = name_map_path
    if not mapping_path:
        here = os.path.dirname(__file__)
        mapping_path = os.path.join(here, "entities_to_names.json")
    try:
        with open(mapping_path, "r", encoding="utf8") as f:
            name_map: Dict[str, str] = json.load(f)
    except FileNotFoundError:
        # Fallback: use the label as-is if mapping is unavailable
        name_map = {}

    return {ft: name_map.get(ft.split("-")[-1], ft) for ft in fine_types}


def main() -> None:
    parser = argparse.ArgumentParser(description="R-Precision for FewNERD fine types (FAISS retrieval)")
    parser.add_argument("--entities", default="fewnerd_entities.json", help="Path to FewNERD sentence JSON")
    parser.add_argument("--embeddings", default="fewnerd_embeddings.pth", help="Path to entity embeddings .pth")
    parser.add_argument("--type-name-map", dest="type_name_map", default=None,
                        help="Path to JSON mapping fine_type -> readable name (defaults to demo/entities_to_names.json)")
    args = parser.parse_args()

    # Load inputs
    with open(args.entities, "r", encoding="utf8") as f:
        entities = json.load(f)

    # Filter out sentences with fewer than 5 words
    def _word_count_ok(text: str, min_words: int = 5) -> bool:
        return len(str(text).split()) >= min_words

    entities = [rec for rec in entities if _word_count_ok(rec.get("sentence", ""))]

    embeds = torch.load(args.embeddings)
    if not isinstance(embeds, dict):
        raise ValueError("Embeddings file must be a dict[text_id] -> List[vector]")

    # Keep only embeddings for sentences that passed the word-count filter
    allowed_ids = {str(rec["id"]) for rec in entities}
    embeds = {tid: vecs for tid, vecs in embeds.items() if str(tid) in allowed_ids}

    fine_type_texts = _load_fine_types(path=None, entities=entities, name_map_path=args.type_name_map)
    fine_type_set = set(fine_type_texts.keys())
    fine_type_to_ids = _build_fine_type_to_ids(entities, allowed_fine_types=fine_type_set)

    # Compute R-precision per fine type
    rp = compute_r_precision(embeds, fine_type_texts, fine_type_to_ids)

    # Report
    rows: List[Tuple[str, int, float]] = []
    for ft in _tqdm(sorted(fine_type_texts.keys()), desc="Collecting per-type results", dynamic_ncols=True):
        size = len(fine_type_to_ids.get(ft, set()))
        rows.append((ft, size, rp.get(ft, 0.0)))

    # Print compact table
    header = f"{'fine_type':40s}  {'size':>6s}  {'R-precision':>11s}"
    print(header)
    print("-" * len(header))
    for ft, size, r in _tqdm(rows, desc="Printing rows", dynamic_ncols=True, leave=False):
        print(f"{ft:40.40s}  {size:6d}  {r:11.4f}")

    # Macro average over all non-empty fine types, excluding readable name 'Other' (case-insensitive)
    def _is_other(ft: str) -> bool:
        name = str(fine_type_texts.get(ft, ""))
        return name.strip().lower() == "other"

    filtered = [r for ft, size, r in rows if size > 0 and not _is_other(ft)]
    macro = float(np.mean(filtered)) if filtered else 0.0
    print("\nMacro R-precision (non-empty types): {:.4f}".format(macro))


if __name__ == "__main__":
    main()
