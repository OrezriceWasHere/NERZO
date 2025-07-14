import numpy as np
import pandas as pd
import faiss
from typing import Dict, Sequence, Set, Union

import clearml_poc

OTHER_MULTICONER = {"otherLOC", "otherper", "otherprod"}


def _is_other(fine_type: str) -> bool:
    """Return True if the fine type represents an "other" category."""
    return fine_type.endswith("-other") or fine_type.lower() in OTHER_MULTICONER

Vector = Union[Sequence[float], np.ndarray]


def _to_numpy(vec: Vector) -> np.ndarray:
    if isinstance(vec, np.ndarray):
        return vec.astype("float32")
    return np.asarray(vec, dtype="float32")


class SingleVectorRPrecision:
    """Generic R-precision evaluator for single query vectors."""

    def __init__(
        self,
        embeddings: Dict[str, Vector],
        fine_type_embeddings: Dict[str, Vector],
        fine_type_to_ids: Dict[str, Set[str]],
    ) -> None:
        self.embeddings = {tid: _to_numpy(e) for tid, e in embeddings.items()}
        self.fine_type_embeddings = {
            ft: _to_numpy(e) for ft, e in fine_type_embeddings.items()
        }
        self.fine_type_to_ids = fine_type_to_ids
        self.fine_types = list(self.fine_type_embeddings.keys())
        self.index, self.index_to_tid = self._build_index()

    def _build_index(self):
        all_vecs = []
        id_map = []
        for tid, emb in self.embeddings.items():
            vec = emb.astype("float32")
            norm = np.linalg.norm(vec) + 1e-10
            all_vecs.append(vec / norm)
            id_map.append(tid)
        if not all_vecs:
            raise ValueError("No embeddings loaded")
        dim = all_vecs[0].shape[0]
        index = faiss.IndexFlatIP(dim)
        index.add(np.stack(all_vecs))
        return index, id_map

    def _search_all(self):
        query_vecs = []
        for ft in self.fine_types:
            vec = self.fine_type_embeddings[ft].astype("float32")
            norm = np.linalg.norm(vec) + 1e-10
            query_vecs.append(vec / norm)
        queries = np.stack(query_vecs)
        D, I = self.index.search(queries, 4 * len(self.index_to_tid))
        return D, I

    def evaluate(self) -> pd.DataFrame:
        rows = {}
        D, I = self._search_all()
        for idx_ft, ft in enumerate(self.fine_types):
            relevant = self.fine_type_to_ids[ft]
            ranking = [self.index_to_tid[i] for i in I[idx_ft][:len(relevant)]]

            row = {"size": len(relevant)}
            sizes = [10, 50, 100, 200, 500, len(relevant)]
            desc = ["10", "50", "100", "200", "500", "size"]
            for s, d in zip(sizes, desc):
                k = min(s, len(relevant))
                retrieved_k = ranking[:k]
                hits = len(set(retrieved_k) & relevant)
                recall = hits / len(relevant) if len(relevant) else 0.0
                precision = hits / k if k else 0.0
                row[f"recall@{d}"] = recall
                row[f"precision@{d}"] = precision
                if d == "size":
                    r_prec = precision
            row["R-precision"] = r_prec
            rows[ft] = row

        df = pd.DataFrame.from_dict(rows, orient="index")
        clearml_poc.add_table(
            title="R-precision per fine type",
            series="r_precision",
            iteration=0,
            table=df,
        )

        clearml_poc.add_table(
            title="average R-precision",
            series="r_precision",
            iteration=0,
            table=df.mean().to_frame(),
        )

        non_other = df[~df.index.to_series().apply(_is_other)]
        if not non_other.empty:
            clearml_poc.add_table(
                title="average non-other R-precision",
                series="r_precision",
                iteration=0,
                table=non_other.mean().to_frame(),
            )
        return df
