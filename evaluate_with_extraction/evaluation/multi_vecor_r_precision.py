import numpy as np
import pandas as pd
import faiss
from typing import Dict, Iterable, Sequence, Set, Union

import clearml_poc

Vector = Union[Sequence[float], np.ndarray]


def _to_numpy(vec: Vector) -> np.ndarray:
    if isinstance(vec, np.ndarray):
        return vec.astype("float32")
    return np.asarray(vec, dtype="float32")


class MultiVecorRPrecision:
    """Generic R-precision evaluator for multiple query vectors."""

    def __init__(
        self,
        embeddings: Dict[str, Iterable[Vector]],
        fine_type_embeddings: Dict[str, Vector],
        fine_type_to_ids: Dict[str, Set[str]],
    ) -> None:
        self.embeddings = {tid: [
            _to_numpy(e) for e in emb_list
        ] for tid, emb_list in embeddings.items()}
        self.fine_type_embeddings = {
            ft: _to_numpy(e) for ft, e in fine_type_embeddings.items()
        }
        self.fine_type_to_ids = fine_type_to_ids
        self.fine_types = list(self.fine_type_embeddings.keys())
        self.index, self.index_to_tid = self._build_index()

    def _build_index(self):
        all_vecs = []
        id_map = []
        for tid, embs in self.embeddings.items():
            for emb in embs:
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
            relevant = self.fine_type_to_ids.get(ft, set())
            ranking = []
            seen = set()
            max_k = max(500, len(relevant))
            for idx in I[idx_ft]:
                tid = self.index_to_tid[idx]
                if tid not in seen:
                    ranking.append(tid)
                    seen.add(tid)
                    if len(ranking) >= max_k:
                        break

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
        clearml_poc.add_table(title="R-precision per fine type", series="r_precision", iteration=0, table=df)
        clearml_poc.add_table(title="average R-precision", series="r_precision", iteration=0, table=df.mean().to_frame())
        return df
