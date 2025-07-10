import json
import os
from collections import defaultdict
from typing import Dict

import ijson
import numpy as np
import pandas as pd
import torch
from clearml import Dataset
import faiss
from tqdm import tqdm

import clearml_poc
from contrastive import fewnerd_processor
from sentence_embedder import SentenceEmbedder

OTHER_MULTICONER = {"OtherLOC", "OtherPER", "OtherPROD"}


def _is_other(fine_type: str) -> bool:
    """Return True if the fine type represents an "other" category."""
    return fine_type.endswith("-other") or fine_type in OTHER_MULTICONER


class FewNerdSentenceRPrecision:
    """Evaluate FewNERD retrieval using NV-Embed-v2 sentence embeddings."""

    def __init__(self) -> None:
        print("Initializing evaluator...")
        self.embedder = SentenceEmbedder(llm_id="nvidia/NV-Embed-v2")
        self.type_to_name = fewnerd_processor.type_to_name()
        self.metadata = self._load_metadata()
        self.embeddings = self._load_embeddings()
        self.fine_type_to_ids = self._calc_fine_type_to_ids()
        self.fine_type_embeddings = self._embed_fine_types()
        self.index, self.index_to_tid = self._build_index()
        self.fine_types = list(self.fine_type_embeddings.keys())
        print("Initialization done.")

    @staticmethod
    def _load_dataset(name: str) -> str:
        ds = Dataset.get(dataset_name=name, dataset_project="fewnerd_pipeline")
        return os.path.join(ds.get_local_copy(), name)

    def _load_metadata(self) -> Dict[str, Dict]:
        path = self._load_dataset("span_extraction_results.json")
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _load_embeddings(self) -> Dict[str, torch.Tensor]:
        path = self._load_dataset("sentence_embeddings_nv.json")
        result: Dict[str, torch.Tensor] = {}
        with open(path, "r", encoding="utf-8") as fh:
            for tid, emb in tqdm(
                ijson.kvitems(fh, ''), desc="Loading embeddings"
            ):
                result[tid] = torch.tensor(emb)
        return result

    def _calc_fine_type_to_ids(self) -> Dict[str, set]:
        mapping: Dict[str, set] = defaultdict(set)
        for tid, record in self.metadata.items():
            for g in record.get("gold", []):
                mapping[g["fine_type"]].add(tid)
        return mapping

    def _embed_fine_types(self) -> Dict[str, torch.Tensor]:
        result = {}
        for ft in self.fine_type_to_ids.keys():
            readable = self.type_to_name[ft.split("-")[-1]]
            emb = self.embedder.forward_query(readable)[0].cpu()
            result[ft] = emb
        return result

    def _build_index(self):
        all_vecs = []
        id_map = []
        for tid, emb in self.embeddings.items():
            vec = emb.numpy().astype("float32")
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
            vec = self.fine_type_embeddings[ft].numpy().astype("float32")
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


def main() -> None:
    clearml_poc.clearml_init(
        task_name="FewNERD Sentence R-Precision Evaluation",
        project_name="fewnerd_pipeline",
        requirements=["transformers==4.46.2", "sentence_transformers", "accelerate", "einops"],
    )
    evaluator = FewNerdSentenceRPrecision()
    evaluator.evaluate()


if __name__ == "__main__":
    main()
