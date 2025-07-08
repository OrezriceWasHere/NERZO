import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

import torch
import pandas as pd
from clearml import Dataset
import faiss

import clearml_poc
import clearml_helper
from llm_interface import LLMInterface
from contrastive import fewnerd_processor
from contrastive.args import Arguments, FineTuneLLM


class FewNerdRPrecision:
    """Evaluate FewNERD retrieval using R-precision."""

    def __init__(self, mlp_id: str):
        print("Starting initialization of FewNerdRPrecision...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mlp = clearml_helper.get_mlp_by_id(mlp_id, device=self.device)
        self.args: Arguments = clearml_helper.get_args_by_mlp_id(mlp_id)
        print("Initializing LLM...")
        self.llm = LLMInterface(llm_id=FineTuneLLM.llm_id,
                                max_llm_layer=FineTuneLLM.max_llm_layer)
        print("LLM  initialized.")
        self.layer = FineTuneLLM.layer
        self.type_to_name = fewnerd_processor.type_to_name()
        print("Loading metadata...")
        self.metadata = self._load_metadata()
        print("Metadata loaded.")
        print("Loading embeddings...")
        self.embeddings = self._load_embeddings()
        print("Embeddings loaded.")
        print("Calculating fine type to IDs mapping...")
        self.fine_type_to_ids = self._calc_fine_type_to_ids()
        print("Fine type to IDs mapping calculated.")
        print("Embedding fine types...")
        self.fine_type_embeddings = self._embed_fine_types()
        print("Fine types embedded.")
        print("Building Faiss index...")
        self.index, self.index_to_tid = self._build_index()
        print("Faiss index built.")
        self.fine_types = list(self.fine_type_embeddings.keys())
        print("Initialization of FewNerdRPrecision finished.")

    @staticmethod
    def _load_dataset(name: str) -> str:
        print(f"Loading dataset '{name}'...")
        ds = Dataset.get(dataset_name=name, dataset_project="fewnerd_pipeline")
        print(f"Dataset '{name}' loaded.")
        return os.path.join(ds.get_local_copy(), name)

    def _load_embeddings(self) -> Dict[str, List[torch.Tensor]]:
        print("Reading embeddings from file...")
        path = self._load_dataset("llm_mlp_embeddings.json")
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        print("Embeddings file read.")
        return {tid: [torch.tensor(e) for e in emb_list] for tid, emb_list in data.items()}

    def _load_metadata(self) -> Dict[str, Dict]:
        print("Reading metadata from file...")
        path = self._load_dataset("span_extraction_results.json")
        with open(path, "r", encoding="utf-8") as fh:
            result = json.load(fh)
        print("Metadata file read.")
        return result

    def _calc_fine_type_to_ids(self) -> Dict[str, set]:
        print("Calculating mapping from fine types to IDs...")
        mapping: Dict[str, set] = defaultdict(set)
        for tid, record in self.metadata.items():
            for g in record.get("gold", []):
                fine_type = g["fine_type"]
                mapping[fine_type].add(tid)
        print("Mapping from fine types to IDs calculated.")
        return mapping

    def _embed_fine_types(self) -> Dict[str, torch.Tensor]:
        print("Embedding all fine types...")
        result = {}
        for fine_type in self.fine_type_to_ids.keys():
            readable = self.type_to_name[fine_type.split("-")[-1]]
            tokens = self.llm.tokenize(readable).to(self.device)
            with torch.no_grad():
                hidden = self.llm.get_llm_at_layer(tokens, layer=self.layer)
            start = hidden[0, 0]
            end = hidden[0, -1]
            rep = fewnerd_processor.choose_llm_representation(
                end=end.cpu().tolist(), start=start.cpu().tolist(),
                input_tokens=self.args.input_tokens)
            with torch.no_grad():
                emb = self.mlp(rep.to(self.device)).cpu()
            result[fine_type] = emb
        print("All fine types embedded.")
        return result

    def _build_index(self):
        print("Building FAISS index from embeddings...")
        all_vecs = []
        id_map: List[str] = []
        for tid, embs in self.embeddings.items():
            for emb in embs:
                vec = emb.numpy().astype('float32')
                norm = np.linalg.norm(vec) + 1e-10
                all_vecs.append(vec / norm)
                id_map.append(tid)
        if not all_vecs:
            raise ValueError("No embeddings loaded")
        dim = all_vecs[0].shape[0]
        index = faiss.IndexFlatIP(dim)
        index.add(np.stack(all_vecs))
        print("FAISS index built.")
        return index, id_map

    def _search_all(self):
        """Search the index for all fine-type embeddings at once."""
        print("Preparing queries for FAISS search...")
        query_vecs = []
        for ft in self.fine_types:
            vec = self.fine_type_embeddings[ft].numpy().astype("float32")
            norm = np.linalg.norm(vec) + 1e-10
            query_vecs.append(vec / norm)
        queries = np.stack(query_vecs)
        print("Starting Faiss search for all fine types...")
        D, I = self.index.search(queries, 4*len(self.index_to_tid))
        print("Finished Faiss search.")
        return D, I

    def evaluate(self) -> pd.DataFrame:
        print("Starting evaluation (R-precision calculation)...")
        rows = {}
        D, I = self._search_all()
        for idx_ft, ft in enumerate(self.fine_types):
            relevant = self.fine_type_to_ids[ft]
            retrieved = []
            seen = set()
            for idx in I[idx_ft]:
                tid = self.index_to_tid[idx]
                if tid not in seen:
                    retrieved.append(tid)
                    seen.add(tid)
                    if len(retrieved) == len(relevant):
                        break
            r_prec = len(set(retrieved) & relevant) / len(relevant) if relevant else 0.0
            rows[ft] = {"R-precision": r_prec, "size": len(relevant)}
        df = pd.DataFrame.from_dict(rows, orient="index")
        print("Evaluation finished. Logging results to ClearML...")
        clearml_poc.add_table(title="R-precision per fine type", series="r_precision", iteration=0, table=df)
        clearml_poc.add_table(title="average R-precision", series="r_precision", iteration=0, table=df.mean().to_frame())
        print("Results logged.")
        return df


def main():
    print("Initializing ClearML task...")
    clearml_poc.clearml_init(task_name="FewNERD R-Precision Evaluation", project_name="fewnerd_pipeline")
    mlp_id = FineTuneLLM.mlp_head_model_id_from_clearml
    print("Creating evaluator...")
    evaluator = FewNerdRPrecision(mlp_id)
    print("Running evaluation...")
    evaluator.evaluate()
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
