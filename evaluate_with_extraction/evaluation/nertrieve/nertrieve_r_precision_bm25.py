import json
import os
from collections import defaultdict
from typing import Dict, List

import Stemmer
import pandas as pd
from clearml import Dataset
import bm25s
from tqdm import tqdm

import clearml_poc
from clearml_pipelines.nertreieve_dataset import nertrieve_processor

DATASET_PROJECT = "nertrieve_pipeline"
DATASET_NAME = "nertrieve_test_ir_base_combined.json"


def _load_dataset(name: str) -> str:
    ds = Dataset.get(dataset_name=name, dataset_project=DATASET_PROJECT)
    return os.path.join(ds.get_local_copy(), name)


class NertrieveRPrecisionBM25:
    def __init__(self) -> None:
        print("Loading metadata...")
        self.metadata = self._load_metadata()
        print("Metadata loaded.")
        self.type_to_name = nertrieve_processor.type_to_name()
        print("Calculating fine type to IDs mapping...")
        self.fine_type_to_ids = self._calc_fine_type_to_ids()
        print("Fine type to IDs mapping calculated.")
        print("Preparing BM25 corpus...")
        self.corpus_tokens, self.text_ids = self._prepare_corpus()
        self.bm25 = bm25s.BM25()
        self.bm25.index(self.corpus_tokens)
        print("BM25 corpus ready.")
        self.fine_types = list(self.fine_type_to_ids.keys())

    def _load_metadata(self) -> Dict[str, Dict]:
        path = _load_dataset(DATASET_NAME)
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _calc_fine_type_to_ids(self) -> Dict[str, set]:
        mapping: Dict[str, set] = defaultdict(set)
        for tid, record in self.metadata.items():
            for g in record.get("gold", []):
                mapping[g["fine_type"]].add(tid)
        return mapping

    def _prepare_corpus(self) -> tuple[List[List[str]], List[str]]:
        corpus: List[str] = []
        text_ids: List[str] = []
        for tid, record in self.metadata.items():
            corpus.append(record["sentence"])
            text_ids.append(tid)
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
        return corpus_tokens, text_ids

    def evaluate(self) -> pd.DataFrame:
        rows = {}
        for fine_type in tqdm(self.fine_types):
            query_text = self.type_to_name.get(fine_type, fine_type)
            query_tokens = bm25s.tokenize(query_text, stopwords="en", stemmer=stemmer, show_progress=False)
            relevant = self.fine_type_to_ids[fine_type]
            count_type = len(relevant)
            max_k = max(500, count_type)
            results, _scores = self.bm25.retrieve(query_tokens=query_tokens, k=max_k, show_progress=False)
            ranking = [self.text_ids[idx] for idx in results[0]]

            row = {"size": count_type}
            sizes = [10, 50, 100, 200, 500, count_type]
            desc = ["10", "50", "100", "200", "500", "size"]
            for s, d in zip(sizes, desc):
                k = min(s, count_type)
                retrieved_k = ranking[:k]
                hits = len(set(retrieved_k) & relevant)
                recall = hits / count_type if count_type else 0.0
                precision = hits / k if k else 0.0
                row[f"recall@{d}"] = recall
                row[f"precision@{d}"] = precision
                if d == "size":
                    r_prec = precision

            row["R-precision"] = r_prec
            rows[fine_type] = row

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
        return df


def main() -> None:
    clearml_poc.clearml_init(task_name="NERtrieve R-Precision BM25 Evaluation", project_name=DATASET_PROJECT)
    evaluator = NertrieveRPrecisionBM25()
    evaluator.evaluate()


if __name__ == "__main__":
    stemmer = Stemmer.Stemmer("english")
    main()
