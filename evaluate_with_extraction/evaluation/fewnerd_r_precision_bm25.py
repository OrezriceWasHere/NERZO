import json
import os
from collections import defaultdict
from typing import Dict, List

import pandas as pd
from clearml import Dataset
import bm25s

import clearml_poc
from contrastive import fewnerd_processor


class FewNerdRPrecisionBM25:
    """Evaluate FewNERD retrieval using BM25 and R-precision."""

    def __init__(self) -> None:
        print("Loading metadata...")
        self.metadata = self._load_metadata()
        print("Metadata loaded.")
        self.type_to_name = fewnerd_processor.type_to_name()
        print("Calculating fine type to IDs mapping...")
        self.fine_type_to_ids = self._calc_fine_type_to_ids()
        print("Fine type to IDs mapping calculated.")
        print("Preparing BM25 corpus...")
        self.corpus_tokens, self.text_ids = self._prepare_corpus()
        self.bm25 = bm25s.BM25()
        self.bm25.index(self.corpus_tokens)
        print("BM25 corpus ready.")
        self.fine_types = list(self.fine_type_to_ids.keys())

    @staticmethod
    def _load_dataset(name: str) -> str:
        ds = Dataset.get(dataset_name=name, dataset_project="fewnerd_pipeline")
        return os.path.join(ds.get_local_copy(), name)

    def _load_metadata(self) -> Dict[str, Dict]:
        path = self._load_dataset("span_extraction_results.json")
        with open(path, "r", encoding="utf-8") as fh:
            result = json.load(fh)
        return result

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
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en")
        return corpus_tokens, text_ids

    def evaluate(self) -> pd.DataFrame:
        rows = {}
        for ft in self.fine_types:
            query_text = self.type_to_name[ft.split("-")[-1]]
            query_tokens = bm25s.tokenize(query_text, stopwords="en")
            results, _scores = self.bm25.retrieve(query_tokens=query_tokens, k=len(self.text_ids))
            ranking = [self.text_ids[idx] for idx in results[0]]
            relevant = self.fine_type_to_ids[ft]
            r_size = len(relevant)
            retrieved = ranking[:r_size]
            r_prec = len(set(retrieved) & relevant) / r_size if r_size else 0.0
            rows[ft] = {"R-precision": r_prec, "size": r_size}
        df = pd.DataFrame.from_dict(rows, orient="index")
        clearml_poc.add_table(title="R-precision per fine type", series="r_precision", iteration=0, table=df)
        clearml_poc.add_table(title="average R-precision", series="r_precision", iteration=0, table=df.mean().to_frame())
        return df


def main() -> None:
    clearml_poc.clearml_init(task_name="FewNERD R-Precision BM25 Evaluation", project_name="fewnerd_pipeline")
    evaluator = FewNerdRPrecisionBM25()
    evaluator.evaluate()


if __name__ == "__main__":
    main()
