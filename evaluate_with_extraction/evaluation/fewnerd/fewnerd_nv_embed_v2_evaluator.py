import json
import os
from collections import defaultdict
from typing import Dict, Set

import torch
from clearml import Dataset
from tqdm import tqdm

import clearml_poc
from contrastive import fewnerd_processor
from sentence_embedder import SentenceEmbedder
from evaluate_with_extraction.evaluation.single_vector_r_precision import SingleVectorRPrecision

SENTENCE_EMBEDDER_ID = "nvidia/NV-Embed-v2"


def _load_dataset(name: str) -> str:
    ds = Dataset.get(dataset_name=name, dataset_project="fewnerd_pipeline")
    return os.path.join(ds.get_local_copy(), name)


def load_embeddings() -> Dict[str, torch.Tensor]:
    path = _load_dataset("sentence_embeddings_nv.pth")
    data = torch.load(path)
    result: Dict[str, torch.Tensor] = {}
    for tid, emb in tqdm(data.items(), desc="Loading embeddings"):
        result[tid] = torch.tensor(emb, dtype=torch.float)
    return result


def load_metadata() -> Dict[str, Dict]:
    path = _load_dataset("span_extraction_results.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def calc_fine_type_to_ids(metadata: Dict[str, Dict]) -> Dict[str, Set[str]]:
    mapping: Dict[str, Set[str]] = defaultdict(set)
    for tid, record in metadata.items():
        for g in record.get("gold", []):
            mapping[g["fine_type"]].add(tid)
    return mapping


def embed_fine_types(fine_type_to_ids: Dict[str, Set[str]]) -> Dict[str, torch.Tensor]:
    embedder = SentenceEmbedder(llm_id=SENTENCE_EMBEDDER_ID)
    type_to_name = fewnerd_processor.type_to_name()
    result = {}
    for fine_type in fine_type_to_ids.keys():
        readable = type_to_name[fine_type.split("-")[-1]]
        emb = embedder.forward_query(readable)[0].cpu()
        result[fine_type] = emb
    return result


def main() -> None:
    clearml_poc.clearml_init(
        task_name="FewNERD Sentence R-Precision Evaluation " + SENTENCE_EMBEDDER_ID,
        project_name="fewnerd_pipeline",
        requirements=["transformers==4.46.2", "sentence_transformers", "accelerate", "einops"],
    )
    metadata = load_metadata()
    embeddings = load_embeddings()
    ft_to_ids = calc_fine_type_to_ids(metadata)
    ft_embeds = embed_fine_types(ft_to_ids)
    evaluator = SingleVectorRPrecision(
        embeddings=embeddings,
        fine_type_embeddings=ft_embeds,
        fine_type_to_ids=ft_to_ids,
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()
