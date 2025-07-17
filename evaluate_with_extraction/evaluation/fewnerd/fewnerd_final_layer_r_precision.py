import json
import os
from collections import defaultdict
from typing import Dict, List, Set

import torch
from clearml import Dataset
from tqdm import tqdm

import clearml_poc
from llm_interface import LLMInterface
from contrastive import fewnerd_processor
from contrastive.args import Arguments, FineTuneLLM

from evaluate_with_extraction.evaluation.multi_vecor_r_precision import MultiVecorRPrecision


def _load_dataset(name: str) -> str:
    ds = Dataset.get(dataset_name=name, dataset_project="fewnerd_pipeline")
    return os.path.join(ds.get_local_copy(), name)


def load_embeddings(valid_ids: Set[str]) -> Dict[str, List[torch.Tensor]]:
    path = _load_dataset("final_layer_embeddings.pt")
    data = torch.load(path)
    result: Dict[str, List[torch.Tensor]] = {}
    for tid, embs in data.items():
        if tid not in valid_ids:
            continue
        result[tid] = [torch.tensor(e, dtype=torch.float) for e in embs.get("llama_3_entire_model_entity_end", [])]
    return result


def load_metadata() -> Dict[str, Dict]:
    path = _load_dataset("span_extraction_results.json")
    with open(path, "r", encoding="utf-8") as fh:
        data: Dict[str, Dict] = json.load(fh)
    return {
        tid: rec
        for tid, rec in data.items()
        if len(rec.get("sentence", "").split()) > 4
    }


def calc_fine_type_to_ids(metadata: Dict[str, Dict]) -> Dict[str, Set[str]]:
    mapping: Dict[str, Set[str]] = defaultdict(set)
    for tid, record in metadata.items():
        for g in record.get("gold", []):
            mapping[g["fine_type"]].add(tid)
    return mapping


def embed_fine_types(fine_type_to_ids: Dict[str, Set[str]]) -> Dict[str, torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = "model"
    llm = LLMInterface(llm_id=FineTuneLLM.llm_id, layer=layer)
    args = Arguments()
    type_to_name = fewnerd_processor.type_to_name()
    result = {}
    for fine_type in fine_type_to_ids.keys():
        readable = type_to_name[fine_type.split("-")[-1]]
        tokens = llm.tokenize(readable).to(device)
        with torch.no_grad():
            hidden = llm.get_llm_at_layer(tokens, layer=layer)
        start = hidden[0, 0]
        end = hidden[0, -1]
        rep = fewnerd_processor.choose_llm_representation(
            end.cpu().tolist(),
            start.cpu().tolist(),
            input_tokens=args.input_tokens,
        )
        result[fine_type] = rep.cpu()
    return result


def main() -> None:
    clearml_poc.clearml_init(task_name="FewNERD Final Layer R-Precision Evaluation", project_name="fewnerd_pipeline")
    metadata = load_metadata()
    embeddings = load_embeddings(set(metadata.keys()))
    ft_to_ids = calc_fine_type_to_ids(metadata)
    ft_embeds = embed_fine_types(ft_to_ids)
    evaluator = MultiVecorRPrecision(
        embeddings=embeddings,
        fine_type_embeddings=ft_embeds,
        fine_type_to_ids=ft_to_ids,
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()
