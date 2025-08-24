import json
import os
from collections import defaultdict
from typing import Dict, List, Set

import torch
from clearml import Dataset
from tqdm import tqdm

import clearml_poc
import clearml_helper
from llm_interface import LLMInterface
from contrastive import fewnerd_processor
from contrastive.args import Arguments, FineTuneLLM
from clearml_pipelines.nertreieve_dataset import nertrieve_processor
from evaluate_with_extraction.evaluation.multi_vecor_r_precision import MultiVecorRPrecision

DATASET_PROJECT = "nertrieve_pipeline"
DATASET_NAME = "nertrieve_test_ir_base_combined.json"


def _load_dataset(name: str) -> str:
    ds = Dataset.get(dataset_name=name, dataset_project=DATASET_PROJECT)
    return os.path.join(ds.get_local_copy(), name)


def load_embeddings() -> Dict[str, List[torch.Tensor]]:
    path = _load_dataset("llm_mlp_embeddings_gold.pth")
    data = torch.load(path)
    result: Dict[str, List[torch.Tensor]] = {}
    for tid, emb_list in tqdm(data.items(), desc="Loading embeddings"):
        result[tid] = [torch.tensor(e, dtype=torch.float) for e in emb_list]
    return result


def load_metadata() -> Dict[str, Dict]:
    path = _load_dataset(DATASET_NAME)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def calc_fine_type_to_ids(metadata: Dict[str, Dict]) -> Dict[str, Set[str]]:
    mapping: Dict[str, Set[str]] = defaultdict(set)
    for tid, record in metadata.items():
        for g in record.get("gold", []):
            mapping[g["fine_type"]].add(tid)
    return mapping


def embed_fine_types(fine_type_to_ids: Dict[str, Set[str]], mlp_id: str) -> Dict[str, torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp = clearml_helper.get_mlp_by_id(mlp_id, device=device)
    args: Arguments = clearml_helper.get_args_by_mlp_id(mlp_id)
    llm = LLMInterface(llm_id=FineTuneLLM.llm_id, max_llm_layer=FineTuneLLM.max_llm_layer)
    layer = FineTuneLLM.layer
    name_map = nertrieve_processor.type_to_name()

    result = {}
    for fine_type in fine_type_to_ids.keys():
        readable = name_map.get(fine_type, fine_type)
        tokens = llm.tokenize(readable).to(device)
        with torch.no_grad():
            hidden = llm.get_llm_at_layer(tokens, layer=layer)
        start = hidden[0, 0]
        end = hidden[0, -1]
        rep = fewnerd_processor.choose_llm_representation(
            end=end.cpu().tolist(), start=start.cpu().tolist(), input_tokens=args.input_tokens
        )
        with torch.no_grad():
            emb = mlp(rep.to(device)).cpu()
        result[fine_type] = emb
    return result


def main() -> None:
    clearml_poc.clearml_init(task_name="NERtrieve R-Precision GOLD Evaluation", project_name=DATASET_PROJECT)
    mlp_id = FineTuneLLM.mlp_head_model_id_from_clearml
    metadata = load_metadata()
    embeddings = load_embeddings()
    ft_to_ids = calc_fine_type_to_ids(metadata)
    ft_embeds = embed_fine_types(ft_to_ids, mlp_id)
    evaluator = MultiVecorRPrecision(
        embeddings=embeddings,
        fine_type_embeddings=ft_embeds,
        fine_type_to_ids=ft_to_ids,
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()
