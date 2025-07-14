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

from evaluate_with_extraction.evaluation.multi_vecor_r_precision import MultiVecorRPrecision

MLP_ID = "137515e67cb14851b85d55846e630337"


def _load_dataset(name: str) -> str:
    ds = Dataset.get(dataset_name=name, dataset_project="fewnerd_pipeline")
    return os.path.join(ds.get_local_copy(), name)


def load_embeddings() -> Dict[str, Dict[str, List[torch.Tensor]]]:
    path = _load_dataset("ablation_embeddings.pt")
    data = torch.load(path)
    result: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    for tid, embs in data.items():
        for key, lst in embs.items():
            result.setdefault(key, {})[tid] = [torch.tensor(e, dtype=torch.float) for e in lst]
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


def embed_fine_types(fine_type_to_ids: Dict[str, Set[str]], embedding_key: str) -> Dict[str, torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm = LLMInterface(llm_id=FineTuneLLM.llm_id, max_llm_layer=FineTuneLLM.max_llm_layer)
    layer = FineTuneLLM.layer
    args: Arguments = clearml_helper.get_args_by_mlp_id(MLP_ID)
    type_to_name = fewnerd_processor.type_to_name()
    mlp = None
    if embedding_key.startswith("mlp_"):
        mlp = clearml_helper.get_mlp_by_id(MLP_ID, device=device).float()
        mlp.eval()
    result = {}
    eos_token = llm.tokenizer.eos_token
    for fine_type in fine_type_to_ids.keys():
        readable = type_to_name[fine_type.split("-")[-1]]
        text_with_eos = readable + eos_token
        tokens = llm.tokenize(text_with_eos).to(device)
        with torch.no_grad():
            hidden = llm.get_llm_at_layer(tokens, layer=layer)
        eos_idx = llm.tokens_count(text_with_eos) - 1
        eos_vec = hidden[0, eos_idx]
        end_idx = llm.tokens_count(readable) - 1
        start_vec = hidden[0, 0]
        end_vec = hidden[0, end_idx]
        rep = fewnerd_processor.choose_llm_representation(
            end_vec.cpu().tolist(),
            start_vec.cpu().tolist(),
            input_tokens=args.input_tokens,
        )
        if embedding_key.startswith("mlp_"):
            inp = torch.cat((rep.to(device), eos_vec), dim=-1)
            with torch.no_grad():
                emb = mlp(inp).cpu()
        elif embedding_key.endswith("_eos"):
            emb = eos_vec.cpu()
        else:
            emb = rep.cpu()
        result[fine_type] = emb
    return result


def main() -> None:
    clearml_poc.clearml_init(task_name="FewNERD Ablation R-Precision Evaluation", project_name="fewnerd_pipeline")
    metadata = load_metadata()
    all_embeddings = load_embeddings()
    ft_to_ids = calc_fine_type_to_ids(metadata)
    for key, embeddings in all_embeddings.items():
        ft_embeds = embed_fine_types(ft_to_ids, key)
        evaluator = MultiVecorRPrecision(
            embeddings=embeddings,
            fine_type_embeddings=ft_embeds,
            fine_type_to_ids=ft_to_ids,
        )
        clearml_poc.add_text(key)
        evaluator.evaluate(description=key)


if __name__ == "__main__":
    main()
