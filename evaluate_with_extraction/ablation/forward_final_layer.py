import os
import json
from typing import Dict, List, Tuple

import torch
from clearml import Dataset
from tqdm import tqdm

import clearml_poc
from llm_interface import LLMInterface
from contrastive.args import Arguments, FineTuneLLM
from contrastive import fewnerd_processor

BATCH_SIZE = 10


def load_dataset() -> Dict[str, Dict]:
    ds = Dataset.get(dataset_name="span_extraction_results.json", dataset_project="fewnerd_pipeline")
    path = os.path.join(ds.get_local_copy(), "span_extraction_results.json")
    with open(path, "r", encoding="utf-8") as fh:
        data: Dict[str, Dict] = json.load(fh)
    return {
        tid: rec
        for tid, rec in data.items()
        if len(rec.get("sentence", "").split()) > 4
    }


def process_batch(
    batch: List[Tuple[str, Dict]],
    llm: LLMInterface,
    layer: str,
    args: Arguments,
    device: torch.device,
) -> Dict[str, Dict[str, List[List[float]]]]:
    sentences = [record["sentence"] for _, record in batch]
    tokens = llm.tokenize(sentences).to(device)

    with torch.no_grad():
        hidden = llm.get_llm_at_layer(tokens, layer)

    result: Dict[str, Dict[str, List[List[float]]]] = {}
    for (text_id, record), h in zip(batch, hidden):
        for ent in record.get("predicted", []):
            indices = (ent["start"], ent["end"])
            if indices[0] == indices[1]:
                continue
            tok_idx = llm.token_indices_given_text_indices(record["sentence"], indices)
            start = h[tok_idx[0] - 1]
            end = h[tok_idx[1]]
            rep = fewnerd_processor.choose_llm_representation(
                end.cpu().tolist(),
                start.cpu().tolist(),
                input_tokens=args.input_tokens,
            )
            result.setdefault(text_id, {}).setdefault("llama_3_entire_model_entity_end", []).append(rep.tolist())
    return result


def main() -> None:
    clearml_poc.clearml_init(
        task_name="FewNERD LLM Final Layer Forward", project_name="fewnerd_pipeline",
        requirements=["transformers==4.46.2", "accelerate"]
    )

    args = Arguments()
    clearml_poc.clearml_connect_hyperparams(args, "general")
    llm_args = FineTuneLLM()
    llm_args.layer = "model"
    clearml_poc.clearml_connect_hyperparams(llm_args, "llm_args")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm = LLMInterface(llm_id=llm_args.llm_id, max_llm_layer=llm_args.max_llm_layer)
    llm.model.eval()

    records = load_dataset()

    result: Dict[str, Dict[str, List[List[float]]]] = {}
    batch: List[Tuple[str, Dict]] = []

    for text_id, record in tqdm(records.items()):
        batch.append((text_id, record))
        if len(batch) >= BATCH_SIZE:
            batch_result = process_batch(batch, llm, llm_args.layer, args, device)
            for k, v in batch_result.items():
                if k in result:
                    for key_emb, arr in v.items():
                        result[k].setdefault(key_emb, []).extend(arr)
                else:
                    result[k] = v
            batch = []

    if batch:
        batch_result = process_batch(batch, llm, llm_args.layer, args, device)
        for k, v in batch_result.items():
            if k in result:
                for key_emb, arr in v.items():
                    result[k].setdefault(key_emb, []).extend(arr)
            else:
                result[k] = v

    output_path = "final_layer_embeddings.pt"
    torch.save(result, output_path)

    cl_ds = Dataset.create(dataset_name=output_path, dataset_project="fewnerd_pipeline")
    cl_ds.add_files(path=output_path)
    cl_ds.add_tags([output_path])
    cl_ds.upload()
    cl_ds.finalize()


if __name__ == "__main__":
    main()
