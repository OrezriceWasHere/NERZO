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
import clearml_helper

BATCH_SIZE = 10
MAX_BATCHES = 40  # ~1000 records for debugging



def load_dataset() -> Dict[str, Dict]:
    """Download the extraction dataset from ClearML and return it as a mapping
    from text id to record."""
    ds = Dataset.get(dataset_name="span_extraction_results.json",
                     dataset_project="fewnerd_pipeline")
    path = os.path.join(ds.get_local_copy(), "span_extraction_results.json")
    with open(path, "r", encoding="utf-8") as fh:
        data: Dict[str, Dict] = json.load(fh)
    return data


def process_batch_llm(
    batch: List[Tuple[str, Dict]],
    llm: LLMInterface,
    layer: str,
    args: Arguments,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[str]]:
    """Forward a batch of texts through the LLM and collect entity representations."""
    sentences = [record["sentence"] for _, record in batch]
    tokens = llm.tokenize(sentences).to(device)

    with torch.no_grad():
        hidden = llm.get_llm_at_layer(tokens, layer)

    llm_reprs: List[torch.Tensor] = []
    owners: List[str] = []

    for (text_id, record), h in zip(batch, hidden):
        for ent in record.get("predicted", []):
            indices = (ent["start"], ent["end"])
            if ent["end"] == ent["start"]:
                continue
            tok_idx = llm.token_indices_given_text_indices(record["sentence"], indices)
            start = h[tok_idx[0] - 1]
            end = h[tok_idx[1]]
            repr_tensor = fewnerd_processor.choose_llm_representation(
                end.cpu().tolist(),
                start.cpu().tolist(),
                input_tokens=args.input_tokens,
            ).to(device)
            llm_reprs.append(repr_tensor)
            owners.append(text_id)

    return llm_reprs, owners


def process_batch_mlp(
    llm_reprs: List[torch.Tensor],
    owners: List[str],
    mlp: torch.nn.Module,
    device: torch.device,
) -> Dict[str, List[List[float]]]:
    """Forward LLM representations through the MLP."""
    mlp.eval()

    batch_result: Dict[str, List[List[float]]] = {}
    if llm_reprs:
        batch_tensor = torch.stack(llm_reprs).to(device)
        with torch.no_grad():
            mlp_out = mlp(batch_tensor).cpu().tolist()
        for oid, emb in zip(owners, mlp_out):
            batch_result.setdefault(str(oid), []).append(emb)

    return batch_result


def main():
    clearml_poc.clearml_init(task_name="FewNERD LLM+MLP forward",
                             project_name="fewnerd_pipeline",
                             requirements=["transformers==4.46.2", "accelerate"])

    args = Arguments()
    clearml_poc.clearml_connect_hyperparams(args, "general")
    llm_args = FineTuneLLM()
    clearml_poc.clearml_connect_hyperparams(llm_args, "llm_args")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm = LLMInterface(llm_id=llm_args.llm_id, max_llm_layer=llm_args.max_llm_layer)
    mlp = clearml_helper.get_mlp_by_id(llm_args.mlp_head_model_id_from_clearml, device=device)
    mlp = mlp.float()
    llm.model.eval()
    mlp.eval()

    records = load_dataset()

    result: Dict[str, List[List[float]]] = {}
    batch: List[Tuple[str, Dict]] = []
    batches_done = 0

    for text_id, record in tqdm(records.items()):
        batch.append((text_id, record))
        if len(batch) >= BATCH_SIZE:
            llm_reprs, owners = process_batch_llm(
                batch, llm, llm_args.layer, args, device
            )
            batch_result = process_batch_mlp(llm_reprs, owners, mlp, device)
            for k, v in batch_result.items():
                result.setdefault(k, []).extend(v)
            batch = []
            batches_done += 1
            # if batches_done >= MAX_BATCHES:
            #     break
    # if batch and batches_done < MAX_BATCHES:
    if batch and batches_done :

        llm_reprs, owners = process_batch_llm(
            batch, llm, llm_args.layer, args, device
        )
        batch_result = process_batch_mlp(llm_reprs, owners, mlp, device)
        for k, v in batch_result.items():
            result.setdefault(k, []).extend(v)

    output_path = "llm_mlp_embeddings.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh)

    cl_ds = Dataset.create(dataset_name=output_path, dataset_project="fewnerd_pipeline")
    cl_ds.add_files(path=output_path)
    cl_ds.add_tags([output_path])
    cl_ds.upload()
    cl_ds.finalize()


if __name__ == "__main__":
    main()
