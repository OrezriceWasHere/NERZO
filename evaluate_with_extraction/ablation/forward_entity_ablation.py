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

MLP_ID = "137515e67cb14851b85d55846e630337"


def load_dataset() -> Dict[str, Dict]:
    """Download the extraction dataset from ClearML and return it."""
    ds = Dataset.get(dataset_name="span_extraction_results.json", dataset_project="fewnerd_pipeline")
    path = os.path.join(ds.get_local_copy(), "span_extraction_results.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def process_batch(
    batch: List[Tuple[str, Dict]],
    llm: LLMInterface,
    layer: str,
    args: Arguments,
    device: torch.device,
    mlp: torch.nn.Module,
) -> Dict[str, Dict[str, List[List[float]]]]:
    """Forward a batch of texts and return embeddings."""
    eos_token = llm.tokenizer.eos_token
    sentences = [record["sentence"] + eos_token for _, record in batch]
    tokens = llm.tokenize(sentences).to(device)
    # Extract the EOS embedding at the real end of each sentence. We append the
    # EOS token to every text and then compute its token index using
    # `tokens_count` similar to the logic in
    # `step_process_ir_enrich_with_llama_test_to_document.py`.
    eos_indices = [llm.tokens_count(text) - 1 for text in sentences]

    with torch.no_grad():
        hidden = llm.get_llm_at_layer(tokens, layer)

    result: Dict[str, Dict[str, List[List[float]]]] = {}
    owners: List[str] = []
    eos_vecs: List[torch.Tensor] = []
    llm_reprs: List[torch.Tensor] = []

    for (text_id, record), h, eos_idx in zip(batch, hidden, eos_indices):
        eos_vec = h[eos_idx]
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
            result.setdefault(text_id, {}).setdefault("llama_3_1_17_v_entity_end", []).append(rep.tolist())
            result[text_id].setdefault("llama_3_1_17_v_entity_eos", []).append(eos_vec.cpu().tolist())
            llm_reprs.append(rep.to(device))
            eos_vecs.append(eos_vec.to(device))
            owners.append(text_id)

    if llm_reprs:
        concat = torch.stack([torch.cat((r, e), dim=-1) for r, e in zip(llm_reprs, eos_vecs)]).to(device)
        with torch.no_grad():
            mlp_out = mlp(concat).cpu().tolist()
        for oid, emb in zip(owners, mlp_out):
            result.setdefault(oid, {}).setdefault("mlp_137515e67cb14851b85d55846e630337", []).append(emb)

    return result


def main() -> None:
    clearml_poc.clearml_init(
        task_name="FewNERD Ablation Forward", project_name="fewnerd_pipeline",
        requirements=["transformers==4.46.2", "accelerate"]
    )

    args = Arguments()
    clearml_poc.clearml_connect_hyperparams(args, "general")
    llm_args = FineTuneLLM()
    clearml_poc.clearml_connect_hyperparams(llm_args, "llm_args")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm = LLMInterface(llm_id=llm_args.llm_id, max_llm_layer=llm_args.max_llm_layer)
    mlp = clearml_helper.get_mlp_by_id(MLP_ID, device=device).float()
    llm.model.eval()
    mlp.eval()

    records = load_dataset()

    result: Dict[str, Dict[str, List[List[float]]]] = {}
    batch: List[Tuple[str, Dict]] = []

    for text_id, record in tqdm(records.items()):
        batch.append((text_id, record))
        if len(batch) >= BATCH_SIZE:
            batch_result = process_batch(batch, llm, llm_args.layer, args, device, mlp)
            for k, v in batch_result.items():
                if k in result:
                    for key_emb, arr in v.items():
                        result[k].setdefault(key_emb, []).extend(arr)
                else:
                    result[k] = v
            batch = []

    if batch:
        batch_result = process_batch(batch, llm, llm_args.layer, args, device, mlp)
        for k, v in batch_result.items():
            if k in result:
                for key_emb, arr in v.items():
                    result[k].setdefault(key_emb, []).extend(arr)
            else:
                result[k] = v

    output_path = "ablation_embeddings.pt"
    torch.save(result, output_path)

    cl_ds = Dataset.create(dataset_name=output_path, dataset_project="fewnerd_pipeline")
    cl_ds.add_files(path=output_path)
    cl_ds.add_tags([output_path])
    cl_ds.upload()
    cl_ds.finalize()


if __name__ == "__main__":
    main()
