"""
nertrieve_extraction.py
-----------------------

Span-extraction evaluation on the NERtrieve dataset with CascadeNER.

* Downloads the dataset from the official URL.
* Samples ``N_EXAMPLES`` random sentences from the test split.
* GPU-batched inference (BATCH_SIZE).
* Saves raw generated text per sample.
* Stores predicted spans with character offsets.
* Logs running precision / recall / F1.
* Dumps JSON and uploads to ClearML.
"""

import bz2
import gc
import hashlib
import json
import os
import re
import random
from typing import Dict, Iterable, List, Set, Tuple

import requests
import torch
from clearml import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import clearml_poc

# --------------------------------------------------------------------
# Regex for ##entity## delimiters
# --------------------------------------------------------------------
ENTITY_REGEX = re.compile(r"##(.*?)##")

# --------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------
BATCH_SIZE = 1  # GPU batch
MAX_NEW = 256     # generation cut-off
N_EXAMPLES = 100  # number of sentences to evaluate

REPO = "CascadeNER/models_for_CascadeNER"
SUBF = "extractor"

# URLs for the supervised NERtrieve dataset
DATA_URLS = [
    (
        "https://storage.googleapis.com/neretrieve_dataset/supervised_ner/NERetrive_sup_test.jsonl.bz2",
        "nertrieve_sup_test.jsonl.bz2",
    ),
]


# --------------------------------------------------------------------
# Helper ① – author\'s positions extractor
# --------------------------------------------------------------------
def extract_entities_with_positions(sentence: str, response: str) -> List[Dict]:
    """Return list of {'text','start','end'} for every ##…## chunk."""
    entities: List[Dict[str, int | str]] = []
    for m in ENTITY_REGEX.finditer(response):
        ent_text = m.group(1)
        start = sentence.find(ent_text)
        if start != -1:
            entities.append({"text": ent_text, "start": start, "end": start + len(ent_text)})
        else:
            for m2 in re.finditer(re.escape(ent_text), sentence):
                entities.append({"text": ent_text, "start": m2.start(), "end": m2.end()})
                break
    return entities


# --------------------------------------------------------------------
# Helper ② – gold-span reconstruction
# --------------------------------------------------------------------
def gold_spans(document: dict) -> Tuple[List[Dict[str, str]], Set[str]]:
    """Reconstruct gold spans from a NERtrieve document."""
    spans: List[Dict[str, str]] = []
    sentence: str = document["content"]
    tokens: List[str] = document.get("document_token_sequence", sentence.split())
    tagging = document.get("tagged_entities", {})

    for entity_type, entity_dict in tagging.items():
        for entity_id, phrase_dict in entity_dict.items():
            for phrase, references in phrase_dict.items():
                for ref in references:
                    if not ref:
                        continue
                    word_index_start = min(ref)
                    index_start = sum(len(word) for word in tokens[:word_index_start]) + word_index_start
                    index_end = index_start + len(" ".join(tokens[min(ref) : max(ref) + 1]))
                    if sentence[index_start:index_end].lower() == phrase.lower():
                        spans.append({"text": phrase, "fine_type": entity_type})
    return spans, {d["text"] for d in spans}


# --------------------------------------------------------------------
# Helper ③ – build prompt (matches CascadeNER/demo.py)
# --------------------------------------------------------------------
def build_prompt(sentence: str, tokenizer: AutoTokenizer) -> str:
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": sentence},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# --------------------------------------------------------------------
# Dataset utilities
# --------------------------------------------------------------------
def download_dataset(url: str, dst: str) -> str:
    if not os.path.exists(dst):
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(dst, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                fh.write(chunk)
    return dst


def iter_documents(path: str) -> Iterable[dict]:
    opener = bz2.open if path.endswith(".bz2") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
if __name__ == "__main__":
    clearml_poc.clearml_init(
        task_name="CascadeNER − NERtrieve Extraction", queue_name="a100_gpu", requirements=["transformers==4.46.2", "accelerate"],
    )

    # ---------------------------------------------------------------
    # Download and load the dataset
    # ---------------------------------------------------------------
    docs = []
    for url, fname in DATA_URLS:
        local = download_dataset(url, fname)
        docs.extend(list(iter_documents(local)))

    if N_EXAMPLES:
        rng = random.Random(0)
        docs = rng.sample(docs, min(N_EXAMPLES, len(docs)))

    # ---------------------------------------------------------------
    # Load CascadeNER extractor (Qwen2-7B)
    # ---------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(REPO, subfolder=SUBF, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        REPO,
        subfolder=SUBF,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    # ---------------------------------------------------------------
    # Prepare sentences, gold lists/sets, prompts
    # ---------------------------------------------------------------
    sentences, gold_lists, gold_sets, prompts, ids = [], [], [], [], []
    for ex in docs:
        sent = ex.get("content", " ".join(ex.get("document_token_sequence", [])))
        g_list, g_set = gold_spans(ex)
        sentences.append(sent)
        gold_lists.append(g_list)
        gold_sets.append(g_set)
        prompts.append(build_prompt(sent, tok))
        ids.append(ex.get("id"))

    # ---------------------------------------------------------------
    # Evaluation loop
    # ---------------------------------------------------------------
    gold_total = pred_total = correct = 0
    results: Dict[str, Dict] = {}
    pbar = tqdm(range(0, len(prompts), BATCH_SIZE))

    with torch.no_grad():
        for start in pbar:
            b_prompts = prompts[start : start + BATCH_SIZE]
            b_sents = sentences[start : start + BATCH_SIZE]
            b_gold_l = gold_lists[start : start + BATCH_SIZE]
            b_gold_set = gold_sets[start : start + BATCH_SIZE]

            enc = tok(b_prompts, return_tensors="pt", padding=True).to(model.device)
            prompt_len = enc["attention_mask"].sum(1)

            outs = model.generate(**enc, max_new_tokens=MAX_NEW)
            decoded = tok.batch_decode(
                [seq[prompt_len[i] :] for i, seq in enumerate(outs)], skip_special_tokens=True
            )

            for i, generated in enumerate(decoded):
                preds_pos = extract_entities_with_positions(b_sents[i], generated)
                for p in preds_pos:
                    assert b_sents[i][p["start"] : p["end"]] == p["text"]
                preds_txt = {p["text"].strip() for p in preds_pos if p["text"].strip()}

                gold_total += len(b_gold_set[i])
                pred_total += len(preds_txt)
                correct += len(b_gold_set[i] & preds_txt)

                sent_hash = hashlib.sha1(b_sents[i].encode()).hexdigest()
                results[sent_hash] = {
                    "id": ids[start + i],
                    "sentence": b_sents[i],
                    "gold": b_gold_l[i],
                    "predicted": preds_pos,
                    "generated": generated,
                }
            del enc
            del outs
            del decoded
            torch.cuda.empty_cache()
            prec = correct / pred_total if pred_total else 0
            rec = correct / gold_total if gold_total else 0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
            pbar.set_description(f"Prec {prec:.2f}  Rec {rec:.2f}  F1 {f1:.2f}")

            torch.cuda.empty_cache()
            gc.collect()

    # ---------------------------------------------------------------
    # Final report + persistence
    # ---------------------------------------------------------------
    print("\n— Final results —")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1-score:  {f1:.3f}")

    results_path = "nertrieve_span_extraction.json"
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    cl_ds = Dataset.create(dataset_name=results_path, dataset_project="nertrieve_pipeline")
    cl_ds.add_files(path=results_path)
    cl_ds.add_tags([results_path])
    cl_ds.upload()
    cl_ds.finalize()
