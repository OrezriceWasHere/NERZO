"""
fewnerd_cascadener_eval_with_positions.py
——————————
Span-extraction evaluation + character offsets for every prediction.
"""
import gc
import hashlib
import random, re, json, torch
from collections import defaultdict

from clearml import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

import clearml_poc

ENTITY_REGEX = re.compile(r"##(.*?)##")

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
BATCH_SIZE   = 10
MAX_NEW      = 256
N_EXAMPLES   = 1000
REPO         = "CascadeNER/models_for_CascadeNER"
SUBF         = "extractor"

# ---------------------------------------------------------------------
# Helper: author’s own function (from infer.py)
# ---------------------------------------------------------------------
def extract_entities_with_positions(sentence: str, response: str):
    entities = []
    for match in ENTITY_REGEX.finditer(response):
        entity_text = match.group(1)
        start_idx = sentence.find(entity_text)
        if start_idx != -1:
            end_idx = start_idx + len(entity_text)
            entities.append({"text": entity_text, "start": start_idx, "end": end_idx})
        else:
            # fallback: first occurrence
            for m in re.finditer(re.escape(entity_text), sentence):
                entities.append({"text": entity_text,
                                 "start": m.start(), "end": m.end()})
                break
    return entities

# ---------------------------------------------------------------------
# Gold-span reconstruction (contiguous non-zero tag = mention)
# ---------------------------------------------------------------------
def gold_spans(tokens, tags):
    spans, cur, prev = [], [], 0
    for w, t in zip(tokens, tags):
        if t:
            if t == prev:
                cur.append(w)
            else:
                if cur: spans.append(" ".join(cur))
                cur = [w]
            prev = t
        else:
            if cur: spans.append(" ".join(cur)); cur=[]
            prev = 0
    if cur: spans.append(" ".join(cur))
    return set(spans)

if __name__ == "__main__":
    clearml_poc.clearml_init(
        task_name="CascadeNER - Span Extraction Evaluation",
        queue_name="a100_gpu",
        requirements=["transformers==4.46.2", "accelerate"],
    )


# ---------------------------------------------------------------------
# Load data + model
# ---------------------------------------------------------------------
ds   = load_dataset("DFKI-SLT/few-nerd", "supervised", split="test")
data = random.sample(list(ds), N_EXAMPLES)

tok = AutoTokenizer.from_pretrained(REPO, subfolder=SUBF,
                                    trust_remote_code=True,
                                    padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
            REPO, subfolder=SUBF,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True)

# prep sentences, gold sets, prompts
sentences = [" ".join(ex["tokens"]) for ex in data]
gold_sets = [gold_spans(ex["tokens"], ex["ner_tags"]) for ex in data]

def build_prompt(s):
    return tok.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user",   "content": s}],
        tokenize=False, add_generation_prompt=True)



prompts = [build_prompt(s) for s in sentences]

    # ---------------------------------------------------------------------
    # Evaluation loop
    # ---------------------------------------------------------------------
gold_total = pred_total = correct = 0
results = dict()                          # collect predictions with offsets
pbar = tqdm(range(0, N_EXAMPLES, BATCH_SIZE))

model.eval()
with torch.no_grad():
    for start in pbar:
        batch_prompts = prompts[start:start+BATCH_SIZE]
        batch_sents   = sentences[start:start+BATCH_SIZE]
        batch_gold    = gold_sets[start:start+BATCH_SIZE]

        enc = tok(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_len = enc["attention_mask"].sum(1)

        outs = model.generate(**enc, max_new_tokens=MAX_NEW)
        decoded = tok.batch_decode([
            seq[prompt_len[idx]:] for idx, seq in enumerate(outs)
        ], skip_special_tokens=True)

        for i, generated in enumerate(decoded):

            # — predicted spans & offsets —
            preds_pos = extract_entities_with_positions(batch_sents[i], generated)
            for p in preds_pos:
                assert batch_sents[i][p["start"]:p["end"]] == p["text"]
            preds_txt = {p["text"].strip() for p in preds_pos if p["text"].strip()}


            # — metrics —
            ground_truth = batch_gold[i]
            gold_total += len(ground_truth)
            pred_total += len(preds_txt)
            correct    += len(ground_truth & preds_txt)

            sentence_hash = hashlib.sha1(batch_sents[i].encode()).hexdigest()

            results[sentence_hash] = {
                "id":        data[start+i]["id"],
                "sentence":  batch_sents[i],
                "gold":      list(batch_gold[i]),
                "predicted": preds_pos
            }

        prec = correct / pred_total if pred_total else 0
        rec  = correct / gold_total if gold_total else 0
        f1   = 2*prec*rec/(prec+rec) if prec+rec else 0
        pbar.set_description(f"Prec {prec:.2f}  Rec {rec:.2f}  F1 {f1:.2f}")

    torch.cuda.empty_cache()
    gc.collect()

# ---------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------
print("\n— Final results —")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1-score:  {f1:.3f}\n")

import os

# Save results to JSON file
results_path = "span_extraction_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)


clearml_dataset = Dataset.create(dataset_name=results_path, dataset_project="fewnerd_pipeline")
clearml_dataset.add_files(path=results_path)
clearml_dataset.add_tags([results_path])

    # Dataset is uploaded to the ClearML Server by default
clearml_dataset.upload()
clearml_dataset.finalize()
