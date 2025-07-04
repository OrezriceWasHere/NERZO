"""
fewnerd_cascadener_eval_with_positions.py
——————————
Span-extraction evaluation on Few-NERD with CascadeNER.

• GPU-batched inference (BATCH_SIZE).
• Saves raw generated text per sample.
• Keeps gold mentions as  {'text', 'fine_type'}.
• Stores predicted spans with character offsets.
• Logs running precision / recall / F1.
• Dumps JSON and uploads to ClearML.
"""
import gc
import hashlib
import json
import re
from typing import Dict, List, Set, Tuple

import torch
from clearml import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

import clearml_poc

# --------------------------------------------------------------------
# Regex for ##entity## delimiters
# --------------------------------------------------------------------
ENTITY_REGEX = re.compile(r"##(.*?)##")

# --------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------
BATCH_SIZE = 200          # GPU batch
MAX_NEW    = 256          # generation cut-off
N_EXAMPLES = 1000         # sample size (None ⇒ whole split)

REPO   = "CascadeNER/models_for_CascadeNER"
SUBF   = "extractor"
CONFIG = "supervised"     # Few-NERD config


# --------------------------------------------------------------------
# Helper ① – author’s positions extractor
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
# Helper ② – gold-span reconstruction (keeps fine type)
# --------------------------------------------------------------------
def gold_spans(
    tokens: List[str],
    ner_tags: List[int],
    fine_tags: List[int],
    fine_id2lab: List[str] | None = None,
) -> Tuple[List[Dict[str, str]], Set[str]]:
    """
    Few-NERD marks every entity token with a single, non-zero tag id.
    A contiguous run of the SAME id = one mention.

    Returns
    -------
    gold_list : [{'text': 'Little Chef', 'fine_type': 'organization_restaurant'}, …]
    gold_set  : {'Little Chef', …}   # for span-level metrics
    """
    spans: List[Dict[str, str]] = []
    current: List[str] = []
    cur_fine = 0
    prev_tag = 0

    def flush():
        if current:
            spans.append(
                {
                    "text": " ".join(current),
                    "fine_type": fine_id2lab[cur_fine] if fine_id2lab else cur_fine,
                }
            )

    for w, t, ft in zip(tokens, ner_tags, fine_tags):
        if t:                              # inside entity
            if t == prev_tag:
                current.append(w)
            else:
                flush()
                current, cur_fine = [w], ft
            prev_tag = t
        else:                              # outside
            flush()
            current, prev_tag = [], 0
    flush()
    return spans, {d["text"] for d in spans}


# --------------------------------------------------------------------
# Helper ③ – build prompt (matches CascadeNER/demo.py)
# --------------------------------------------------------------------
def build_prompt(sentence: str, tokenizer: AutoTokenizer) -> str:
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": sentence},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
if __name__ == "__main__":
    clearml_poc.clearml_init(
        task_name="CascadeNER − Span Extraction Evaluation",
        queue_name="a100_gpu",
        requirements=["transformers==4.46.2", "accelerate"],
    )

    # ---------------------------------------------------------------
    # Load Few-NERD test data
    # ---------------------------------------------------------------
    ds   = load_dataset("DFKI-SLT/few-nerd", CONFIG)
    data = list(ds["test"]) + list(ds["validation"]) + list(ds["train"])

    fine_feat    = ds["test"].features["fine_ner_tags"]
    fine_id2lab  = fine_feat.feature.names if hasattr(fine_feat, "feature") else None

    # ---------------------------------------------------------------
    # Load CascadeNER extractor (Qwen2-7B)
    # ---------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(
        REPO, subfolder=SUBF, trust_remote_code=True, padding_side="left"
    )
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
    sentences, gold_lists, gold_sets, prompts = [], [], [], []
    for ex in data:
        sent = " ".join(ex["tokens"])
        g_list, g_set = gold_spans(ex["tokens"], ex["ner_tags"], ex["fine_ner_tags"], fine_id2lab)
        sentences.append(sent)
        gold_lists.append(g_list)
        gold_sets.append(g_set)
        prompts.append(build_prompt(sent, tok))

    # ---------------------------------------------------------------
    # Evaluation loop
    # ---------------------------------------------------------------
    gold_total = pred_total = correct = 0
    results: Dict[str, Dict] = {}
    pbar = tqdm(range(0, len(prompts), BATCH_SIZE))

    with torch.no_grad():
        for start in pbar:
            b_prompts  = prompts[start : start + BATCH_SIZE]
            b_sents    = sentences[start : start + BATCH_SIZE]
            b_gold_l   = gold_lists[start : start + BATCH_SIZE]
            b_gold_set = gold_sets[start : start + BATCH_SIZE]

            enc = tok(b_prompts, return_tensors="pt", padding=True).to(model.device)
            prompt_len = enc["attention_mask"].sum(1)            # prompt length per sample

            outs = model.generate(**enc, max_new_tokens=MAX_NEW)
            decoded = tok.batch_decode(
                [seq[prompt_len[i] :] for i, seq in enumerate(outs)],
                skip_special_tokens=True,
            )

            for i, generated in enumerate(decoded):
                # predicted spans & offsets
                preds_pos = extract_entities_with_positions(b_sents[i], generated)
                for p in preds_pos:
                    assert b_sents[i][p["start"] : p["end"]] == p["text"]
                preds_txt = {p["text"].strip() for p in preds_pos if p["text"].strip()}

                # metrics
                gold_total += len(b_gold_set[i])
                pred_total += len(preds_txt)
                correct    += len(b_gold_set[i] & preds_txt)

                sent_hash = hashlib.sha1(b_sents[i].encode()).hexdigest()
                results[sent_hash] = {
                    "id":        data[start + i]["id"],
                    "sentence":  b_sents[i],
                    "gold":      b_gold_l[i],      # ← keeps fine_type
                    "predicted": preds_pos,
                    "generated": generated,
                }
            del enc
            del outs
            del decoded
            torch.cuda.empty_cache()
            prec = correct / pred_total if pred_total else 0
            rec  = correct / gold_total if gold_total else 0
            f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0
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

    # Save JSON
    results_path = "span_extraction_results.json"
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    # Upload to ClearML
    cl_ds = Dataset.create(dataset_name=results_path, dataset_project="fewnerd_pipeline")
    cl_ds.add_files(path=results_path)
    cl_ds.add_tags([results_path])
    cl_ds.upload()
    cl_ds.finalize()