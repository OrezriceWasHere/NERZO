"""
multiconer_cascadener_eval_with_positions.py
--------------------------------------------
Span-extraction evaluation on MultiCoNER v2 (English) with CascadeNER.

* GPU-batched inference (BATCH_SIZE).
* Saves raw generated text per sample.
* Keeps gold mentions as  {'text', 'fine_type'}.
* Stores predicted spans with character offsets.
* Logs running precision / recall / F1.
* Dumps JSON and uploads to ClearML.
"""
from typing import Dict, List, Set, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer

from base_extractor import (
    LLMExtractionRunner,
    extract_entities_with_positions,
)

import clearml_poc

# --------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------
BATCH_SIZE = 50          # GPU batch
MAX_NEW    = 4096          # generation cut-off

REPO   = "CascadeNER/models_for_CascadeNER"
SUBF   = "extractor"





# --------------------------------------------------------------------
# Helper ② – gold-span reconstruction for MultiCoNER labels
# --------------------------------------------------------------------
def gold_spans(tokens: List[str], ner_tags: List[str]) -> Tuple[List[Dict[str, str]], Set[str]]:
    spans: List[Dict[str, str]] = []
    current: List[str] = []
    cur_label: str | None = None

    def flush() -> None:
        nonlocal current, cur_label
        if current:
            spans.append({"text": " ".join(current), "fine_type": cur_label or ""})
            current, cur_label = [], None

    for w, tag in zip(tokens, ner_tags):
        if tag == "O":
            flush()
        elif tag.startswith("B-"):
            flush()
            current = [w]
            cur_label = tag[2:].lower()
        elif tag.startswith("I-") and current:
            current.append(w)
        else:
            flush()
            current = [w]
            cur_label = tag[2:].lower() if tag.startswith("I-") else tag.lower()
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
        task_name="CascadeNER − MultiCoNER Span Extraction Evaluation",
        queue_name="a100_gpu",
        requirements=["transformers==4.46.2", "accelerate"],
    )

    ds = load_dataset("MultiCoNER/multiconer_v2", "English (EN)")
    data = list(ds["test"]) + list(ds["validation"]) + list(ds["train"])

    sentences, gold_lists, gold_sets, ids = [], [], [], []
    for ex in data:
        sent = " ".join(ex["tokens"])
        g_list, g_set = gold_spans(ex["tokens"], ex["ner_tags"])
        sentences.append(sent)
        gold_lists.append(g_list)
        gold_sets.append(g_set)
        ids.append(ex.get("id", ex.get("sample_id")))

    runner = LLMExtractionRunner(
        sentences=sentences,
        gold_lists=gold_lists,
        gold_sets=gold_sets,
        ids=ids,
        extract_fn=extract_entities_with_positions,
        batch_size=BATCH_SIZE,
        max_new_tokens=MAX_NEW,
        dataset_project="multiconer_pipeline",
        results_path="span_extraction_results.json",
    )
    runner.evaluate()
