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
from typing import Dict, List, Set, Tuple

from datasets import load_dataset

from base_extractor import (
    LLMExtractionRunner,
    align_to_original
)

import clearml_poc

# --------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------
BATCH_SIZE = 100          # GPU batch
MAX_NEW    = 4096          # generation cut-off

REPO   = "CascadeNER/models_for_CascadeNER"
SUBF   = "extractor"
CONFIG = "supervised"     # Few-NERD config




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
    char_pos = 0
    span_start = 0
    span_end = 0

    def flush() -> None:
        nonlocal current, cur_fine, span_start, span_end
        if current:
            spans.append(
                {
                    "text": " ".join(current),
                    "fine_type": fine_id2lab[cur_fine] if fine_id2lab else cur_fine,
                    "start": span_start,
                    "end": span_end,
                }
            )
            current = []

    for idx, (w, t, ft) in enumerate(zip(tokens, ner_tags, fine_tags)):
        token_start = char_pos
        char_pos += len(w)
        token_end = char_pos
        if idx < len(tokens) - 1:
            char_pos += 1

        if t:  # inside entity
            if t == prev_tag and current:
                current.append(w)
                span_end = token_end
            else:
                flush()
                current = [w]
                cur_fine = ft
                span_start = token_start
                span_end = token_end
            prev_tag = t
        else:  # outside
            flush()
            prev_tag = 0
    flush()
    return spans, {d["text"] for d in spans}


if __name__ == "__main__":
    clearml_poc.clearml_init(
        task_name="CascadeNER − Span Extraction Evaluation fewnerd",
        queue_name="a100_gpu",
        requirements=["transformers==4.46.2", "accelerate"],
    )

    ds = load_dataset("DFKI-SLT/few-nerd", CONFIG)
    data = list(ds["test"]) + list(ds["validation"]) + list(ds["train"])

    fine_feat = ds["test"].features["fine_ner_tags"]
    fine_id2lab = fine_feat.feature.names if hasattr(fine_feat, "feature") else None

    sentences, gold_lists, gold_sets, ids = [], [], [], []
    for ex in data:
        sent = " ".join(ex["tokens"])
        g_list, g_set = gold_spans(ex["tokens"], ex["ner_tags"], ex["fine_ner_tags"], fine_id2lab)
        sentences.append(sent)
        gold_lists.append(g_list)
        gold_sets.append(g_set)
        ids.append(ex["id"])

    runner = LLMExtractionRunner(
        sentences=sentences,
        gold_lists=gold_lists,
        gold_sets=gold_sets,
        ids=ids,
        extract_fn=align_to_original,
        batch_size=BATCH_SIZE,
        max_new_tokens=MAX_NEW,
        dataset_project="fewnerd_pipeline",
        results_path="span_extraction_results.json",
    )
    runner.evaluate()
