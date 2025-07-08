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
import random
from typing import Dict, Iterable, List, Set, Tuple

import requests
import torch
from clearml import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from .base_extractor import (
    LLMExtractionRunner,
    build_prompt,
    extract_entities_with_positions,
)

import clearml_poc

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
                        spans.append(
                            {
                                "text": phrase,
                                "fine_type": entity_type,
                                "start": index_start,
                                "end": index_end,
                            }
                        )
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

    docs = []
    for url, fname in DATA_URLS:
        local = download_dataset(url, fname)
        docs.extend(list(iter_documents(local)))

    if N_EXAMPLES:
        rng = random.Random(0)
        docs = rng.sample(docs, min(N_EXAMPLES, len(docs)))

    sentences, gold_lists, gold_sets, ids = [], [], [], []
    for ex in docs:
        sent = ex.get("content", " ".join(ex.get("document_token_sequence", [])))
        g_list, g_set = gold_spans(ex)
        sentences.append(sent)
        gold_lists.append(g_list)
        gold_sets.append(g_set)
        ids.append(ex.get("id"))

    runner = LLMExtractionRunner(
        sentences=sentences,
        gold_lists=gold_lists,
        gold_sets=gold_sets,
        ids=ids,
        extract_fn=extract_entities_with_positions,
        batch_size=BATCH_SIZE,
        max_new_tokens=MAX_NEW,
        dataset_project="nertrieve_pipeline",
        results_path="nertrieve_span_extraction.json",
    )
    runner.evaluate()

