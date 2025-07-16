"""nertrieve_extraction.py
-----------------------

Span-extraction evaluation on the NERtrieve dataset with CascadeNER.

* Downloads the corpus and entity annotations from ClearML.
* GPU-batched inference (``BATCH_SIZE``).
* Saves raw generated text per sample.
* Stores predicted spans with character offsets.
* Logs running precision / recall / F1.
* Dumps JSON and uploads to ClearML.
"""

import json
import os
from typing import Dict
from clearml import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from base_extractor import (
    LLMExtractionRunner,
    extract_entities_with_positions,
)

import clearml_poc

# --------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------
BATCH_SIZE = 100  # GPU batch
MAX_NEW = 4096  # generation cut-off

REPO = "CascadeNER/models_for_CascadeNER"
SUBF = "extractor"

DATASET_NAME = "neretrieve_test_ir_base"
DATASET_PROJECT = "neretrieve_pipeline"

ENTITIES_FILE = "neretrieve_downsampled_350k.jsonl"
CORPUS_FILES = "NERetrive_IR_corpus.jsonl"


def download_jsonl_file(file) -> list[Dict]:
    """Download the extraction dataset from ClearML and return it."""
    ds = Dataset.get(dataset_name=DATASET_NAME, dataset_project=DATASET_PROJECT)
    path = os.path.join(ds.get_local_copy(), file)
    result = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in tqdm(fh, desc=f"Loading {file}"):
            result.append(json.loads(line))

    return result


def clean_word(text: str) -> str:
    remove_chars = [",", ".", "(", ")", ":", ";", "'", "'s"]
    for char in remove_chars:
        text = text.replace(char, "")
    return text.lower()


def parse_base_dataset(corpus_file, entities_file) -> Dict[str, Dict]:
    corpus = {record["id"]: record for record in corpus_file}
    entities_file = {record["id"]: record for record in entities_file}

    parsed_dataset: Dict[str, Dict] = {}

    for key in tqdm(entities_file.keys(), desc="building dataset"):
        sentence = corpus[key]["content"]
        split = sentence.split(" ")
        gold = []
        for fine_type, span in entities_file[key]["tagged_entities"].items():
            for instances in span.values():
                for text, spans in instances.items():
                    for location in spans:
                        if not location:
                            continue
                        start_word = min(location)
                        end_word = max(location)
                        start_index = (
                            len(" ".join(split[0:start_word])) + 1
                            if start_word > 0
                            else 0
                        )
                        end_index = len(" ".join(split[0 : end_word + 1]))
                        gold.append(
                            {
                                "text": text,
                                "fine_type": fine_type,
                                "start": start_index,
                                "end": end_index,
                            }
                        )

        parsed_dataset[key] = {"text": sentence, "gold": gold}

    return parsed_dataset


# --------------------------------------------------------------------
# Helper ③ – build prompt (matches CascadeNER/demo.py)
# --------------------------------------------------------------------
def build_prompt(sentence: str, tokenizer: AutoTokenizer) -> str:
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": sentence},
    ]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
if __name__ == "__main__":
    clearml_poc.clearml_init(
        task_name="CascadeNER − NERtrieve Extraction",
        queue_name="a100_gpu",
        requirements=["transformers==4.46.2", "accelerate"],
    )

    corpus_file = download_jsonl_file(CORPUS_FILES)
    entities_file = download_jsonl_file(ENTITIES_FILE)
    dataset = parse_base_dataset(corpus_file, entities_file)

    sentences, gold_lists, gold_sets, ids = [], [], [], []
    for doc_id, doc in dataset.items():
        sent = doc["text"]
        g_list = doc["gold"]
        g_set = {g["text"] for g in g_list}
        sentences.append(sent)
        gold_lists.append(g_list)
        gold_sets.append(g_set)
        ids.append(doc_id)

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
