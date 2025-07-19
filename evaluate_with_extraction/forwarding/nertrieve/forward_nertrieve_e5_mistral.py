import os
import json
from typing import Dict

import torch
from clearml import Dataset

import clearml_poc
from sentence_embedder import SentenceEmbedder
from evaluate_with_extraction.forwarding.sentence_embedder_forwarder import forward_dataset, BATCH_SIZE

EMBEDDER_ID = "intfloat/e5-mistral-7b-instruct"

DATASET_NAME = "nertrieve_test_ir_base_combined.json"
DATASET_PROJECT = "nertrieve_pipeline"
OUTPUT_FILE = "sentence_embeddings_e5.pth"


def load_dataset() -> Dict[str, Dict]:
    ds = Dataset.get(dataset_name=DATASET_NAME, dataset_project=DATASET_PROJECT)
    path = os.path.join(ds.get_local_copy(), DATASET_NAME)
    with open(path, "r", encoding="utf-8") as fh:
        data: Dict[str, Dict] = json.load(fh)
    return data


def upload_result(path: str) -> None:
    cl_ds = Dataset.create(dataset_name=path, dataset_project=DATASET_PROJECT)
    cl_ds.add_files(path=path)
    cl_ds.add_tags([path])
    cl_ds.upload()
    cl_ds.finalize()


def main() -> None:
    clearml_poc.clearml_init(
        task_name="NERtrieve sentence embedder forward E5",
        project_name=DATASET_PROJECT,
        requirements=["transformers==4.46.2", "sentence_transformers", "accelerate", "einops"],
    )

    embedder = SentenceEmbedder(llm_id=EMBEDDER_ID)

    records = load_dataset()
    result = forward_dataset(records, embedder, batch_size=BATCH_SIZE)

    torch.save(result, OUTPUT_FILE)

    upload_result(OUTPUT_FILE)


if __name__ == "__main__":
    main()
