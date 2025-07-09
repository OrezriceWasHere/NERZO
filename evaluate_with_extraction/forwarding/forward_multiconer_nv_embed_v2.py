import os
import json
from typing import Dict

from clearml import Dataset

import clearml_poc
from sentence_embedder import SentenceEmbedder
from sentence_embedder_forwarder import forward_dataset, BATCH_SIZE

# ID of the sentence embedder used for forwarding
EMBEDDER_ID = "nvidia/NV-Embed-v2"

DATASET_NAME = "span_extraction_results.json"
DATASET_PROJECT = "multiconer_pipeline"
OUTPUT_FILE = "sentence_embeddings_nv.json"


def load_dataset() -> Dict[str, Dict]:
    """Download the extraction dataset from ClearML and return it."""
    ds = Dataset.get(dataset_name=DATASET_NAME, dataset_project=DATASET_PROJECT)
    path = os.path.join(ds.get_local_copy(), DATASET_NAME)
    with open(path, "r", encoding="utf-8") as fh:
        data: Dict[str, Dict] = json.load(fh)
    return data


def upload_result(path: str) -> None:
    """Upload the resulting embedding dataset to ClearML."""
    cl_ds = Dataset.create(dataset_name=path, dataset_project=DATASET_PROJECT)
    cl_ds.add_files(path=path)
    cl_ds.add_tags([path])
    cl_ds.upload()
    cl_ds.finalize()


def main() -> None:
    clearml_poc.clearml_init(
        task_name="MultiCoNER sentence embedder forward NV",
        project_name=DATASET_PROJECT,
        requirements=["transformers==4.46.2", "sentence_transformers", "accelerate"],
    )

    embedder = SentenceEmbedder(llm_id=EMBEDDER_ID)

    records = load_dataset()
    result = forward_dataset(records, embedder, batch_size=BATCH_SIZE)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(result, fh)

    upload_result(OUTPUT_FILE)


if __name__ == "__main__":
    main()
