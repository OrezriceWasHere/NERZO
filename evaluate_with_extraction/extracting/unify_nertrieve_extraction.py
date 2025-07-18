import json
import os
from clearml import Dataset

DATASET_NAME = "nertrieve_test_ir_base"
DATASET_PROJECT = "nertrieve_pipeline"
PARTS = 20


def _load_original_ids():
    ds = Dataset.get(dataset_name=DATASET_NAME, dataset_project=DATASET_PROJECT)
    base = ds.get_local_copy()
    corpus = os.path.join(base, "NERetrive_IR_corpus.jsonl")
    ids = set()
    with open(corpus, "r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            ids.add(rec["id"])
    return ids


def _load_part(i: int) -> dict:
    name = f"nertrieve_test_ir_base_part_{i + 1}_out_of_{PARTS}.json"
    ds = Dataset.get(dataset_name=name, dataset_project=DATASET_PROJECT)
    path = os.path.join(ds.get_local_copy(), name)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main():
    original_ids = _load_original_ids()
    combined = {}
    for i in range(PARTS):
        combined.update(_load_part(i))

    extracted_ids = {v["id"] for v in combined.values()}
    missing = original_ids - extracted_ids
    if missing:
        raise ValueError(f"Missing {len(missing)} ids from extraction")

    output_file = "nertrieve_test_ir_base_combined.json"
    with open(output_file, "w", encoding="utf-8") as fh:
        json.dump(combined, fh, ensure_ascii=False, indent=2)

    cl_ds = Dataset.create(dataset_name=output_file, dataset_project=DATASET_PROJECT)
    cl_ds.add_files(output_file)
    cl_ds.add_tags([output_file])
    cl_ds.upload()
    cl_ds.finalize()


if __name__ == "__main__":
    main()
