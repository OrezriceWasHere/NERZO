import json
import os
from clearml import Dataset
from tqdm import tqdm

from evaluate_with_extraction.extracting import extracting_entities_nertrieve

DATASET_NAME = "neretrieve_test_ir_base"
DATASET_PROJECT = "nertrieve_pipeline"
PARTS = 20


def _load_original_ids():
    ds = Dataset.get(dataset_name=DATASET_NAME, dataset_project="neretrieve_pipeline")
    base = ds.get_local_copy()
    corpus = os.path.join(base, extracting_entities_nertrieve.ENTITIES_FILE)
    ids = set()
    with open(corpus, "r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            ids.add(rec["id"])
    return ids


def _load_part(i: int) -> dict:
    from evaluate_with_extraction.evaluation.extract_entities_regex_v2 import align_to_original
    name = f"nertrieve_test_ir_base_part_{i + 1}_out_of_{PARTS}.json"
    ds = Dataset.get(dataset_name=name, dataset_project=DATASET_PROJECT)
    path = os.path.join(ds.get_local_copy(), name)
    with open(path, "r", encoding="utf-8") as fh:
        content = json.load(fh)
        for k, v in content.items():
            marked = v["generated"]
            original = v["sentence"]
            prediction = align_to_original(marked=marked, original=original,longest_only=True,all_occurrences=True)
            v["predicted"] = prediction
        return content




def main():
    import clearml_poc
    clearml_poc.clearml_init(
        task_name="Unify NERtrieve extraction",
        project_name=DATASET_PROJECT
    )
    original_ids = _load_original_ids()
    combined = {}
    for i in tqdm(range(PARTS), desc="Loading parts"):
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
