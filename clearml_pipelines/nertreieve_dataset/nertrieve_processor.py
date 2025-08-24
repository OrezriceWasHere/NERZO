import json
from functools import partial, cache
import glob



@cache
def type_to_name() -> dict[str, str]:
    return json.load(open("clearml_pipelines/nertreieve_dataset/entities_to_names.json"))

@cache
def text_id_to_labels() -> dict[str, list[str]]:
    dir = "clearml_pipelines/nertreieve_dataset"
    prefix = "text_id_to_labels"
    # Find all JSON files with the specified prefix
    json_files = sorted(glob.glob(f"{dir}/{prefix}*.json"))
    merged_data = {}

    for file in json_files:
        try:
            with open(file, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    merged_data.update(data)  # Merge dictionaries
                else:
                    print(f"Skipping {file}: Not a dictionary")
        except json.JSONDecodeError:
            print(f"Skipping {file}: Invalid JSON")

    return merged_data

