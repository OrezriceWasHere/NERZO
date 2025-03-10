import json
from functools import partial, cache



@cache
def type_to_name() -> dict[str, str]:
    return json.load(open("clearml_pipelines/nertreieve_dataset/entities_to_names.json"))

@cache
def text_id_to_labels() -> dict[str, list[str]]:
    with open("clearml_pipelines/nertreieve_dataset/text_id_to_labels.json", "r") as f:
        mapping = json.load(f)
        mapping = {text_id: list(set(values)) for text_id, values in mapping.items()}
        return mapping
