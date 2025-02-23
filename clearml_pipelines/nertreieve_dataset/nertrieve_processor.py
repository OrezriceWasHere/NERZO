import json
from functools import partial, cache



@cache
def type_to_name() -> dict[str, str]:
    return json.load(open("clearml_pipelines/nertreieve_dataset/entities_to_names.json"))