import json
import os

from clearml import Task
from elasticsearch import Elasticsearch
from tqdm import tqdm

# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name="fewnerd_pipeline", task_name="Pipeline step 3 index to database",
                 reuse_last_task_id=False)

task.execute_remotely()


hosts = os.environ.get("ELASTICSEARCH_HOSTS") or "https://dsicscpu01:9200"
user = os.environ.get("ELASTICSEARCH_USER") or "elastic"
password = os.environ.get("ELASTICSEARCH_PASSWORD") or "XXX"

mapping = {
    "mappings": {
        "properties": {
            "full_text": {
                "type": "text"
            },
            "tagging": {
                "properties": {
                    "coarse_type": {
                        "type": "keyword"
                    },
                    "fine_type": {
                        "type": "keyword"
                    },
                    "index_start": {
                        "type": "integer"
                    },
                    "index_end": {
                        "type": "integer"
                    },
                    "phrase": {
                        "type": "text"
                    },
                    "text_id": {
                        "type": "keyword"
                    }
                }
            },
            "text_id": {
                "type": "keyword"
            }
        }
    }
}

es = Elasticsearch(hosts=hosts,
                   verify_certs=False,
                   basic_auth=(user, password))


def ensure_existing_free_index(index_name, mapping):
    es.options(ignore_status=[400, 404]).indices.delete(index=index_name)
    es.options(ignore_status=[400, 404]).indices.create(index=index_name, body=mapping)


def write_to_index(data, index):
    response = es.index(index=index, body=data, id=data["text_id"])
    return response



# Find task of previous step by name - pipeline step 2 jsonify dataset
previous_task = Task.get_task(project_name="fewnerd_pipeline",
                              task_name="Pipeline step 2 jsonify dataset",
                              task_filter={'order_by': ["-last_update"]})
# Get the artifacts from the previous step
previous_task_artifacts = previous_task.artifacts

# Iterate over all dataset, and get artifacts
from step_download import datasets

for dataset in tqdm(datasets):
    env = dataset["env"]
    # Load the artifact
    dataset = json.load(open(dataset["json"], "r"))
    # Process the dataset
    print(f"Processing dataset {env}")

    ensure_existing_free_index(f"fewnerd_v2_{env}", mapping)

    for document in tqdm(dataset):
        write_to_index(document, f"fewnerd_v2_{env}")



print('Notice, artifacts are uploaded in the background')
print('Done')
