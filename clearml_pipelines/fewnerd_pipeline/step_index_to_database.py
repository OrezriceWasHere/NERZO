import json
import os

from clearml import Task, Dataset
from elasticsearch import Elasticsearch
from tqdm import tqdm

from clearml_pipelines.fewnerd_pipeline import fewnerd_dataset

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
            "all_text": {
                "type": "text"
            },
            "coarse_type": {
                "type": "keyword"
            },
            "fine_type": {
                "type": "keyword"
            },
            "index_end": {
                "type": "integer"
            },
            "index_start": {
                "type": "integer"
            },
            "phrase": {
                "type": "text"
            },
            "text_id": {
                "type": "keyword"
            },
            "embeddings": {
                "type": "object",
                "properties": {
                    "llama_3_17_v_proj": {
                        "type": "object",
                        "properties": {
                            "start": {
                                "type": "dense_vector",
                                "dims": 1024,
                                "index": "false"
                            },
                            "end": {
                                "type": "dense_vector",
                                "dims": 1024,
                                "index": "false"
                            }
                        }
                    }
                }
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


# Iterate over all dataset, and get artifacts
# from step_download import datasets


for dataset in tqdm(fewnerd_dataset.datasets):
    env = dataset["env"]
    # Load the artifact
    dataset_dir = Dataset.get(dataset_name=dataset["json"], dataset_project="fewnerd_pipeline").get_local_copy()
    dataset = json.load(open(os.path.join(dataset_dir, dataset["json"])))
    # Process the dataset
    print(f"Processing dataset {env}")

    index = f"fewnerd_v4_{env}"

    ensure_existing_free_index(index, mapping)

    for document in tqdm(dataset):
        write_to_index(document, index)

print('Notice, artifacts are uploaded in the background')
print('Done')
