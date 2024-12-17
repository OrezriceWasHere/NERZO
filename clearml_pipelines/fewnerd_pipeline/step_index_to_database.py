import os
import hashlib
import ijson
from clearml import Task, Dataset
from elasticsearch import Elasticsearch
from tqdm import tqdm

from clearml_pipelines.fewnerd_pipeline import fewnerd_dataset
import urllib3
urllib3.disable_warnings()

# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name="fewnerd_pipeline", task_name="Pipeline step 3 index to database",
                 reuse_last_task_id=False)

task.execute_remotely()

hosts = os.environ.get("ELASTICSEARCH_HOSTS") or "http://dsicpu01:9200"
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
            "doc_id": {
                "type": "keyword"
            },
            "embedding": {
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
                    },
                    "llama_3_entire_model": {
                        "type": "object",
                        "properties": {
                            "start": {
                                "type": "dense_vector",
                                "dims": 4096,
                                "index": "false"
                            },
                            "end": {
                                "type": "dense_vector",
                                "dims": 4096,
                                "index": "false"
                            }
                        }
                    },
                    "llama_3_31_v_proj": {
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
                    },
                    "llama_3_3_13_k_proj": {
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
    },
    "settings": {
        "index": {
            "max_inner_result_window": 1000000
        }
    }
}

es = Elasticsearch(hosts=hosts,
                   verify_certs=False,
                    max_retries=10,
                   basic_auth=(user, password))


def ensure_existing_index(index_name, mapping):
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=mapping)

def generate_id(item):
    keys = ["all_text", "coarse_type", "fine_type", "index_end", "index_start"]
    hash_v = hashlib.sha1(str.encode("".join([str(item[key]) for key in keys]))).hexdigest()
    return f"fnd_{hash_v}"

def write_to_index(data, index):
    data["doc_id"] = generate_id(data)
    body = {
        "doc": data,
        "doc_as_upsert": True
    }
    response = es.update(index=index, id=data["doc_id"], body=body)
    return response


# Iterate over all dataset, and get artifacts
# from step_download import datasets
#
# env_to_version_map = {
#     "dev": "1.0.6",
#     "test": "1.0.4",
#     "train": "1.0.3"
# }

for dataset in tqdm(fewnerd_dataset.datasets):
    env = dataset["env"]
    # Load the artifact
    dataset_dir = Dataset.get(dataset_name=dataset["json"],
                              # dataset_version=env_to_version_map[env],
                              dataset_project="fewnerd_pipeline").get_local_copy()

    print(f"Processing dataset {env}")
    index = f"fewnerd_v4_{env}"
    ensure_existing_index(index, mapping)

    with open(os.path.join(dataset_dir, dataset["json"])) as file:
        for document in tqdm(ijson.items(file, "item")):
            write_to_index(document, index)



print('Notice, artifacts are uploaded in the background')
print('Done')
