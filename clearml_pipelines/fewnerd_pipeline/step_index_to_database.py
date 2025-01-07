import asyncio
import os
import hashlib

import ijson
from clearml import Task, Dataset
from elasticsearch import Elasticsearch, AsyncElasticsearch
from tqdm.asyncio import  tqdm

from clearml_pipelines.fewnerd_pipeline import fewnerd_dataset
import urllib3

from runtime_args import RuntimeArgs

urllib3.disable_warnings()

# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name="fewnerd_pipeline", task_name="Pipeline step 3 index to database",
                 reuse_last_task_id=False)

task.execute_remotely()


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


async def ensure_existing_index(index_name, mapping):
    if not await es.indices.exists(index=index_name):
        await es.indices.create(index=index_name, body=mapping)

def generate_id(item):
    keys = ["all_text", "coarse_type", "fine_type", "index_end", "index_start"]
    hash_v = hashlib.sha1(str.encode("".join([str(item[key]) for key in keys]))).hexdigest()
    return f"fnd_{hash_v}"

async def write_to_index(data, index):
    data["doc_id"] = generate_id(data)
    body = {
        "doc": data,
        "doc_as_upsert": True
    }
    response = await es.update(index=index, id=data["doc_id"], body=body)
    return response


async def index_task(dataset):

    env = dataset["env"]
    # Load the artifact
    dataset_dir = Dataset.get(dataset_name=dataset["json"],
                              # dataset_version=env_to_version_map[env],
                              dataset_project="fewnerd_pipeline").get_local_copy()

    print(f"Processing dataset {env}")
    index = f"fewnerd_v4_{env}"
    await ensure_existing_index(index, mapping)

    with open(os.path.join(dataset_dir, dataset["json"])) as file:
        for document in tqdm(ijson.items(file, "item")):
            await write_to_index(document, index)



print('Notice, artifacts are uploaded in the background')
print('Done')

async def main():
    tasks = [index_task(dataset) for dataset in fewnerd_dataset.datasets]
    await asyncio.gather(*tasks)
    await es.close()


if __name__ == "__main__":

    runtime_args = RuntimeArgs()
    auth = (runtime_args.elasticsearch_user, runtime_args.elasticsearch_password)
    es = AsyncElasticsearch(hosts=runtime_args.elasticsearch_host,
                            verify_certs=False,
                            max_retries=10,
                            request_timeout=30,
                            retry_on_timeout=True,
                            basic_auth=auth)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()

