import asyncio
import os
import hashlib
import ijson
from clearml import Task, Dataset
from elasticsearch import  AsyncElasticsearch
from tqdm.asyncio import tqdm

import runtime_args
from clearml_pipelines.fewnerd_pipeline import fewnerd_dataset
import urllib3

from contrastive.args import Arguments

urllib3.disable_warnings()

# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name="fewnerd_pipeline", task_name="Pipeline step 3 index to database",
                 auto_connect_streams=False,
                 reuse_last_task_id=False)

task.execute_remotely()

mapping = fewnerd_dataset.elasticsearch_storage_mapping

es = AsyncElasticsearch(**runtime_args.ElasticsearchConnection().model_dump())


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


# Iterate over all dataset, and get artifacts
# from step_download import datasets
#
env_to_version_map = {
    "dev": "1.0.6",
    "test": "1.0.4",
    "train": "1.0.3"
}


async def index_task(dataset):
    env = dataset["env"]
    # Load the artifact
    dataset_dir = Dataset.get(dataset_name=dataset["json"],
                              dataset_tags=[dataset_tag_obj["dataset_tag"]],
                              # dataset_version=env_to_version_map[env],
                              dataset_project="fewnerd_pipeline").get_local_copy()

    print(f"Processing dataset {env}")
    index = f"fewnerd_v4_{env}"
    await ensure_existing_index(index, mapping)

    with open(os.path.join(dataset_dir, dataset["json"])) as file:
        for document in tqdm(ijson.items(file, "item")):
            await write_to_index(document, index)


async def main():
    tasks = [index_task(dataset) for dataset in fewnerd_dataset.datasets]
    await asyncio.gather(*tasks)
    await es.close()


if __name__ == "__main__":
    args = Arguments()
    dataset_tag_obj = {"dataset_tag": args.llm_layer}
    task.connect(dataset_tag_obj)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
