import os
import hashlib
import ijson
from clearml import Dataset
from tqdm.asyncio import tqdm
import clearml_poc
import urllib3
import asyncio
import dataset_provider
from clearml_pipelines.nertreieve_dataset import neretrieve_dataset
from contrastive.args import Arguments

urllib3.disable_warnings()

# Connecting ClearML with the current process,
# from here on everything is logged automatically


mapping = neretrieve_dataset.elasticsearch_storage_mapping





def generate_id(item):
    keys = ["all_text", "text_id", "phrase", "entity_type", "entity_id"]
    hash_v = hashlib.sha1(str.encode("".join([str(item[key]) for key in keys]))).hexdigest()
    return f"nert_{hash_v}"

async def write_to_elastic_worker(queue):
    pbar = tqdm()
    batch = []
    while True:
        record, index = await queue.get()
        if record is None:
            break
        batch.append((record, index))
        if len(batch) == BATCH_SIZE:
            await write_batch(batch)
            batch = []
        pbar.update(len(batch))
    if batch:
        await write_batch(batch)


async def write_batch(bulk):
    batch = []
    for record, index in bulk:
        doc_id = generate_id(record)
        batch.append({"update": {"_index": index, "_id": doc_id}})
        batch.append({"doc": {**record}, "doc_as_upsert": True})
        # batch.append({
        #     '_id': doc_id,
        #     '_op_type': 'update',
        #     '_index': index,
        #     'doc': {**record, "doc_id": doc_id},
        # })
    x = await dataset_provider.bulk(batch)
    pass

async def write_to_index(data, index):
    data["doc_id"] = generate_id(data)
    body = {
        "doc": data,
        "doc_as_upsert": True
    }
    response = await dataset_provider.upsert(index=index, doc_id=data["doc_id"], data=body)
    return response


async def load_json_task(dataset):
    env = dataset["env"]
    # Load the artifact
    dataset_dir = Dataset.get(dataset_name=dataset["json"],
                              dataset_tags=[dataset_tag_obj["dataset_tag"]],
                              dataset_project="neretrieve_pipeline").get_local_copy()

    print(f"Processing dataset {env}")
    index = f"nertrieve_{env}"
    await dataset_provider.ensure_existing_index(index, mapping)

    with open(os.path.join(dataset_dir, dataset["json"])) as file:
        for document in ijson.items(file, "item", use_float=True):
            await queue.put((document, index))

async def main():
    await asyncio.gather(
        load_json_task(neretrieve_dataset.datasets[0]),
        write_to_elastic_worker(queue),
    )



if __name__ == "__main__":
    clearml_poc.clearml_init(
        project_name="neretrieve_pipeline",
        task_name="Pipeline step 3 index to database",
        requirements=["aiohttp"]
    )

    BATCH_SIZE = 1024
    args = Arguments()
    dataset_tag_obj = {"dataset_tag": args.llm_layer}
    clearml_poc.clearml_connect_hyperparams(dataset_tag_obj)
    queue = asyncio.Queue(maxsize=10000)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())




    loop.close()
