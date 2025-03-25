import json
import os
import hashlib
from time import sleep

import aiofiles
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
	keys = ["text_id", "phrase", "entity_type", "entity_id", "index_start", "index_end"]
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
			pbar.update(BATCH_SIZE)
	if batch:
		await write_batch(batch)
		pbar.update(len(batch))


async def write_batch(bulk, id_generator=generate_id):
	batch = []
	for record, index in bulk:
		doc_id = id_generator(record)
		batch.append({"update": {"_index": index, "_id": doc_id}})
		batch.append({"doc": {**record, "doc_id": doc_id}, "doc_as_upsert": True})
	x = await dataset_provider.bulk(batch)
	return x


async def load_json_task(dataset, queue):
	env = dataset["env"]
	# Load the artifact
	dataset_dir = Dataset.get(
		dataset_name=dataset["json"],
		dataset_tags=[dataset_tag_obj["dataset_tag"]],
		dataset_project="neretrieve_pipeline"
	).get_local_copy()
	# dataset_dir = "/home/orsh/.clearml/venvs-builds/3.11/task_repository/NERZO.git/"

	print(f"Processing dataset {env}")
	index = f"nertrieve_{env}"
	await dataset_provider.ensure_existing_index(index, mapping)

	file_path = os.path.join(dataset_dir, dataset["json"])
	async with aiofiles.open(file_path, mode='r') as file:
		async for line in file:
			document = json.loads(line)

			await queue.put((document, index))

		# Signal completion to all workers
		await queue.put((None, index))


async def main():
	# Create multiple worker queues and tasks
	queue = asyncio.Queue(maxsize=10000)
	worker_task = write_to_elastic_worker(queue)

	dataset = {
		"url": "https://storage.googleapis.com/neretrieve_dataset/IR/NERetrive_IR_train.jsonl.bz2",
		"name": "retrieval-train-supervised.txt",
		"json": "retrieval-train-supervised.json",
		"env": "train"
	}

	# Gather all tasks
	await asyncio.gather(
		load_json_task(dataset, queue),
		worker_task
	)


if __name__ == "__main__":
	clearml_poc.clearml_init(
		project_name="neretrieve_pipeline",
		task_name="Pipeline step 3 index to database",
		requirements=["aiohttp", "aiofiles"],
		queue_name='dsicsgpu'
	)


	# Configuration constants
	BATCH_SIZE = 2048

	args = Arguments()
	dataset_tag_obj = {"dataset_tag": args.llm_layer}
	clearml_poc.clearml_connect_hyperparams(dataset_tag_obj)

	loop = asyncio.get_event_loop()
	loop.run_until_complete(main())
	loop.close()
