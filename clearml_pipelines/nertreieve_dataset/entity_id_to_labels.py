import json
import os
import hashlib
import aiofiles
from clearml import Dataset
from tqdm.asyncio import tqdm
import clearml_poc
import urllib3
import asyncio
import dataset_provider
from clearml_pipelines.nertreieve_dataset import neretrieve_dataset
from contrastive.args import Arguments
from collections import defaultdict

urllib3.disable_warnings()

# Connecting ClearML with the current process,
# from here on everything is logged automatically




async def mapping_entity_id_to_all_possible_labels(queue):
	pbar = tqdm()
	entity_id_to_labels = defaultdict(list)
	while True:
		record = await queue.get()
		if record is None:
			break
		entity_id_to_labels[record['text_id']].append(record["entity_type"])
		pbar.update(1)
	file_name = 'text_id_to_labels.json'
	async with aiofiles.open(file_name, "w") as f:
		await f.write(json.dumps(entity_id_to_labels))
	clearml_dataset = Dataset.create(dataset_name=file_name, dataset_project="neretrieve_pipeline")
	clearml_dataset.add_files(path=file_name)
	clearml_dataset.add_tags(['entity_to_labels'])
	clearml_dataset.finalize()



async def load_json_task(dataset, queue):
	print(f"Processing dataset test")
	index = f"nertrieve_test"
	query = {
		"query": {"match_all": {}},
		"sort": ["text_id"],
		"_source":["text_id", "entity_type"],
		"size": 5000,
	}
	async for batch in dataset_provider.consume_big_query(query=query, index=index):
		for record in batch:
			await queue.put(record["_source"])


	# Signal completion to all workers
	await queue.put(None)


async def main():
	# Create multiple worker queues and tasks
	queue = asyncio.Queue(maxsize=10000)
	worker_task = mapping_entity_id_to_all_possible_labels(queue)

	# Gather all tasks
	await asyncio.gather(
		load_json_task(neretrieve_dataset.datasets[0], queue),
		worker_task
	)


if __name__ == "__main__":
	clearml_poc.clearml_init(
		project_name="neretrieve_pipeline",
		task_name="text id to all labels",
		requirements=["aiohttp", "aiofiles"],
		queue_name='dsicsgpu'
	)
	args = Arguments()
	dataset_tag_obj = {"dataset_tag": args.llm_layer}


	# Configuration constants
	BATCH_SIZE = 2048

	args = Arguments()

	loop = asyncio.get_event_loop()
	loop.run_until_complete(main())
	loop.close()