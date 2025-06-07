import asyncio

from clearml import Dataset
from datasets import tqdm

import clearml_poc
import dataset_provider

async def fetch_all_doc_id_from_index(index) -> set[str]:
	"""
	Fetch all document IDs from the given index.
	"""
	pbar = tqdm(f"fetching from index {index}")
	doc_ids = set()
	query = {
		"size": 10000,
		"_source": ["doc_id"],
		"query": {
			"match_all": {}
		},
		"sort": [
			{"doc_id": "asc"}
		]
	}

	async for batch in dataset_provider.consume_big_query(query=query, index=index):
		for item in batch:
			doc_ids.add(item["_source"]["doc_id"])
		pbar.update(len(batch))

	return doc_ids




if __name__ == '__main__':
	clearml_poc.clearml_init(task_name='fetching left fewnerd ids', queue_name='dsicsgpu')
	loop = asyncio.get_event_loop()
	tasks =[
		fetch_all_doc_id_from_index("fewnerd_v4_*"),
		fetch_all_doc_id_from_index("fewnerd_analyzer"),
	]
	doc_ids_all, doc_ids_passed = loop.run_until_complete(asyncio.gather(*tasks))
	doc_ids_remaining = doc_ids_all - doc_ids_passed
	import json
	with open("fewnerd_doc_ids_remaining.json", "w") as f:
		json.dump(list(doc_ids_remaining), f, indent=4)

	dataset = Dataset.create(
		dataset_name="fewnerd_doc_ids_remaining",
		dataset_tags=["fewnerd_doc_ids_remaining"],
		dataset_project="fewnerd_pipeline",
		dataset_version="1.0.0",
	)
	dataset.add_files("fewnerd_doc_ids_remaining.json")
	dataset.upload()
	dataset.finalize()
