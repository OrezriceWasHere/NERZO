import asyncio
from copy import deepcopy

import torch
from tqdm import tqdm
import json
import dataset_provider



class EmbedderEnricher:

	def __init__(self, embedding_field, elastic_index,text_field):
		self.embedding_field = embedding_field
		self.elastic_index = elastic_index
		self.text_field = text_field
		self.pbar = None


	async def load_to_queue(
			self,
			queue: asyncio.Queue,
			batch_size: int = 10,
			fields_to_sort: list[str] | None = None,
			filter_query: str | None = None,
	):
		filter_query = filter_query or {"match_all": {}}
		fields_to_sort = fields_to_sort or [{"text_id":{"order":"desc"}}, {"doc_id":{"order":"desc"}}]
		search_query = {
			"query": {
				"bool": {
					"must": [
						filter_query,
					],
					"must_not": [
						{"exists": {"field": self.embedding_field}},
					]
				}
			},
			"_source": [self.text_field, "index_start", "index_end"],
			"sort":  fields_to_sort,
			"size": batch_size
		}

		count_query = {
			"query": {
				"bool": {
					"must": [
						filter_query,
					],
					"must_not": [
						{"exists": {"field": self.embedding_field}},
					]
				}
			}
		}

		total_count = await dataset_provider.count(self.elastic_index, count_query)
		self.pbar = tqdm(total=total_count)
		if total_count > 0:
			async for result in dataset_provider.consume_big_query(search_query, self.elastic_index):
				await queue.put(result)
			await queue.put(None)


	def generate_text_to_embedding_dict(self, batch) -> dict[str, torch.Tensor]:
		raise NotImplementedError()

	def generate_embedding_batch(self, batch):
		bulk = []
		text_to_embedding = self.generate_text_to_embedding_dict(batch)
		for doc in batch:
			embedding = text_to_embedding[doc["_source"][self.text_field]]
			bulk.append({"update": {"_index": doc["_index"], "_id": doc["_id"]}})
			bulk.append({"doc": {self.embedding_field: embedding.tolist()}, "doc_as_upsert": True})
			# batch.append({"update": {"_index": index, "_id": doc_id}})
			# batch.append({"doc": {**record, "doc_id": doc_id}, "doc_as_upsert": True})

		del text_to_embedding
		return bulk


	async def generate_embedding_task(self,
			queue: asyncio.Queue):
		while True:
			batch = await queue.get()
			if batch is None:
				break
			bulk = self.generate_embedding_batch(batch)
			response = await dataset_provider.bulk(bulk)
			self.pbar.update(len(bulk))
			del batch
			del bulk
