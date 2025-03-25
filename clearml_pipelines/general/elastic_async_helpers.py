import asyncio
from copy import deepcopy

import torch
from tqdm import tqdm
import json
import dataset_provider
from sentence_embedder import SentenceEmbedder



class EmbedderEnricher:

	def __init__(self, embedding_field, elastic_index,text_field):
		self.embedding_field = embedding_field
		self.elastic_index = elastic_index
		self.text_field = text_field


	async def load_to_queue(
			self,
			queue: asyncio.Queue,
			batch_size: int = 10,
			fields_to_sort: list[str] | None = None,
			filter_query: str | None = None,
	):
		filter_query = filter_query or {"match_all": {}}
		fields_to_sort = fields_to_sort or [{"text_id":"asc"}, {"doc_id":"asc"}]
		query = {
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
			"sort": fields_to_sort,
			"size": batch_size
		}
		query_count_copy = deepcopy(query)
		query_count_copy.pop("sort")
		query_count_copy.pop("size")
		query_count_copy.pop("_source")
		total_count = await dataset_provider.count(self.elastic_index, query_count_copy)
		pbar = tqdm(total=total_count)
		async for result in dataset_provider.consume_big_query(query, self.elastic_index):
			await queue.put(result)
			pbar.update(len(result))
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
			try:
				bulk = self.generate_embedding_batch(batch)
				response = await dataset_provider.bulk(bulk)
				del batch
			except Exception as e:
				print(json.dumps({"error": str(e)}))
				print(json.dumps({"bulk": bulk}))

