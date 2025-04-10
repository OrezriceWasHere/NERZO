import abc
import asyncio
import random
from abc import abstractclassmethod

from sklearn.metrics import ndcg_score
from collections import defaultdict
import pandas as pd
import torch
from tqdm import tqdm
import clearml_helper
import dataset_provider
import contrastive.fewnerd_processor
import clearml_poc
from clearml_pipelines.fewnerd_pipeline import fewnerd_dataset
from clearml_pipelines.nertreieve_dataset import nertrieve_processor
from contrastive.args import FineTuneLLM, Arguments


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k  # Construct new key
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())  # Recursively flatten
        else:
            items.append((new_key, v))  # Store key-value pair
    return dict(items)

class RetrievalEval(abc.ABC):


	def __init__(self, index, layer, entity_to_embedding: dict[str, torch.Tensor | list[float]], is_bm25=False):
		self.loop = asyncio.get_event_loop()
		self.index = index
		self.layer = layer
		self.embedding_per_type = entity_to_embedding
		self.all_test_types = list(entity_to_embedding.keys())
		self.entity_field = self.entity_type_field_name()
		self.semaphore = asyncio.Semaphore(20)
		self.text_id_to_labels = self.calc_text_id_to_labels()
		self.is_bm25 = is_bm25


	def eval_zero_shot(self):
		zero_shot_tasks = self.generate_zero_shot_task()
		self.pbar = tqdm(total=len(zero_shot_tasks))
		self.execute_tasks(zero_shot_tasks, series='zero shot')

	def eval_one_shot(self):
		one_shot_tasks = self.generate_one_shot_task()
		total_amount = sum([len(x) for x in self.anchors().values()])
		self.pbar = tqdm(total=total_amount)
		self.execute_tasks(one_shot_tasks, series='one shot')


	def generate_zero_shot_task(self):
		all_types = self.all_test_types
		zero_shot_tasks = [self.zero_shot_task(type) for type in all_types]
		return zero_shot_tasks

	def generate_one_shot_task(self):
		anchors = self.anchors()
		one_shot_tasks = [self.handle_type_one_shot(entity_type, doc_ids) for entity_type, doc_ids in anchors.items()]
		return one_shot_tasks

	def execute_tasks(self, evaluation_tasks, series):
		loop = self.loop
		result = defaultdict(list)
		task_results = loop.run_until_complete(asyncio.gather(*evaluation_tasks))
		for entity_type, task_result in zip(self.all_test_types, task_results):
			for result_key, result_value in task_result.items():
				result[result_key].append(result_value)
			result["index"].append(entity_type)

		index_column = result.pop("index")
		df = pd.DataFrame(data=result, index=index_column)
		clearml_poc.add_table(
			iteration=0,
			series=series,
			title="recall per fine type",
			table=df
		)

		clearml_poc.add_table(
			iteration=0,
			series=series,
			title="average recall",
			table=df.mean().to_frame()
		)

	async def handle_type_one_shot(self, entity_type, ids):
		count_type = await self.get_count_entity_type(entity_type)
		embedding_field = self.get_embedding_field_name()
		assert count_type > 0, "no count for entity type {}".format(entity_type)
		result = defaultdict(list)
		sizes = [10, 50, 100,  200, 500, count_type]
		descriptions = ["10", "50", "100", "200", "500", "size"]
		for doc_id in ids:
			document = await self.get_item_by_id(doc_id)
			embedding = flatten_dict(document)[embedding_field]
			phrase = document["phrase"]
			async with self.semaphore:
				similar_doc = await self.search_similar_items(embedding, count_type + 1, entity_type, phrase)
			similar_doc = sorted(similar_doc.items(), key=lambda x: x[1]["score"], reverse=True)
			for size, size_desc in zip(sizes, descriptions):
				k = min(size, count_type)
				size_eval = self.evaluate(
					similar_doc,
					k,
					size_desc,
					entity_type,
					count_type,
					pop_max=False
				)
				for key, metric in size_eval.items():
					result[key].append(metric)

		result["size"] = count_type
		self.pbar.update(1)
		avg_result = {
			key: (sum(values) / len(values) ) if isinstance(values, list) else values
			for key, values in result.items()
		}
		return avg_result

	async def zero_shot_task(self, entity_type):
		count_type = await self.get_count_entity_type(entity_type)
		assert count_type > 0, "no count for entity type {}".format(entity_type)
		result = {}
		sizes = [10, 50, 100,  200, 500, count_type]
		descriptions = ["10", "50", "100", "200", "500", "size"]
		embedding = self.embedding_per_type[entity_type]
		async with self.semaphore:
			similar_doc = await self.search_similar_items(embedding, count_type, entity_type)
		similar_doc = sorted(similar_doc.items(), key=lambda x: x[1]["score"], reverse=True)

		for size, size_desc in zip(sizes, descriptions):
			k = min(size, count_type)
			size_eval = self.evaluate(
				similar_doc,
				k,
				size_desc,
				entity_type,
				count_type,
				pop_max=False
			)
			result.update(size_eval)


		self.pbar.update(1)
		result["size"] = count_type
		return result

	def evaluate(self, similar_doc, k, size_desc, entity_type,  count_type, pop_max=False):
		result = {}
		top_k_results = similar_doc[:k] if not pop_max else similar_doc[1: k + 1]
		assert len(top_k_results) == k, f"could not fetch {size_desc} docs for {entity_type}"

		hits = [1 if entity_type in results["labels"] else 0 for text_id, results in top_k_results]
		cosine_similarities = [results["score"] - 1.0 for text_id, results in top_k_results]

		recall = sum(hits) / count_type
		result[f"recall@{size_desc}"] = recall
		precision = sum(hits) / k
		result[f"precision@{size_desc}"] = precision
		result[f"nDSG@{size_desc}"] = ndcg_score([hits], [cosine_similarities], k=k)

		return result

	async def get_item_by_id(self, doc_id, **kwargs):
		query = {"query": {"term": {"doc_id": doc_id}}}

		results = await dataset_provider.search_async(index=self.index, query=query, **kwargs)
		assert len(results["hits"]["hits"]) == 1
		return results["hits"]["hits"][0]["_source"]

	def get_embedding_field_name(self):
		return f'embedding.{self.layer}'

	def vector_query(self, embedding, k):
		embedding_field = self.get_embedding_field_name()
		query = {
			"size": min(3 * k, 10000),
			"query": {
				"script_score": {
					"query": {"match_all": {}},
					"script": {
						f"source": f"cosineSimilarity(params.query_vector, '{embedding_field}') + 1.0",
						"params": {"query_vector": embedding}
					}
				}
			},
			"_source": ["text_id", "fine_type", "entity_type"],
			"sort": [
				{"_score": "desc"},
				{"doc_id": "asc"}
			]
		}
		return query

	def bm25_query(self, k, entity_type, phrase):
		text_to_look_by = phrase or entity_type
		query = {
			"size": min(3 * k, 5000),

			"query": {
				"bool": {
					"must": [
						{"match_all": {}}
					],
					"should": [
						{
							"function_score": {
								"query": {
									"match": {
										"all_text": text_to_look_by,
									}
								},
								"boost_mode": "replace"
							}
						}
					]
				}

			},

			"_source": ["text_id", "fine_type", "entity_type"],
			"sort": [
				{"_score": "desc"},
				{"doc_id": "asc"}
			]

		}
		return query

	async def search_similar_items(self, embedding, k, entity_type, phrase=None):
		score_list = defaultdict(float)

		if self.is_bm25:
			query = self.bm25_query(k, entity_type, phrase)
		else:
			query = self.vector_query(embedding, k)

		async for batch in dataset_provider.consume_big_query(query=query, index=self.index):
			if len(score_list) == k:
				break
			for item in batch:
				text_id = item["_source"]["text_id"]
				if len(score_list) == k:
					break
				score_list[text_id] = max(score_list[text_id], item["_score"])

		return {text_id: {"labels": self.text_id_to_labels[text_id], "score": score_list[text_id]} for text_id in
		        score_list}


	async def get_count_entity_type(self, entity_type):
		embedding_filed = self.get_embedding_field_name()
		query = {
		  "size": 0,
		  "query": {
						"bool": {
							"must": [
								{"term": {self.entity_field: entity_type}},
								{"exists": {"field": embedding_filed}}
							]
						}
					},
		  "aggs": {
		    "unique_text_ids": {
		      "scripted_metric": {
		        "init_script": "state.text_ids = new HashSet();",
		        "map_script": "state.text_ids.add(doc['text_id'].value);",
		        "combine_script": "return state.text_ids;",
		        "reduce_script": "Set all_ids = new HashSet(); for (s in states) { all_ids.addAll(s); } return all_ids.size();"
		      }
		    }
		  }
		}
		x =  await dataset_provider.search_async(index=self.index, query=query)
		return x["aggregations"]["unique_text_ids"]["value"]

	@abc.abstractmethod
	def anchors(self):
		raise NotImplementedError()

	@abc.abstractmethod
	def entity_type_field_name(self):
		raise NotImplementedError()

	@abc.abstractmethod
	def calc_text_id_to_labels(self):
		raise NotImplementedError()

