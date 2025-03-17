import abc
import asyncio
import random
from collections import defaultdict
import pandas as pd
import torch
from tqdm import tqdm
import clearml_helper
import dataset_provider
import fewnerd_processor
import clearml_poc
from clearml_pipelines.fewnerd_pipeline import fewnerd_dataset
from clearml_pipelines.nertreieve_dataset import nertrieve_processor
from contrastive.args import FineTuneLLM, Arguments


class RetrievalEval(abc.ABC):


	def __init__(self, index, layer, entity_to_embedding: dict[str, torch.Tensor | list[float]]):
		self.loop = asyncio.get_event_loop()
		self.index = index
		self.layer = layer
		self.embedding_per_type = entity_to_embedding
		self.all_test_types = list(entity_to_embedding.keys())
		self.entity_field = self.entity_type_field_name()
		self.semaphore = asyncio.Semaphore(10)
		self.text_id_to_labels = nertrieve_processor.text_id_to_labels()
		self.entity_to_name = nertrieve_processor.type_to_name()

	def eval_zero_shot(self):
		zero_shot_tasks = self.generate_zero_shot_task()
		self.pbar = tqdm(total=len(zero_shot_tasks) * 3)
		self.execute_tasks(zero_shot_tasks, series='zero shot')

	def eval_one_shot(self):
		one_shot_tasks = self.generate_one_shot_task()
		self.pbar = tqdm(total=len(one_shot_tasks) * 3 * 3)
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
		sizes = [10, 50, count_type]
		for doc_id in ids:
			document = await self.get_item_by_id(doc_id, source=[embedding_field])
			embedding = document[embedding_field]
			async with self.semaphore:
				similar_doc = await self.search_similar_items(embedding, count_type + 1, entity_type)

			for k in sizes:
				top_k_results = self.top_k_ids(similar_doc, k + 1, pop_max=True)
				assert len(top_k_results) == k - 1, f"could not fetch {k} docs for {entity_type}"
				hits = [1 if entity_type in similar_doc[text_id]["labels"]
				        else 0 for text_id in top_k_results]
				recall = sum(hits) / k
				size_desc = k if k in [10, 50] else "size"
				result[f"recall@{size_desc}"].append(recall)
				self.pbar.update(1)
			result["size"] = count_type
		return result

	def top_k_ids(self, data, k, pop_max=False):
		top_k = sorted(data.items(), key=lambda x: x[1]["score"], reverse=True)
		if pop_max:
			top_k = top_k[1:k]
		else:
			top_k = top_k[:k]
		return [key for key, _ in top_k]

	async def zero_shot_task(self, entity_type):
		count_type = await self.get_count_entity_type(entity_type)
		assert count_type > 0, "no count for entity type {}".format(entity_type)
		result = {}
		sizes = [10, 50, count_type]
		embedding = self.embedding_per_type[entity_type]
		async with self.semaphore:
			similar_doc = await self.search_similar_items(embedding, count_type, entity_type)

		for k in sizes:
			top_k_results = self.top_k_ids(similar_doc, k, pop_max=False)
			assert len(top_k_results) == k, f"could not fetch {k} docs for {entity_type}"
			hits = [1 if entity_type in similar_doc[text_id]["labels"]
			        else 0 for text_id in top_k_results]
			recall = sum(hits) / k
			size_desc = k if k in [10, 50] else "size"
			result[f"recall@{size_desc}"] = recall
			self.pbar.update(1)
		result["size"] = count_type
		return result

	async def get_item_by_id(self, doc_id, **kwargs):
		results = await dataset_provider.get_by_id(index_name=self.index, doc_id=doc_id)
		return results["_source"]

	# async def recall_by_id(self, entity_type, fewnerd_id, k):
	# 	embedding_field = self.get_embedding_field_name()
	#
	# 	document = await self.get_item_by_id(fewnerd_id, source=[embedding_field])
	# 	embedding = document["_source"][embedding_field]
	# 	similar_items = await self.search_similar_items(embedding, k, )
	# 	returned_fine_types = [1 if item["_source"][self.entity_field] == entity_type else 0 for item in similar_items]
	# 	return recall_score(y_true=[1] * k, y_pred=returned_fine_types)

	async def recall_by_type(self, entity_type, k):
		key = self.entity_type_field_name()

	# return recall_score(y_true=[1] * k, y_pred=returned_fine_types)

	def get_embedding_field_name(self):
		return f'embedding.{self.layer}'

	async def search_similar_items(self, embedding, k, entity_type):
		score_list = defaultdict(float)

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
		# query = {
		# 	"size": min(3 * k, 5000),
		#
		# 	"query": {
		# 		"bool": {
		# 			"must": [
		# 				{"match_all": {}}
		# 			],
		# 			"should": [
		# 				{
		# 					"function_score": {
		# 						"query": {
		# 							"match": {
		# 								"all_text": entity_type,
		# 							}
		# 						},
		# 						"boost_mode": "replace"
		# 					}
		# 				}
		# 			]
		# 		}
		#
		# 	},
		#
		# 	"_source": ["text_id", "fine_type", "entity_type"],
		# 	"sort": [
		# 		{"_score": "desc"},
		# 		{"doc_id": "asc"}
		# 	]
		#
		# }
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

	# query_get_entity_type = {
	# 	"size": 3 * len(unique_ids),
	# 	"query":{
	# 		"terms":{
	# 			"text_id": list(unique_ids)
	# 		}
	# 	},
	# 	"_source": ["text_id", self.entity_type_field_name()],
	# 	"sort": ["text_id"]
	# }
	# answer = defaultdict(list)
	# async for batch in dataset_provider.consume_big_query(query=query_get_entity_type, index=self.index):
	# 	for item in batch:
	# 		answer[item["_source"]["text_id"]].append(item)
	# return answer

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
					"cardinality": {
						"field": "text_id"
					}
				}
			}
		}
		x =  await dataset_provider.search_async(index=self.index, query=query)
		return x["aggregations"]["unique_text_ids"]["value"]

	def anchors(self):
		raise NotImplementedError()

	def entity_type_field_name(self):
		raise NotImplementedError()


if __name__ == "__main__":
	clearml_poc.clearml_init(task_name="calculate recall")

	layer_obj = {
		"layer_id": "de8cbfe796714725930af567f488230f",
		"llm_layer": FineTuneLLM.layer,
		"llm_id": FineTuneLLM.llm_id
	}
	clearml_poc.clearml_connect_hyperparams(layer_obj, name="layer")
	layer = layer_obj["layer_id"]
	llm_layer = layer_obj["llm_layer"]
	llm_id = layer_obj["llm_id"]
	index = "fewnerd_tests"

	all_test_types = anchors()

	embedding_per_type = type_to_tensor = fewnerd_processor.load_entity_name_embeddings(
		layer_name=fewnerd_dataset.llm_and_layer_to_elastic_name(
			llm_id=llm_id,
			layer=llm_layer
		),
		entity_name_strategy=Arguments.entity_name_embeddings
	)
	print('passing test in mlp')
	similarity_mlp = clearml_helper.get_mlp_by_id(layer)
	embedding_per_type = {key: similarity_mlp(value).detach().tolist()
	                      for key, value in embedding_per_type.items()
	                      if key in fewnerd_processor.test_fine_types()
	                      }

	one_shot_tasks = [handle_type_one_shot(type, ids) for type, ids in all_test_types.items()]

	llm_args = ()

	main(one_shot_tasks, series="one shot")

	zero_shot_tasks = [zero_shot_task(type) for type in all_test_types.keys()]
	main(zero_shot_tasks, series="zero shot")
