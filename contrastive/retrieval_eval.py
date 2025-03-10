import abc
import asyncio
from collections import defaultdict
import pandas as pd
import torch
from sklearn.metrics import recall_score
from tqdm import tqdm

import clearml_helper
import dataset_provider
import fewnerd_processor
import clearml_poc
from clearml_pipelines.fewnerd_pipeline import fewnerd_dataset
from clearml_pipelines.nertreieve_dataset import neretrieve_dataset, nertrieve_processor
from contrastive.args import FineTuneLLM, Arguments


class RetrievalEval(abc.ABC):


	def __init__(self, index, layer, entity_to_embedding: dict[str, torch.Tensor]):
		self.loop = asyncio.get_event_loop()
		self.index = index
		self.layer = layer
		self.embedding_per_type = entity_to_embedding
		self.all_test_types = list(entity_to_embedding.keys())
		self.entity_field = self.entity_type_field_name()
		self.semaphore = asyncio.Semaphore(10)
		self.text_id_to_labels = nertrieve_processor.text_id_to_labels()

	def eval_zero_shot(self):
		zero_shot_tasks = self.generate_zero_shot_task()
		self.pbar = tqdm(total=len(zero_shot_tasks) * 3)
		self.execute_tasks(zero_shot_tasks, series='zero shot')

	def generate_zero_shot_task(self):
		all_types = self.all_test_types
		zero_shot_tasks = [self.zero_shot_task(type) for type in all_types]
		return zero_shot_tasks

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

	# async def handle_type_one_shot(self, entity_type, ids):
	# 	count_type = await self.get_count_entity_type(entity_type)
	# 	result = {}
	# 	for k in (10, 50, count_type):
	# 		recall_list = []
	# 		for fewnerd_id in ids:
	# 			recall = await self.recall_by_id(entity_type, fewnerd_id, k)
	# 			recall_list.append(recall)
	# 		size_desc = k if k in [10, 50] else "size"
	# 		result[f"recall@{size_desc}"] = sum(recall_list) / len(recall_list)
	# 	result["size"] = count_type
	# 	return result

	async def zero_shot_task(self, entity_type):
		async with self.semaphore:
			count_type = await self.get_count_entity_type(entity_type)
			result = {}
			for k in (10, 50, count_type):
				recall_list = []
				recall = await self.recall_by_type(entity_type, k)
				recall_list.append(recall)
				size_desc = k if k in [10, 50] else "size"
				result[f"recall@{size_desc}"] = sum(recall_list) / len(recall_list)
				self.pbar.update(1)
			result["size"] = count_type
		return result

	async def get_item_by_id(self, doc_id, **kwargs):
		results = await dataset_provider.get_by_id(index_name=self.index, doc_id=doc_id)
		return results["_source"]

	# async def recall_by_id(self, entity_type, fewnerd_id, k):
	# 	document = await self.get_item_by_id(fewnerd_id, source=[f"embedding.{self.layer}"])
	# 	embedding = document["embedding"][self.layer]
	# 	similar_items = await self.search_similar_items(embedding, k)
	# 	returned_fine_types = [1 if item["_source"][self.entity_field] == entity_type else 0 for item in similar_items]
	# 	return recall_score(y_true=[1] * k, y_pred=returned_fine_types)

	async def recall_by_type(self, entity_type, k):
		key = self.entity_type_field_name()
		embedding = self.embedding_per_type[entity_type]

		similar_doc = await self.search_similar_items(embedding, k, entity_type)
		returned_fine_types = [1 if entity_type in similar_doc[text_id]
		                       else 0 for text_id in similar_doc]
		return recall_score(y_true=[1] * k, y_pred=returned_fine_types)

	def get_embedding_field_name(self):
		return f'embedding.{self.layer}'

	async def search_similar_items(self, embedding, k, entity_type):
		answer = defaultdict(list)
		unique_ids = set()
		count = 0

		embedding_field = self.get_embedding_field_name()
		query = {
			"size":1,
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
        { "_score": "desc" },
        { "doc_id": "asc" }
    ]
		}
		async for batch in dataset_provider.consume_big_query(query=query, index=self.index):
			if len(unique_ids) == k:
				break
			for item in batch:
				unique_ids.add(item["_source"]["text_id"])
				if len(unique_ids) == k:
					break


		return {text_id:self.text_id_to_labels[text_id] for text_id in unique_ids}
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
			"query": {
				"bool": {
					"must": [
						{"term": {self.entity_field: entity_type}},
						{"exists": {"field": embedding_filed}}
					]
				}
			}
		}
		return await dataset_provider.count(index_name=self.index, query=query)

	# def anchors(self):
	# 	raise NotImplementedError()

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
