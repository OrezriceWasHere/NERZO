import asyncio
from collections import defaultdict
import pandas as pd
from sklearn.metrics import recall_score
from elasticsearch import AsyncElasticsearch
import clearml_helper
import fewnerd_processor
import clearml_poc
import queries
import runtime_args
from clearml_pipelines.fewnerd_pipeline import fewnerd_dataset
from contrastive.args import FineTuneLLM, Arguments
from contrastive.retrieval_eval import RetrievalEval


class FewnerdEvalZeroShot(RetrievalEval):

	def __init__(self, layer, entity_to_embedding):
		super(FewnerdEvalZeroShot, self).__init__(
			index='fewnerd_tests',
			layer=layer,
			entity_to_embedding=entity_to_embedding
		)

	def entity_type_field_name(self):
		return 'fine_type'




def main(tasks, series):
	loop = asyncio.get_event_loop()
	result = defaultdict(list)
	task_results = loop.run_until_complete(asyncio.gather(*tasks))
	for fine_type, task_result in zip(all_test_types.keys(), task_results):
		for result_key, result_value in task_result.items():
			result[result_key].append(result_value)
		result["index"].append(fine_type)

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


async def handle_type_one_shot(fine_type, ids):
	count_type = await get_count_fine_type(fine_type)
	result = {}
	for k in (10, 50, count_type):
		recall_list = []
		for fewnerd_id in ids:
			recall = await handle_fewnerd_id_k(fine_type, fewnerd_id, k)
			recall_list.append(recall)
		size_desc = k if k in [10, 50] else "size"
		result[f"recall@{size_desc}"] = sum(recall_list) / len(recall_list)
	result["size"] = count_type
	return result


async def zero_shot_task(fine_type):
	count_type = await get_count_fine_type(fine_type)
	result = {}
	for k in (10, 50, count_type):
		recall_list = []
		recall = await handle_fewnerd_type_k(fine_type, k)
		recall_list.append(recall)
		size_desc = k if k in [10, 50] else "size"
		result[f"recall@{size_desc}"] = sum(recall_list) / len(recall_list)
	result["size"] = count_type
	return result


async def get_item_by_id(fewnerd_id, **kwargs):
	results = await es.get(index=index, id=fewnerd_id, **kwargs)
	return results["_source"]


async def handle_fewnerd_id_k(fine_type, fewnerd_id, k):
	document = await get_item_by_id(fewnerd_id, source=[f"embedding.{layer}"])
	embedding = document["embedding"][layer]
	similar_items = await search_similar_items(embedding, k)
	returned_fine_types = [1 if item["_source"]["fine_type"] == fine_type else 0 for item in similar_items]
	return recall_score(y_true=[1] * k, y_pred=returned_fine_types)


async def handle_fewnerd_type_k(fine_type, k):
	embedding = embedding_per_fine_type[fine_type]
	similar_items = await search_similar_items(embedding, k)
	returned_fine_types = [1 if item["_source"]["fine_type"] == fine_type else 0 for item in similar_items]
	return recall_score(y_true=[1] * k, y_pred=returned_fine_types)


async def search_similar_items(embedding, k):
	results = []
	remaining_items = k + 1
	query = queries.query_search_by_similarity(embedding, f'embedding.{layer}')
	query = {
		"query": query,
		"_source": [
			"fine_type"
		],
		"sort": [
			"_score", "fine_type"
		],
		"size": min(remaining_items, 10000)
	}

	response = await es.search(index=index, body=query)
	hits = response.get("hits", {}).get("hits", [])
	search_after = hits[-1]["sort"] if hits else None

	while remaining_items > 0:
		results.extend(hits)
		remaining_items -= len(hits)
		query["search_after"] = search_after
		response = await es.search(index=index, body=query)
		hits = response.get("hits", {}).get("hits", [])
		search_after = hits[-1]["sort"] if hits else None

	return results[1:k + 1]


async def get_count_fine_type(fine_type):
	query = {
		"bool": {
			"must": [
				{"term": {"fine_type": fine_type}},
				{"exists": {"field": f'embedding.{layer}'}}
			]
		}
	}
	query_results = await es.count(index=index, query=query)
	return query_results["count"]


def anchors():
	return fewnerd_processor.retrieval_anchors_test()


if __name__ == "__main__":
	clearml_poc.clearml_init(task_name="calculate recall", queue_name="dsicsgpu")

	layer_obj = {
		"layer_id": "b5f0c4909f5c40818183c4ff8fbdce59",
		"llm_layer": FineTuneLLM.layer,
		"llm_id": FineTuneLLM.llm_id
	}
	clearml_poc.clearml_connect_hyperparams(layer_obj, name="layer")
	layer = layer_obj["layer_id"]
	llm_layer = layer_obj["llm_layer"]
	llm_id = layer_obj["llm_id"]
	index = "fewnerd_tests"
	es = AsyncElasticsearch(**runtime_args.ElasticsearchConnection().model_dump())

	all_test_types = anchors()

	embedding_per_fine_type = type_to_tensor = fewnerd_processor.load_entity_name_embeddings(
		layer_name=fewnerd_dataset.llm_and_layer_to_elastic_name(
			llm_id=llm_id,
			layer=llm_layer
		),
		entity_name_strategy=Arguments.entity_name_embeddings
	)
	print('passing test in mlp')
	similarity_mlp = clearml_helper.get_mlp_by_id(layer)
	embedding_per_fine_type = {key: similarity_mlp(value).detach().tolist()
	                           for key, value in embedding_per_fine_type.items()
	                           if key in fewnerd_processor.test_fine_types()
	                           }
	#
	# one_shot_tasks = [handle_type_one_shot(type, ids) for type, ids in all_test_types.items()]
	#
	# llm_args = ()
	#
	# main(one_shot_tasks, series="one shot")

	fewnerd_eval = FewnerdEvalZeroShot(
		layer=layer,
		entity_to_embedding=embedding_per_fine_type,

	)

	# fewnerd_eval.eval_zero_shot()

	zero_shot_tasks = [zero_shot_task(type) for type in all_test_types.keys()]
	main(zero_shot_tasks, series="zero shot")
