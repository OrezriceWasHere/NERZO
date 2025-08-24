import json
import os

import torch
from clearml import Dataset
from elasticsearch import AsyncElasticsearch
from tqdm import tqdm
import clearml_helper
import dataset_provider
import fewnerd_processor
import clearml_poc
import runtime_args
from clearml_pipelines.fewnerd_pipeline import fewnerd_dataset
from contrastive.args import FineTuneLLM, Arguments
from contrastive.retrieval_eval import RetrievalEval
from llm_interface import LLMInterface


class FewnerdEvalZeroShot(RetrievalEval):

	def calc_text_id_to_labels(self):
		dataset_dir = Dataset.get(
			dataset_name="text_id_to_labels.json",
			dataset_tags=['entity_to_labels'],
			dataset_project="fewnerd_pipeline"
			).get_local_copy()

		path = os.path.join(dataset_dir, 'text_id_to_labels.json')
		with open(path, 'r') as f:
			text_to_labels = json.load(f)

		return text_to_labels

	def anchors(self):
		return fewnerd_processor.retrieval_anchors_test()

	def __init__(self, layer, entity_to_embedding):
		super(FewnerdEvalZeroShot, self).__init__(
			index='fewnerd_analyzer',
			layer=layer,
			entity_to_embedding=entity_to_embedding,
		)

	def entity_type_field_name(self):
		return 'fine_type'
	#
	# def get_embedding_field_name(self):
	# 	return 'intfloat/e5-mistral-7b-instruct@output'

if __name__ == "__main__":
	clearml_poc.clearml_init(task_name="calculate recall", queue_name="dsicsgpu")

	layer_obj = {
		"layer_id": "48d1f5c0237149aa9dedd0c028b25b3c",
		"llm_layer": "model.layers.31.self_attn.v_proj",
		"llm_id": FineTuneLLM.llm_id
	}
	clearml_poc.clearml_connect_hyperparams(layer_obj, name="layer")
	layer = layer_obj["layer_id"]
	llm_layer = layer_obj["llm_layer"]
	llm_id = layer_obj["llm_id"]
	es = AsyncElasticsearch(**runtime_args.ElasticsearchConnection().model_dump())

	print('passing test in mlp')
	similarity_mlp = clearml_helper.get_mlp_by_id(layer)
	args = clearml_helper.get_args_by_mlp_id(layer)


	entity_types = dataset_provider.search(
		query={"query": {"match_all": {}}}, index="fewnerd_entity_name_to_embedding", size=100
		)
	embedding_per_fine_type = {}
	layer_name = fewnerd_dataset.llm_and_layer_to_elastic_name(
		llm_id=llm_id,
		layer=llm_layer
	)

	for db_record in tqdm(entity_types["hits"]["hits"]):
		name = db_record["_source"]["entity_name"]
		# embedding = db_record["_source"]["meta-llama/meta-llama-3-1-8b@model-4096.end"]
		# embedding_per_fine_type[name]=embedding
		# embedding = db_record["_source"]["intfloat/e5-mistral-7b-instruct@output"]
		end_embedding = db_record["_source"]["meta-llama/meta-llama-3-1-8b@model-layers-17-self_attn-v_proj.end"]
		embedding = similarity_mlp(torch.tensor(end_embedding)).cpu().tolist()
		embedding_per_fine_type[name] = embedding


	fewnerd_eval = FewnerdEvalZeroShot(
		layer=layer,
		entity_to_embedding=embedding_per_fine_type,
	)

	fewnerd_eval.eval_zero_shot()
