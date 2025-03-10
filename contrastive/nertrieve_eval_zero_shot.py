import torch.functional

import clearml_helper
import clearml_poc
from clearml_pipelines.fewnerd_pipeline import fewnerd_dataset
from contrastive import fewnerd_processor
from contrastive.args import Arguments, FineTuneLLM
from contrastive.retrieval_eval import RetrievalEval


class NERtrieveEvalZeroShot(RetrievalEval):

	def __init__(self, layer, entity_to_embedding):
		super(NERtrieveEvalZeroShot, self).__init__(
			index='nertrieve_test',
			layer=layer,
			entity_to_embedding=entity_to_embedding
		)

	def entity_type_field_name(self):
		return 'entity_type'
	#
	# def get_embedding_field_name(self):
	# 	return 'embedding.llama_3_17_v_proj.end'


if __name__ == '__main__':
	args = Arguments()
	clearml_poc.clearml_init(task_name='eval nertrieve',
	                         queue_name='dsicsgpu')

	layer_obj = {
		"layer_id": "31042a3689e340a08789b2b0cc48c71e",
		"llm_layer": FineTuneLLM.layer,
		"llm_id": FineTuneLLM.llm_id
	}
	clearml_poc.clearml_connect_hyperparams(name='eval nertrieve', hyperparams=layer_obj)
	llm_layer = layer_obj["llm_layer"]
	llm_id = layer_obj["llm_id"]
	entity_type_to_embedding = fewnerd_processor.load_entity_name_embeddings(
		layer_name=fewnerd_dataset.llm_and_layer_to_elastic_name(
			llm_id=llm_id,
			layer=llm_layer
		),
		index='nertrieve_entity_name_to_embedding',
		entity_name_strategy=Arguments.entity_name_embeddings
	)

	mlp = clearml_helper.get_mlp_by_id(layer_obj['layer_id'])
	mlp.eval()
	entity_type_to_embedding = {
		key:mlp(value).tolist()
		# key:value.tolist()

		for key, value in entity_type_to_embedding.items()
	}
	layer = layer_obj["layer_id"]
	llm_layer = layer_obj["llm_layer"]
	llm_id = layer_obj["llm_id"]
	nertrieve = NERtrieveEvalZeroShot(layer=layer, entity_to_embedding=entity_type_to_embedding)
	nertrieve.eval_zero_shot()
