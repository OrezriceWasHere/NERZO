import asyncio
from dataclasses import asdict
import torch
import clearml_poc
import dataset_provider
from clearml_pipelines.general.elastic_async_helpers import EmbedderEnricher
from contrastive.args import FineTuneLLM
from llm_interface import LLMInterface


class LLama317SentenceEnricher(EmbedderEnricher):


	def __init__(self, embedding_field, text_field, elastic_index, sentence_embedder_id, layer):
		super().__init__(
			elastic_index=elastic_index,
			embedding_field=embedding_field,
			text_field=text_field
		)

		self.model = LLMInterface(
			llm_id=sentence_embedder_id,
			max_llm_layer=18,
			interested_layers=[layer]
		)
		self.layer = layer
		self.db_name = "llama_3_17_v_proj"


	def generate_embedding_batch(self, batch):
		eos_token = self.model.tokenizer.eos_token
		texts = list(set([item["_source"][self.text_field] + eos_token for item in batch]))
		tokens = self.model.tokenize(texts).to('cuda')
		embeddings = self.model.get_llm_at_layer(tokens, layer=self.layer)
		text_to_embedding = {text: encoding for text, encoding in zip(texts, embeddings)}
		bulk = []
		for doc in batch:
			text = doc["_source"][self.text_field] + eos_token
			indices = (doc["_source"]["index_start"], doc["_source"]["index_end"])
			h = text_to_embedding[text]
			llm_indices = self.model.token_indices_given_text_indices(text, indices)
			index_of_eos = self.model.tokens_count(text) - 1
			start = h[llm_indices[0] - 1]
			end = h[llm_indices[1]]
			eos = h[index_of_eos]
			result = {
				"embedding": {
					self.db_name: {
						"start": start.tolist(),
						"end": end.tolist(),
						"eos": eos.tolist()
					}
				}
			}

			bulk.append({"update": {"_index": doc["_index"], "_id": doc["_id"]}})
			bulk.append({"doc": result})
		return bulk


def forward_sentence_embedder_in_index(
		elastic_index: str,
		text_field_name: str,
		embedding_field_name,
		llm_id: str,
		layer: str,
		fields_to_sort: list[str] | None = None,
		filter_query: str | None = None,

):
	loop = asyncio.get_event_loop()
	sentence_embedder_enricher = LLama317SentenceEnricher(
		embedding_field=embedding_field_name,
		text_field=text_field_name,
		elastic_index=elastic_index,
		sentence_embedder_id=llm_id,
		layer=layer
	)
	embedding_field = {
		"embedding": {
			"properties": {
				"llama_3_17_v_proj": {
					"properties": {
						"eos": {
							"type": "dense_vector",
							"dims": 1024,
							"index": False
						}
					}
				}
			}
		}

	}
	loop.run_until_complete(dataset_provider.ensure_field(index_name=elastic_index, field_mapping=embedding_field))
	queue = asyncio.Queue(maxsize=10000)
	BATCH_SIZE = 200

	for _ in range(5):
		loop.run_until_complete(
			asyncio.gather(
				sentence_embedder_enricher.load_to_queue(
					queue=queue,
					batch_size=BATCH_SIZE,
					fields_to_sort=fields_to_sort,
					filter_query=filter_query
				),
				sentence_embedder_enricher.generate_embedding_task(
					queue=queue
				)
			)
		)
	del sentence_embedder_enricher

	pass


def executing_pipeline(
		dataset_index: str,
		naming_index: str,
		llm_id: str,
		layer: str,
		**kwargs
):
	assert torch.cuda.is_available(), "no cuda"
	embedding_field_name = "embedding.llama_3_17_v_proj.eos"
	forward_sentence_embedder_in_index(
		elastic_index=dataset_index,
		embedding_field_name=embedding_field_name,
		llm_id=llm_id,
		layer=layer,
		text_field_name="all_text"
	)


if __name__ == "__main__":
	clearml_poc.clearml_init(
		project_name="sentence_embedder_pipeline",
		task_name="forward_sentence",
		queue_name="a100_gpu",
		requirements=["transformers==4.46.2", "einops==0.8.1"]
	)

	conf = {
		# "dataset_index": "nertrieve_train",
		"naming_index": "nertrieve_entity_name_to_embedding",
		"db_name": "embedding.llama_3_17_v_proj"
	}

	llm_args = asdict(FineTuneLLM())

	clearml_poc.clearml_connect_hyperparams(conf, "conf")
	clearml_poc.clearml_connect_hyperparams(llm_args, "llm_args")

	executing_pipeline(**{**conf, **llm_args})
	print("deployed")
