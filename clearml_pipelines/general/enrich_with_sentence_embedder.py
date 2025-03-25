import asyncio

import torch
from clearml.automation.controller import PipelineDecorator
import clearml_poc
import dataset_provider
from clearml_pipelines.fewnerd_pipeline import fewnerd_dataset
from clearml_pipelines.general import elastic_async_helpers
from clearml_pipelines.general.elastic_async_helpers import EmbedderEnricher
from sentence_embedder import SentenceEmbedder


class SentenceEmbedderEnricher(EmbedderEnricher):


	def __init__(self, embedding_field, text_field, elastic_index, sentence_embedder_id):
		super().__init__(
			elastic_index=elastic_index,
			embedding_field=embedding_field,
			text_field=text_field
		)

		self.sentence_embedder = SentenceEmbedder(llm_id=sentence_embedder_id)

	def generate_text_to_embedding_dict(self, batch):
		texts = list(set([item["_source"][self.text_field] for item in batch]))
		embeddings = self.sentence_embedder.forward_passage(texts)
		text_to_embedding = {text: encoding for text, encoding in zip(texts, embeddings)}
		return text_to_embedding


def forward_sentence_embedder_in_index(
		elastic_index: str,
		output_field_name: str,
		text_field_name: str,
		sentence_embedder_id: str,
		fields_to_sort: list[str] | None = None,
		filter_query: str | None= None,
):
	loop = asyncio.get_event_loop()
	sentence_embedder_enricher = SentenceEmbedderEnricher(
		embedding_field=output_field_name,
		text_field=text_field_name,
		elastic_index=elastic_index,
		sentence_embedder_id=sentence_embedder_id,
	)
	embedding_field_mapping = fewnerd_dataset.schema_llm_layer(size=sentence_embedder_enricher.sentence_embedder.dim_size(),
	                                                           layer_name=output_field_name)
	loop.run_until_complete(dataset_provider.ensure_field(elastic_index, embedding_field_mapping))
	queue = asyncio.Queue(maxsize=10000)
	BATCH_SIZE = 100

	for _ in range(5):

		loop.run_until_complete(asyncio.gather(
			sentence_embedder_enricher.load_to_queue(queue=queue,
			                                    batch_size=BATCH_SIZE,
			                                    fields_to_sort=fields_to_sort,
			                                    filter_query=filter_query),
			sentence_embedder_enricher.generate_embedding_task(
				queue=queue
			)
		))
	del sentence_embedder_enricher

	pass



def executing_pipeline(
		dataset_index: str,
		sentence_embedder_id: str,
		naming_index: str,
		):
	assert torch.cuda.is_available(), "no cuda"
	elastic_field_name = fewnerd_dataset.llm_and_layer_to_elastic_name(llm_id=sentence_embedder_id,layer='output')
	forward_sentence_embedder_in_index(
		elastic_index=dataset_index,
		output_field_name=elastic_field_name,
		sentence_embedder_id=sentence_embedder_id,
		text_field_name="all_text"
	)
	forward_sentence_embedder_in_index(
		elastic_index=naming_index,
		output_field_name=elastic_field_name,
		sentence_embedder_id=sentence_embedder_id,
		fields_to_sort=["entity_description"],
		text_field_name="entity_description"
	)



if __name__ == "__main__":
	clearml_poc.clearml_init(
		project_name="sentence_embedder_pipeline",
		task_name="forward_sentence",
		queue_name="a100_gpu",
		requirements=["transformers==4.46.2", "einops==0.8.1" ]
	)

	conf = {
		"dataset_index": "nertrieve_test",
		"naming_index": "nertrieve_entity_name_to_embedding",
		"sentence_embedder_id": "nvidia/NV-Embed-v2"
	}

	clearml_poc.clearml_connect_hyperparams(conf, "conf")

	executing_pipeline(**conf)
	print("deployed")
