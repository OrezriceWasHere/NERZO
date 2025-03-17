from dataclasses import asdict
from typing import Optional
from clearml_pipelines.fewnerd_pipeline import fewnerd_dataset
from clearml.automation.controller import PipelineDecorator
from clearml import TaskTypes
from contrastive import fewnerd_processor
from contrastive.args import FineTuneLLM


# Make the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step
@PipelineDecorator.component(
	return_values=["embedding_json"], task_type=TaskTypes.data_processing,
	execution_queue="a100_gpu"
	)
def step_create_embedding(
		llm_id: str,
		layer: str,
		max_llm_layer: Optional[int],
		entities_to_names: dict[str, str],
		):
	import torch
	from llm_interface import LLMInterface
	from tqdm import tqdm
	import json
	import hashlib

	print("step_create_embedding")
	assert torch.cuda.is_available(), "no gpu"
	device = torch.device("cuda:0")
	llm = LLMInterface(
		llm_id=llm_id,
		interested_layers=layer,
		max_llm_layer=max_llm_layer
	)

	output = []
	elastic_layer_name = fewnerd_dataset.llm_and_layer_to_elastic_name(llm_id=llm_id, layer=layer)
	for entity_name, entity_desc in tqdm(entities_to_names.items()):
		tokens = llm.tokenize(entity_desc).to(device)
		forwarded_tokens = llm.get_llm_at_layer(tokens, layer)[0]
		forwarded_tokens =  forwarded_tokens[1: , :]
		avg_tokens = torch.mean(forwarded_tokens, dim=0)
		last_tokens = forwarded_tokens[-1, :]

		entity_hash = str(hashlib.sha1(str.encode(entity_name)).hexdigest())
		output.append(
				{
					"entity_id": entity_hash,
					"entity_name": entity_name,
					"entity_description": entity_desc,
					"llm_layer": elastic_layer_name,
					f"embedding.{elastic_layer_name}.avg": avg_tokens.detach().cpu().tolist(),
					f"embedding.{elastic_layer_name}.end": last_tokens.detach().cpu().tolist(),
				}
		)

	return output


# Make the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step.
# Specifying `return_values` makes sure the function step can return an object to the pipeline logic
# In this case, the returned tuple will be stored as an artifact named "X_train, X_test, y_train, y_test"
@PipelineDecorator.component(
	packages=["aiohttp"],
	return_values=["success"], task_type=TaskTypes.data_processing,
	execution_queue="dsicsgpu"
)
def step_index_to_elastic(embeddings_json: list[dict], 	index = "fewnerd_entity_name_to_embedding"):
	import asyncio
	import dataset_provider
	print("step_index_to_elastic")
	loop = asyncio.get_event_loop()
	loop.run_until_complete(
		dataset_provider.ensure_existing_index(index, fewnerd_dataset.elasticsearch_fine_type_to_embedding)
		)

	schemas = [fewnerd_dataset.schema_llm_layer(layer_name=key, size=len(value))
	           for key, value in embeddings_json[0].items()
	           if key.startswith("embedding.")]
	create_fields_tasks = [dataset_provider.ensure_field(index, schema) for schema in schemas]
	loop.run_until_complete(asyncio.gather(*create_fields_tasks))

	index_to_elastic_tasks = [dataset_provider.upsert(data=embedding, index=index, doc_id=embedding["entity_id"]) for
	                          embedding in embeddings_json]
	loop.run_until_complete(asyncio.gather(*index_to_elastic_tasks))

	loop.close()


# The actual pipeline execution context
# notice that all pipeline component function calls are actually executed remotely
# Only when a return value is used, the pipeline logic will wait for the component execution to complete
@PipelineDecorator.pipeline(
	name="Forward fine type names in Elasticsearch", project="fewnerd_pipeline", version="0.0.1",
	packages="./requirements.txt",
	pipeline_execution_queue="a100_gpu"
	)
def executing_pipeline(
		llm_id: str,
		layer: str,
		max_llm_layer: Optional[int] = None, **kwargs
		):
	print("pipeline args:", llm_id, layer)

	# Use the pipeline argument to start the pipeline and pass it ot the first step
	print("launch step one")
	entities_to_names = fewnerd_processor.type_to_name()

	embeddings = step_create_embedding(
		llm_id,
		layer,
		max_llm_layer,
		entities_to_names=entities_to_names
		)

	# Use the returned data from the first step (`step_one`), and pass it to the next step (`step_two`)
	# Notice! unless we actually access the `data_frame` object,
	# the pipeline logic does not actually load the artifact itself.
	# When actually passing the `data_frame` object into a new step,
	# It waits for the creating step/function (`step_one`) to complete the execution
	print("launch step two")
	step_index_to_elastic(embeddings)


if __name__ == "__main__":
	# set the pipeline steps default execution queue (per specific step we can override it with the decorator)
	# PipelineDecorator.set_default_execution_queue('default')
	# Run the pipeline steps as subprocesses on the current machine, great for local executions
	# (for easy development / debugging, use `PipelineDecorator.debug_pipeline()` to execute steps as regular functions)
	PipelineDecorator.debug_pipeline()
	llm_args = FineTuneLLM()

	# Start the pipeline execution logic.
	executing_pipeline(**asdict(llm_args))

	print("process completed")
