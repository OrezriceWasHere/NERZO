from dataclasses import asdict
from typing import Optional

from clearml_pipelines.entity_name_to_embedding.index_entity_names_pipeline import step_create_embedding, \
	step_index_to_elastic
from clearml_pipelines.fewnerd_pipeline import fewnerd_dataset
from clearml.automation.controller import PipelineDecorator
from clearml import TaskTypes

from clearml_pipelines.nertreieve_dataset import nertrieve_processor
from contrastive import fewnerd_processor
from contrastive.args import FineTuneLLM



# The actual pipeline execution context
# notice that all pipeline component function calls are actually executed remotely
# Only when a return value is used, the pipeline logic will wait for the component execution to complete
@PipelineDecorator.pipeline(
	name="Forward fine type names in Elasticsearch", project="nertrieve_pipeline", version="0.0.1",
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
	entities_to_names = nertrieve_processor.type_to_name()
	PipelineDecorator.debug_pipeline()

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
	step_index_to_elastic(embeddings, index="nertrieve_entity_name_to_embedding")


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
