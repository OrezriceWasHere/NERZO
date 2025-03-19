import asyncio
import hashlib
import json
import os.path
from dataclasses import asdict
from typing import Optional
from tqdm import tqdm
from clearml_pipelines.entity_name_to_embedding.index_entity_names_pipeline import step_create_embedding
import dataset_provider
from clearml_pipelines.fewnerd_pipeline import fewnerd_dataset
from clearml.automation.controller import PipelineDecorator
from clearml import TaskTypes, Dataset
from clearml_pipelines.nertreieve_dataset import nertrieve_index_to_db
from contrastive.args import FineTuneLLM, Arguments
from datasets import load_dataset
from clearml_pipelines.nertreieve_dataset.step_process_to_document import add_embedding_to_batch




@PipelineDecorator.component(
	 task_type=TaskTypes.data_processing,
	execution_queue="dsicsgpu", cache=True
	)
def step_forward_in_llm(
		db_layer_name: str,
		llm_id: str,
		layer: str,
		max_llm_layer: Optional[int],
		):
	def word_indices(words):
		indices = []
		start_idx = 0

		for word in words:
			end_idx = start_idx + len(word) - 1  # Calculate end index
			indices.append((start_idx, end_idx))
			start_idx = end_idx + 2  # Move to next word (adding 1 for space)

		return indices
	import torch
	from llm_interface import LLMInterface
	from tqdm import tqdm

	print("step_forward_in_llm")
	assert torch.cuda.is_available(), "no gpu"
	fine_type_to_coarse_type = {
		"Facility": "Location",
		"OtherLOC": "Location",
		"HumanSettlement": "Location",
		"Station": "Location",
		"VisualWork": "CreativeWork",
		"MusicalWork": "CreativeWork",
		"WrittenWork": "CreativeWork",
		"ArtWork": "CreativeWork",
		"Software": "CreativeWork",
		"MusicalGRP": "Group",
		"PublicCORP": "Group",
		"PrivateCORP": "Group",
		"AerospaceManufacturer": "Group",
		"SportsGRP": "Group",
		"CarManufacturer": "Group",
		"ORG": "Group",
		"Scientist": "Person",
		"Artist": "Person",
		"Athlete": "Person",
		"Politician": "Person",
		"Cleric": "Person",
		"SportsManager": "Person",
		"OtherPER": "Person",
		"Clothing": "Product",
		"Vehicle": "Product",
		"Food": "Product",
		"Drink": "Product",
		"OtherPROD": "Product",
		"Medication/Vaccine": "Medical",
		"MedicalProcedure": "Medical",
		"AnatomicalStructure": "Medical",
		"Symptom": "Medical",
		"Disease": "Medical"
	}
	fine_to_coarse = {key.lower(): value.lower() for key, value in fine_type_to_coarse_type.items()}
	layers_and_keys_pair = [(layer, db_layer_name)]

	dataset = load_dataset("MultiCoNER/multiconer_v2", "English (EN)")

	device = torch.device("cuda:0")
	llm = LLMInterface(
		llm_id=llm_id,
		interested_layers=layer,
		max_llm_layer=max_llm_layer
	)
	BATCH_SIZE = 50
	batch = []
	for env in dataset:
		len_batch = dataset[env]
		print("current env ", env)
		path = f"multiconer_{env}.json"
		file = open(path, "w")
		for count, item in tqdm(enumerate(dataset[env])):
			all_text = " ".join(item["tokens"])
			fine_labels = item["ner_tags"]
			prev_labels = ([None] + fine_labels)[:len(fine_labels)]
			nex_labels = (fine_labels + [None])[1:]
			text_id = item["sample_id"]
			tagging = []
			word_start_and_end = word_indices(item["tokens"])
			for index, word, ner_tag, prev_tag, next_tag in zip(range(len(item["tokens"])),item["tokens"], item["ner_tags"], prev_labels, nex_labels):
				if ner_tag == 'O':
					continue
				is_inside_tag = 'B-' in ner_tag or 'I-' in ner_tag
				if is_inside_tag:
					tagging.append(word)
				is_end = (next_tag == 'O') or (next_tag is None)
				if is_end:
					last_word_indices = word_start_and_end[index]
					first_word_indices = word_start_and_end[index - len(tagging) + 1]
					fine_tag_label = ner_tag.replace("B-","").replace("I-", "").lower()
					index_start =first_word_indices[0]
					index_end =last_word_indices[1]
					phrase = " ".join(tagging)
					if index_start == index_end and all_text[index_start] == phrase:
						index_end = index_end + 1
					batch.append(
						{
							"phrase": phrase,
							"coarse_type": fine_to_coarse[fine_tag_label],
							"fine_type": fine_tag_label,
							"index_start": index_start,
							"index_end": index_end,
							"text_id": text_id,
							"all_text": all_text
						}
					)
					tagging.clear()

			if count % BATCH_SIZE == 0 or count + 1 == len_batch:
				add_embedding_to_batch(batch, llm=llm, device=device, layers_and_keys_pairs=layers_and_keys_pair)

				for doc in batch:
					file.write(json.dumps(doc) + "\n")
				batch = []

		file.flush()
		file.close()
		clearml_dataset = Dataset.create(dataset_name=path, dataset_project="multiconer_pipeline")
		clearml_dataset.add_files(path=path)
		clearml_dataset.add_tags([env, llm_id])
		clearml_dataset.upload()
		clearml_dataset.finalize()


# Make the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step
@PipelineDecorator.component(
	task_type=TaskTypes.custom,
	cache=False,
	)
def index_to_db(llm_id: str):
	import aiofiles

	def generate_id(item):
		keys = ["all_text", "coarse_type", "fine_type", "index_end", "index_start", "phrase"]
		hash_v = hashlib.sha1(str.encode("".join([str(item[key]) for key in keys]))).hexdigest()
		return f"multiconer_{hash_v}"

	async def document_producer(mapping, queue):
		envs = ["train", "validation", "test"]

		for env in envs:
			index = f'multiconer_{env}'
			path = index + ".json"
			await dataset_provider.ensure_existing_index(index, mapping)
			dataset_dir = Dataset.get(
				dataset_name=path,
				dataset_tags=[env, llm_id],
				dataset_project="multiconer_pipeline"
			).get_local_copy()
			async with aiofiles.open(os.path.join(dataset_dir, path)) as f:
				async for line in f:
					item = json.loads(line)
					await queue.put((item, index))

		await queue.put((None, None))

	async def document_consumer(queue):
		pbar = tqdm()
		doc_id_generator_function = generate_id
		BATCH_SIZE = 2500
		bulk = []
		while True:
			item, index = await queue.get()
			if not item:
				break

			bulk.append((item, index))
			if len(bulk) >= BATCH_SIZE:
				x = await nertrieve_index_to_db.write_batch(bulk, id_generator=doc_id_generator_function)
				pbar.update(len(bulk))

		if bulk:
			await nertrieve_index_to_db.write_batch(bulk, id_generator=doc_id_generator_function)

	print("index to db")
	database_mapping = fewnerd_dataset.elasticsearch_storage_mapping
	loop = asyncio.get_event_loop()
	queue = asyncio.Queue()

	loop.run_until_complete(asyncio.gather(
		document_producer(database_mapping, queue),
		document_consumer(queue)
	))

	return True




# The actual pipeline execution context
# notice that all pipeline component function calls are actually executed remotely
# Only when a return value is used, the pipeline logic will wait for the component execution to complete
@PipelineDecorator.pipeline(
	name="index multi coner", project="multiconer_pipeline", version="0.0.1",
	pipeline_execution_queue="dsicsgpu"
	)
def executing_pipeline(
		args:Arguments,
		llm_id: str,
		layer: str,
		max_llm_layer: Optional[int] = None, **kwargs
		):
	print("pipeline args:", llm_id, layer)
	# step_forward_in_llm(db_layer_name=args.llm_layer, llm_id=llm_id, layer=layer, max_llm_layer=max_llm_layer)
	result = index_to_db(llm_id)

	fine_type_to_name = {
		"OtherLOC": "Location - Other",
		"HumanSettlement": "Human Settlement",
		"VisualWork": "Visual Work",
		"MusicalWork": "Musical Work",
		"WrittenWork": "Written Work",
		"ArtWork": "Art Work",
		"MusicalGRP": "Musical Group",
		"PublicCORP": "Public Corporation",
		"PrivateCORP": "Private Corporation",
		"AerospaceManufacturer": "Aerospace Manufacturer",
		"SportsGRP": "Sports Group",
		"CarManufacturer": "Car Manufacturer",
		"ORG": "Organization",
		"SportsManager": "Sports Manager",
		"OtherPER": "Person - Other",
		"OtherPROD": "Product - Other",
		"Medication/Vaccine": "Medication or Vaccine",
		"MedicalProcedure": "Medical Procedure",
		"AnatomicalStructure": "Anatomical Structure"
	}
	fine_type_to_coarse_type = {
		"Facility": "Location",
		"OtherLOC": "Location",
		"HumanSettlement": "Location",
		"Station": "Location",
		"VisualWork": "CreativeWork",
		"MusicalWork": "CreativeWork",
		"WrittenWork": "CreativeWork",
		"ArtWork": "CreativeWork",
		"Software": "CreativeWork",
		"MusicalGRP": "Group",
		"PublicCORP": "Group",
		"PrivateCORP": "Group",
		"AerospaceManufacturer": "Group",
		"SportsGRP": "Group",
		"CarManufacturer": "Group",
		"ORG": "Group",
		"Scientist": "Person",
		"Artist": "Person",
		"Athlete": "Person",
		"Politician": "Person",
		"Cleric": "Person",
		"SportsManager": "Person",
		"OtherPER": "Person",
		"Clothing": "Product",
		"Vehicle": "Product",
		"Food": "Product",
		"Drink": "Product",
		"OtherPROD": "Product",
		"Medication/Vaccine": "Medical",
		"MedicalProcedure": "Medical",
		"AnatomicalStructure": "Medical",
		"Symptom": "Medical",
		"Disease": "Medical"
	}

	entity_types_to_names = {
		fine_type: fine_type_to_name.get(fine_type, None) or fine_type
		for fine_type in fine_type_to_coarse_type
	}
	step_create_embedding(
		llm_id=llm_id,
		layer=layer,
		max_llm_layer=max_llm_layer,
		entities_to_names=entity_types_to_names,
	)


if __name__ == "__main__":
	# set the pipeline steps default execution queue (per specific step we can override it with the decorator)
	# PipelineDecorator.set_default_execution_queue('default')
	# Run the pipeline steps as subprocesses on the current machine, great for local executions
	# (for easy development / debugging, use `PipelineDecorator.debug_pipeline()` to execute steps as regular functions)
	PipelineDecorator.run_locally()
	llm_args = FineTuneLLM()
	args = Arguments()



	# Start the pipeline execution logic.
	executing_pipeline(args, **asdict(llm_args))

	print("process completed")
