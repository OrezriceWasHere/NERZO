import json
from uuid import uuid4
import torch
from clearml import Task
from tqdm import tqdm
from clearml import StorageManager, Dataset
from bz2 import decompress

import clearml_poc
from clearml_pipelines.fewnerd_pipeline import fewnerd_dataset
from clearml_pipelines.nertreieve_dataset import neretrieve_dataset
from contrastive.args import Arguments, FineTuneLLM
from llm_interface import LLMInterface


# Connecting ClearML with the current process,
# from here on everything is logged automatically

def split_into_document(dataset_file):
	pbar = tqdm()
	with open(dataset_file, "r") as f:
		# For document in file
		for line in f:
			pbar.update(1)
			yield json.loads(line)


def create_embedding(text, indices):
	tokens = llm.tokenize(text).to(device)
	llm_indices = llm.token_indices_given_text_indices(text, indices)
	embeddings = {}
	for model_layer, db_name in layers_and_keys_pairs:
		h = llm.get_llm_at_layer(tokens, model_layer)[0]
		start = h[llm_indices[0] - 1]
		end = h[llm_indices[1]]
		embeddings[db_name] = {
			"start": start.tolist(),
			"end": end.tolist()
		}

	return embeddings


def add_embedding_to_batch(batch):
	all_texts = [x["all_text"] for x in batch]
	distinct_text = set(all_texts)
	distinct_text_list = list(distinct_text)
	tokens = llm.tokenize(distinct_text_list).to(device)
	assert len(layers_and_keys_pairs), "supporting one layer at a time right now"
	layers_and_keys_pair = layers_and_keys_pairs[0]
	hidden_values = llm.get_llm_at_layer(tokens, layers_and_keys_pair[0])
	db_name = layers_and_keys_pair[1]
	text_to_embedding = {text: embedding for text, embedding in zip(distinct_text, hidden_values)}
	for doc in batch:
		h = text_to_embedding[doc["all_text"]]
		llm_indices = llm.token_indices_given_text_indices(doc["all_text"], (doc["index_start"], doc["index_end"]))
		start = h[llm_indices[0] - 1]
		end = h[llm_indices[1]]
		doc["embedding"] = {
			db_name: {
				"start": start.tolist(),
				"end": end.tolist()
			}
		}

	del hidden_values, text_to_embedding
	torch.cuda.empty_cache()

def process_batch(batch):
	docs = []
	for document in batch:
		tagging_array = []
		text_id = document["id"]
		all_text = document["content"]
		for tagging_type, type_content in document["tagged_entities"].items():
			for tagging_instance, instance_content in type_content.items():
				for phrase, references in instance_content.items():
					for reference in references:
						if not reference:
							continue
						word_index_start = min(reference)
						index_start = sum(
							len(word) for word in document['document_token_sequence'][:word_index_start]
							) + word_index_start
						index_end = index_start + len(
							" ".join(document['document_token_sequence'][min(reference):max(reference) + 1])
							)
						if all_text[index_start:index_end].lower() == phrase.lower():
							tagging_array.append(
								{
									"phrase": phrase,
									"entity_type": tagging_type,
									"entity_id": tagging_instance,
									"index_in_text": reference,
									"index_start": index_start,
									"index_end": index_end
								}
							)
		if not tagging_array:
			continue
		distinct_indices = {tuple((tagging["index_start"], tagging["index_end"])) for tagging in tagging_array}
		# embeddings = {indices: create_embedding(all_text, indices) for indices in distinct_indices}

		for tagging in tagging_array:
			tagging["all_text"] = all_text
			tagging["text_id"] = text_id
		docs.extend(tagging_array)
	add_embedding_to_batch(docs)
	# embedding = embeddings[tuple((tagging["index_start"], tagging["index_end"]))]
	# tagging["embedding"] = embedding

	return docs


def process_dataset(dataset_url, output_file):
	dataset_file = StorageManager.get_local_copy(remote_url=dataset_url)
	print("unzipping dataset...")
	source_file = dataset_file
	uncompressed_file = dataset_file[:-4]
	with open(source_file, 'rb') as source, open(uncompressed_file, 'w') as dest:
		dest.write(decompress(source.read()).decode('utf-8'))
	print("processing dataset...")
	documents = split_into_document(uncompressed_file)
	processed_documents = []
	with open(output_file, "w") as file:
		batch = []

		for document in documents:
			if len(batch) < BATCH_SIZE:
				batch.append(document)
				continue
			else:
				processed_documents = process_batch(batch)
				for document in processed_documents:
					file.write(json.dumps(document))
				batch = []
		if batch:
			processed_documents = process_batch(batch)
			for document in processed_documents:
				file.write(json.dumps(document)+"\n")
	return processed_documents


def main_process(dataset):
	file_dir = dataset["json"]
	# with open(file_dir, "w") as file:
	process_dataset(dataset["url"], file_dir)
	tags = db_key + [dataset["env"]]
	clearml_poc.add_tags(tags)

	# file.write(json.dumps(processed_documents))
	clearml_dataset = Dataset.create(dataset_name=file_dir, dataset_project="neretrieve_pipeline")
	clearml_dataset.add_files(path=file_dir)
	clearml_dataset.add_tags(tags)

	# Dataset is uploaded to the ClearML Server by default
	clearml_dataset.upload()
	clearml_dataset.finalize()


if __name__ == "__main__":
	clearml_poc.clearml_init(
		project_name="neretrieve_pipeline",
		task_name="Pipeline step 2 jsonify dataset",
		requirements=[
			"bitsandbytes >=0.43.2"
		]
	)
	BATCH_SIZE = 8
	args = Arguments()
	clearml_poc.clearml_connect_hyperparams(args, "general")
	llm_args = FineTuneLLM()
	clearml_poc.clearml_connect_hyperparams(llm_args, "llm_args")

	llm_id = llm_args.llm_id
	interested_layers = [llm_args.layer]
	db_key = [args.llm_layer]
	layers_and_keys_pairs = list(zip(interested_layers, db_key))
	llm = LLMInterface(
		llm_id=llm_id,
		# interested_layers=list(interested_layers),
		max_llm_layer=llm_args.max_llm_layer
		)
	assert torch.cuda.is_available()
	device = torch.device("cuda")
	neretrieve_dataset.datasets = [neretrieve_dataset.datasets[1]]
	for dataset in neretrieve_dataset.datasets:
		main_process(dataset)

print('Notice, artifacts are uploaded in the background')
print('Done')
