import asyncio
import torch
import tqdm
import urllib3
import clearml_helper
import clearml_poc
import dataset_provider
from contrastive import fewnerd_processor
from contrastive.args import FineTuneLLM
from runtime_args import RuntimeArgs

urllib3.disable_warnings()


def not_processed_documents_query():
	query = {
		"query": {
			"bool": {
				"must_not": [
					{"exists": {"field": f"embedding.{mlp_id}"}},
				]

			}
		},
		"sort": [
			{"text_id":"asc"},
			"index_start"
		],
		"_source": [f"embedding.{args.llm_layer}.*", "entity_type"]
	}

	return query


async def document_producer(index, queue):
	query = not_processed_documents_query()
	query["size"] = BATCH_SIZE
	async for results in dataset_provider.consume_big_query(query, index):
		await queue.put(results)
	await queue.put(None)


def calculate_ne_embedding(embedding_end, embedding_start):
	embedding = fewnerd_processor.choose_llm_representation(
		embedding_end, embedding_start, input_tokens=args.input_tokens
	)
	embedding = embedding.to(device)
	with torch.no_grad():
		embedding = similarity_model(embedding)
	return embedding


async def document_consumer(index, queue):
	search_query = not_processed_documents_query()
	search_query.pop("sort")
	search_query.pop("_source")
	# search_query.pop("search_after")

	total_documents_to_produce = await dataset_provider.count(index, search_query)
	pbar = tqdm.tqdm(total=total_documents_to_produce)

	await ensure_existing_index(index)
	await dataset_provider.update_refresh_interval(index, "-1")
	count_errors = 0
	while True:
		records = await queue.get()
		if not records:
			break
		ids = [x["_id"] for x in records]
		embedding_start = torch.tensor(
			[item["_source"]["embedding"][args.llm_layer]["start"] for item in records], device=device,
			dtype=torch.double,
		).clone().detach()
		embedding_end = torch.tensor(
			[item["_source"]["embedding"][args.llm_layer]["end"] for item in records], device=device,
			dtype=torch.double,
		).clone().detach()
		ne_embedding = calculate_ne_embedding(embedding_end, embedding_start)
		batch = []
		for doc_id, embedding in zip(ids, ne_embedding):
			batch.append({"update": {"_index": index, "_id": doc_id}})
			batch.append({"doc": {f"embedding.{mlp_id}": embedding.tolist()}, "doc_as_upsert": True})
		x = await dataset_provider.bulk(batch)
		if slow_down_intentially:
			await asyncio.sleep(10)
		pbar.update(len(records))
		if x["errors"]:
			await asyncio.sleep(2 ** count_errors)
			count_errors += 1
			print("erros occured during bulk query. response for first item: ")
		else:
			if count_errors > 0:
				print("errors stopped")
			count_errors = 0
	await dataset_provider.update_refresh_interval(index, "30s")



def main():
	loop = asyncio.get_event_loop()
	queue = asyncio.Queue(maxsize=100)
	for x in range(3):
		loop.run_until_complete(asyncio.gather(
			document_producer(write_index, queue),
			document_consumer(write_index, queue)
		))
	loop.close()



async def ensure_existing_index(index_name):
	updated_field_mapping = {

		f'embedding.{mlp_id}': {
			"type": "dense_vector",
			"dims": args.output_layer,
			"index": False

		}

	}
	await  dataset_provider.ensure_field(index_name=index_name, field_mapping=updated_field_mapping)


if __name__ == "__main__":
	runtime_args = RuntimeArgs()
	fine_tune_llm = FineTuneLLM()
	clearml_poc.clearml_init(
		project_name="neretrieve_pipeline", task_name="Pipeline step 4 calculate ne embedding",
		queue_name="dsicsgpu"
	)
	assert torch.cuda.is_available()
	device = torch.device("cuda:0")

	mlp_layer = {"layer_id": "6b11b974e63543eb942741562046c063",
	"slow_down_intentionally": False}
	clearml_poc.clearml_connect_hyperparams(mlp_layer, name="conf")
	slow_down_intentially = mlp_layer["slow_down_intentionally"]
	BATCH_SIZE = 2500
	mlp_id = mlp_layer["layer_id"]
	write_index = "nertrieve_test"

	similarity_model = clearml_helper.get_mlp_by_id(mlp_id, device=device)
	similarity_model = similarity_model.double()
	similarity_model.eval()
	args = clearml_helper.get_args_by_mlp_id(mlp_id)

	main()

