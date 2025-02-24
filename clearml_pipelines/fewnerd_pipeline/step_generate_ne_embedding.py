import asyncio
import torch
import tqdm
from clearml import Task
from elasticsearch import AsyncElasticsearch
import urllib3

import clearml_helper
import contrastive.fewnerd_processor
import queries
from clearml_pipelines.fewnerd_pipeline import fewnerd_dataset
from contrastive import fewnerd_processor
from contrastive.args import FineTuneLLM, Arguments, dataclass_decoder
from contrastive.mlp import ContrastiveMLP
from runtime_args import RuntimeArgs, ElasticsearchConnection
import itertools

urllib3.disable_warnings()

# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name="fewnerd_pipeline", task_name="Pipeline step 4 calculate ne embedding",
                 reuse_last_task_id=False)
task.add_requirements("requirements.txt")

if RuntimeArgs.running_remote:
    task.execute_remotely()


async def iterate_over_dataset(index, coarse_types):
    query = {
        "query": {
            "bool": {
                "must": [{
                    "terms": {
                        "coarse_type": coarse_types
                    }},
                    {"terms": {
                        "fine_type": contrastive.fewnerd_processor.test_fine_types()
                    }}
                ]
            }
        },
        "sort": [
            {
                "doc_id": {
                    "order": "desc"
                }
            }
        ],
        "_source": queries.all_source_except_other_embedding(args.llm_layer),
        "size": 100
    }
    count_query = {"query": query["query"]}
    count = await es.count(index=index, body=count_query)

    response = await es.search(index=index, body=query)
    hits = response.get("hits", {}).get("hits", [])
    search_after = hits[-1]["sort"] if hits else None

    while hits:
        yield hits
        if not search_after:
            break
        query["search_after"] = search_after
        response = await es.search(index=index, body=query)
        hits = response.get("hits", {}).get("hits", [])
        search_after = hits[-1]["sort"] if hits else None


def calculate_ne_embedding(embedding_end, embedding_start):
    embedding = fewnerd_processor.choose_llm_representation(embedding_end, embedding_start, input_tokens=args.input_tokens)
    embedding = embedding.to(device)
    with torch.no_grad():
        embedding = similarity_model(embedding)
    return embedding


async def update_ne_embedding(index, field_id, doc):
    await es.update(index=write_index, id=field_id, body={"doc": doc, "doc_as_upsert": True})


async def main_task(index, coarse_types):
    async for hits in iterate_over_dataset(index, coarse_types):
        for hit in hits:
            field_id = hit["_id"]
            embeddings = hit["_source"]["embedding"][args.llm_layer]
            embedding_start = embeddings["start"]
            embedding_end = embeddings["end"]
            ne_embedding = calculate_ne_embedding(embedding_end, embedding_start)
            doc = {
                **hit["_source"],
                "embedding": {
                    mlp_id: ne_embedding.cpu().numpy().tolist()
                }
            }
            await update_ne_embedding(index, field_id, doc)
            pbar.update(1)


async def main():
    await ensure_existing_index(write_index, fewnerd_dataset.elasticsearch_tests_mapping)
    split_to_indices = ["fewnerd_v4_train", "fewnerd_v4_dev", "fewnerd_v4_test"]
    split_to_coarse_types = [["location"], ["person", "other"], ["organization", "product", "event", "building", "art"]]
    tasks = [main_task(index, types) for index, types in itertools.product(split_to_indices, split_to_coarse_types)]

    await asyncio.gather(*tasks)
    await es.close()


async def ensure_existing_index(index_name, mapping):
    if not await es.indices.exists(index=index_name):
        await es.indices.create(index=index_name, body=mapping)

    if not indexing_original_llm_tokens:
        updated_field_mapping = {

            f'embedding.{mlp_id}': {
                "type": "dense_vector",
                "dims": args.output_layer,
                "index": "true",
                "similarity": "cosine",
                "index_options": {
                    "type": "flat"
                }
            }

        }
        await  es.indices.put_mapping(index=index_name, properties=updated_field_mapping)


if __name__ == "__main__":
    runtime_args = RuntimeArgs()
    fine_tune_llm = FineTuneLLM()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA is not available, running on CPU")

    fewnerd_schema = fewnerd_dataset.elasticsearch_tests_mapping
    original_keys = list(fewnerd_schema["mappings"]["properties"]["embedding"]["properties"].keys())
    mlp_layer = {"layer_id": "1523129bcea64c129ebab77db4880270"}
    task.connect(mlp_layer, name="layer_name")
    mlp_id = mlp_layer["layer_id"]
    indexing_original_llm_tokens = mlp_id in original_keys
    write_index = "fewnerd_tests"
    args = Arguments()
    if indexing_original_llm_tokens:
        similarity_model = torch.nn.Identity()
        args.llm_layer = mlp_id
    else:
        similarity_model = clearml_helper.get_mlp_by_id(mlp_id, device=device)
        args = clearml_helper.get_args_by_mlp_id(mlp_id)
        similarity_model.eval()

    es_conf = ElasticsearchConnection().model_dump()
    es = AsyncElasticsearch(**es_conf)

    pbar = tqdm.tqdm()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
