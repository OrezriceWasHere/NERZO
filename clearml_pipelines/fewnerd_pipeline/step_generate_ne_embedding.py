import asyncio
import torch
from clearml import Task
from elasticsearch import AsyncElasticsearch
import clearml_poc
import urllib3

from contrastive.args import FineTuneLLM, Arguments
from contrastive.mlp import ContrastiveMLP
from runtime_args import RuntimeArgs

urllib3.disable_warnings()

# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name="fewnerd_pipeline", task_name="Pipeline step 4 calculate ne embedding",
                 reuse_last_task_id=False)

task.execute_remotely()


async def iterate_over_dataset(index, coarse_types):
    query = {
        "query": {
            "terms": {
                "coarse_type": coarse_types
            }
        },
        "sort": [
            {
                "doc_id": {
                    "order": "desc"
                }
            }
        ],
        "_source": [
            "-embedding.*",
            f"embedding.{args.llm_layer}.*"
        ],
        "size": 100
    }
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
    embedding = torch.cat([embedding_start, embedding_end], dim=0)
    embedding = embedding.unsqueeze(0)
    embedding = embedding.to(device)
    with torch.no_grad():
        embedding = similarity_model(embedding)
    return embedding

async def update_ne_embedding(index, field_id, doc):
    await es.update(index=index, id=field_id, body={"doc": doc})

async def main_task(index, coarse_types):
    async for hits in iterate_over_dataset(index, coarse_types):
        for hit in hits:
            field_id = hit["_id"]
            embeddings = hit["_source"]["embedding"][args.llm_layer]
            embedding_start = embeddings["start"]
            embedding_end = embeddings["end"]
            ne_embedding = calculate_ne_embedding(embedding_end, embedding_start)
            doc = {
                "embedding": {
                    args.llm_layer: {
                        "ne_embedding": ne_embedding.cpu().numpy().tolist()
                    }
                }
            }
            await update_ne_embedding(index, field_id, doc)


async def main():
    split_to_indices = ["fewnerd_v4_train", "fewnerd_v4_dev", "fewnerd_v4_test"]
    split_to_coarse_types = [["location"], ["person", "other"], ["organization", "product", "event", "building", "art"]]
    tasks = [main_task(index, types) for index, types in zip(split_to_indices, split_to_coarse_types)]
    await asyncio.gather(*tasks)
    await es.close()


if __name__ == "__main__":
    runtime_args = RuntimeArgs()
    fine_tune_llm = FineTuneLLM()
    args = Arguments()
    auth = (runtime_args.elasticsearch_user, runtime_args.elasticsearch_password)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA is not available, running on CPU")

    similarity_model = ContrastiveMLP(args).to(device)
    mlp_id = fine_tune_llm.mlp_head_model_id_from_clearml
    local_mlp_head_path = clearml_poc.download_model(mlp_id)
    assert local_mlp_head_path, "could not download mlp head model"
    local_mlp_head_model = torch.load(local_mlp_head_path,
                                      weights_only=True,
                                      map_location=device)
    similarity_model.load_state_dict(local_mlp_head_model)
    similarity_model.eval()

    es = AsyncElasticsearch(hosts=runtime_args.elasticsearch_host,
                            verify_certs=False,
                            max_retries=10,
                            request_timeout=30,
                            retry_on_timeout=True,
                            basic_auth=auth)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
