import asyncio
from collections import defaultdict
from dataclasses import asdict
from functools import cache

import pandas as pd
from tqdm import tqdm

import queries
import clearml_poc
import sentence_embedder
import dataset_provider
from contrastive import fewnerd_processor
from contrastive.args import FineTuneLLM


def main():
    all_test_types = fewnerd_processor.retrieve_anchors_for_sentence_test()
    tasks = [handle_type(type, ids) for type, ids in all_test_types.items()]
    loop = asyncio.get_event_loop()
    result = defaultdict(list)
    task_results = loop.run_until_complete(asyncio.gather(*tasks))
    loop.close()
    for fine_type, task_result in zip(all_test_types.keys(), task_results):
        for result_key, result_value in task_result.items():
            result[result_key].append(result_value)
        result["index"].append(fine_type)

    index_column = result.pop("index")
    df = pd.DataFrame(data=result, index=index_column)
    clearml_poc.add_table(
        iteration=0,
        series="eval",
        title="recall per fine type",
        table=df
    )

    clearml_poc.add_table(
        iteration=0,
        series="eval",
        title="average recall",
        table=df.mean().to_frame()
    )


async def get_count_fine_type(fine_type):
    query = {
        "query": {
            "nested": {
                "path": "tagging",
                "query": {
                    "bool": {
                        "must": [
                            {"term": {
                                "tagging.fine_type": fine_type
                            }}
                        ]
                    }

                }
            }
        }
    }
    return await dataset_provider.count(index_name=index, query=query)


async def handle_type(fine_type, ids):
    count_type = await get_count_fine_type(fine_type)
    result = {}
    for k in (10, 50, count_type):
        recall_list = []
        for fewnerd_id in ids:
            recall = await calculate_recall_per_fine_type_k_size(fine_type, fewnerd_id, k)
            recall_list.append(recall)
        size_desc = k if k in [10, 50] else "size"
        result[f"recall@{size_desc}"] = sum(recall_list) / len(recall_list)
    result["size"] = count_type
    return result


def prompt(**kwargs):
    return template.format(**kwargs)


async def get_by_sentence_id(sentence_id):
    query = {"query": {"terms": {"text_id": [sentence_id]}}}
    document = await dataset_provider.search_async(index, query)
    results = document.get("hits", {}).get("hits", [])
    assert len(results) == 1, f"not one result for sentence {sentence_id}\n count results {len(results)}"
    return results[0]["_source"]


async def get_k_most_similar_documents(embedding, k):
    size_limit = min(k + 1, 10000)
    query = {
        "query": queries.query_search_by_similarity(embedding, "embedding." + layer),
        "_source": [
            "tagging"
        ],
        "sort": [
            "_score", "text_id"
        ],
        "size": size_limit
    }
    results = []
    async for batch in dataset_provider.consume_big_query(query, index):
        results.extend(batch)
        if len(results) >= k + 1:
            break
    return results[1:k + 1]


def count_expected_entity_type(predictions_documents, expected_entity_type):
    predictions = [1 for array in predictions_documents if
                   any(item for item in array["_source"]["tagging"] if item["fine_type"] == expected_entity_type)]
    return len(predictions)

@cache
def forward_query_in_llm(query):
    return sentence_embedder.forward_query(query)

async def calculate_recall_per_fine_type_k_size(fine_type, sentence_id, k):
    fine_type_results = []
    document = await get_by_sentence_id(sentence_id)
    phrases_with_entity_type = [item["phrase"] for item in document["tagging"] if item["fine_type"] == fine_type]
    for phrase in phrases_with_entity_type:
        query_prompt = prompt(sentence=document["all_text"], phrase=phrase)
        embedding = forward_query_in_llm(query_prompt)
        similar_items = await get_k_most_similar_documents(embedding, k)
        count_with_expected_entity_type = count_expected_entity_type(similar_items, fine_type)
        recall = count_with_expected_entity_type / k
        fine_type_results.append(recall)
    pbar.update(1)
    return sum(fine_type_results) / len(fine_type_results)

if __name__ == "__main__":
    clearml_poc.clearml_init(task_name="sentence evaluation")
    # template = (
    #     "In the following sentence, find some other sentences such that the semantic meaning of entities is most similar."
    #     "The sentence is: {sentence}")
    template = "{sentence}"

    llm_args = FineTuneLLM()
    clearml_obj = {"layer_id": llm_args.llm_id,
                   "prompt_template": template}
    clearml_poc.clearml_connect_hyperparams(clearml_obj, name="layer_name")
    layer = clearml_obj["layer_id"]
    template = clearml_obj["prompt_template"]
    print("template is {}".format(template))
    index = "fewnerd_full_sentence_train,fewnerd_full_sentence_test,fewnerd_full_sentence_dev"
    sentence_embedder = sentence_embedder.SentenceEmbedder(**asdict(llm_args))
    pbar = tqdm(total=20 * 3 * 3)  # 20 fine entity types * 3 sizes  * 3 instances
    main()
