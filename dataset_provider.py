import os
from elasticsearch import Elasticsearch, AsyncElasticsearch
import queries
import runtime_args

elastic_conf = runtime_args.ElasticsearchConnection()

es = Elasticsearch(**elastic_conf.model_dump())

async_es = AsyncElasticsearch(**elastic_conf.model_dump())

elastic_query = None

both_indices = "ner_poc"
new_index = "no_redundant_object"


def search(query=elastic_query, index=both_indices, **kwargs):
    response = es.search(index=index, body=query, **kwargs)
    return response.body


def write_to_index(data, index=new_index):
    response = es.index(index=index, body=data, id=data["id"])
    return response


def get_by_title(title: str):
    query = queries.query_get_by_title(title)
    index = "ner_poc"
    response = search(index=both_indices, query=query)
    return response


def count_apperance_of_entities(title: str):
    query = queries.query_count_apperance_of_entities(title)
    index = "ner_poc"
    response = search(index=both_indices, query=query)
    return response


def get_by_entity_type(entity_type: str):
    query = queries.query_get_by_entity_type(entity_type)
    index = "training_data_object,testing_data_object"
    response = search(index=index, query=query, filter_path=["hits.hits._source"])
    return response


def get_by_fine_grained_type_fewnerd(fine_grained_type: str):
    query = queries.query_get_by_fine_grained_fewnerd(fine_grained_type)
    index = "fewnerd_v2_*"
    response = search(index=index, query=query, filter_path=["hits.hits._source"])
    return response


def get_by_fine_grained_type_fewnerd_v3(fine_grained_type: str, batch_size, randomize=False):
    query = queries.query_get_by_fine_grained_fewnerd_v3(fine_grained_type, randomized=randomize, batch_size=batch_size)
    index = "fewnerd_v3_*"
    response = search(index=index, query=query, filter_path=["hits.hits._source"])
    response = response.get("hits", {}).get("hits", [])
    return response


async def get_randomized_by_fine_type_fewnerd_v4(fine_grained_type: str, batch_size, llm_layer=None):
    query = queries.query_get_by_fine_grained_fewnerd_v3(fine_grained_type, batch_size=batch_size, llm_layer=llm_layer)
    index = "fewnerd_v4_*"
    response = await async_es.search(index=index, body=query, filter_path=["hits.hits._source"])
    response = response.body.get("hits", {}).get("hits", [])
    return response

async def get_hard_negative_fewnerd(fine_types: str | list[str],
                              coarse_type: str | list[str],
                              anchor_text:str,
                              batch_size: int,
                              llm_layer:str=None):
    query = queries.query_hard_negative(
        fine_grained_type=fine_types,
        coarse_grained_type=coarse_type,
        anchor_text=anchor_text,
        size=batch_size,
        llm_layer=llm_layer
    )
    index = "fewnerd_v4_*"
    response = await async_es.search(index=index, body=query, filter_path=["hits.hits._source"])
    response = response.body.get("hits", {}).get("hits", [])
    return response


def yield_by_fine_grained_type_fewnerd_v3(fine_grained_types: list[str], scroll: str = "3m", randomize=False,
                                          batch_size=200):
    """
    This function is used if we want to get all entites in the fine_grained_types list, using search after mechanism
    """
    query = queries.query_get_by_fine_grained_fewnerd_v3(fine_grained_types, randomize, batch_size)
    index = "fewnerd_v3_*"
    response = es.search(index=index, body=query)
    hits = response.get("hits", {}).get("hits", [])
    search_after = hits[-1]["sort"] if hits else None

    while hits:
        yield hits
        if not search_after:
            break
        query["search_after"] = search_after
        response = es.search(index=index, body=query)
        hits = response.get("hits", {}).get("hits", [])
        search_after = hits[-1]["sort"] if hits else None


def random_results_per_fine_type(fine_types, instances_per_type=100):
    """
    This function is used to get a fixed amount of entity instances, for each fine type in the fine_types list.

    """
    query = queries.fewnerd_random_results_per_fine_type(fine_types, instances_per_type)
    index = "fewnerd_v4_*"
    response = es.search(index=index, body=query)

    hits = response.get("aggregations", {}).get("filter_types", {}).get("top_artifacts").get("buckets", [])[0].get(
        "hits", {}).get("hits", {}).get("hits", [])
    for hit in hits:
        yield hit

    search_after = response.get("aggregations", {}).get("filter_types", {}).get("top_artifacts").get("after_key")
    while search_after:
        print('current_type in search after', search_after)
        query["aggs"]["filter_types"]["aggs"]["top_artifacts"]["composite"]["after"] = search_after
        response = es.search(index=index, body=query)
        try:
            hits = response.get("aggregations", {}).get("filter_types", {}).get("top_artifacts").get("buckets", [])[
                0].get(
                "hits", {}).get("hits", {}).get("hits", [])

            for hit in hits:
                yield hit
            search_after = response.get("aggregations", {}).get("filter_types", {}).get("top_artifacts").get(
                "after_key")
        except IndexError:
            search_after = None



def get_by_coarse_grained_type_fewnerd(fine_grained_type: str):
    query = queries.query_get_by_coarse_grained_fewnerd(fine_grained_type)
    index = "fewnerd_v2_*"
    response = search(index=index, query=query, filter_path=["hits.hits._source"])
    return response


def get_top_results_for_entities(count_entity_types=500,
                                 count_per_type=5,
                                 index="training_data_object,testing_data_object"):
    query = queries.query_aggergate_by_type(count_entity_types, count_per_type)
    response = search(index=index, query=query)
    return response

async def ensure_existing_index(index_name, mapping):
    if not await async_es.indices.exists(index=index_name):
        await async_es.indices.create(index=index_name, body=mapping)

async def upsert(data,doc_id, index):
    body = {
        "doc": data,
        "doc_as_upsert": True
    }
    response = await async_es.update(index=index, id=doc_id, body=body)
    return response

async def ensure_field(index_name, field_mapping):
    await async_es.indices.put_mapping(index=index_name, properties=field_mapping)

async def consume_big_aggregation(query, agg_key, index):
    response = await async_es.search(index=index, body=query, size=0)

    while "after_key" in response["aggregations"][agg_key]:
        after_key = response["aggregations"][agg_key]["after_key"]
        for bucket in response["aggregations"][agg_key]["buckets"]:
            yield bucket
        query["aggs"][agg_key]["composite"]["after"] = after_key
        response = await async_es.search(index=index, body=query, size=0)