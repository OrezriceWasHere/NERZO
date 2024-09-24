import os

from elasticsearch import Elasticsearch
from json import load

import queries

hosts = os.environ.get("ELASTICSEARCH_HOSTS") or "https://dsicscpu01:9200"
user = os.environ.get("ELASTICSEARCH_USER") or "elastic"
password = os.environ.get("ELASTICSEARCH_PASSWORD") or "XXX"

es = Elasticsearch(hosts=hosts,
                   verify_certs=False,
                   ssl_show_warn=False,
                   basic_auth=(user, password))

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


def yield_by_fine_grained_type_fewnerd_v3(fine_grained_types: list[str], scroll: str = "3m", randomize=False, batch_size=200):
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
