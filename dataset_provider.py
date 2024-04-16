import os

from elasticsearch import Elasticsearch
from json import load

import queries

hosts = os.environ.get("ELASTICSEARCH_HOSTS") or "https://dsicscpu01:9200"
user = os.environ.get("ELASTICSEARCH_USER") or "elastic"
password = os.environ.get("ELASTICSEARCH_PASSWORD") or "XXX"

es = Elasticsearch(hosts=hosts,
                   verify_certs=False,
                   basic_auth=(user, password))

elastic_query = load(open("elastic_query.json"))

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

def get_top_results_for_entities(count_entity_types=500,
                                 count_per_type=5,
                                 index="training_data_object,testing_data_object"):
    query = queries.query_aggergate_by_type(count_entity_types, count_per_type)
    response = search(index=index, query=query)
    return response