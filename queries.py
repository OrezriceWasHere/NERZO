

def query_get_by_title(title: str) -> dict:
    return {
        "query": {
            "term": {
                "title": title
            }
        },
        "size": 1000
        # "_source": ["title", "tagged_entities", "content"]
    }


def query_count_apperance_of_entities(title: str) -> dict:
    return {
        "size": 0,
        "aggs": {
            "entities": {
                "filter": {
                    "term": {
                        "title": title
                    }
                },
                "aggs": {
                    "all_entities": {
                        "terms": {
                            "field": "tagged_entities.type.keyword",
                            "size": 500,
                            "order": {
                                "_count": "asc"
                            }
                        }
                    }
                }
            }
        }
    }


def query_aggergate_by_type(count_entity_types, count_per_type) -> dict:
    return {
        "size": 0,
        "aggs": {
            "x_terms": {
                "terms": {
                    "field": "tagged_entities.type.keyword",
                    "size": count_entity_types
                },
                "aggs": {
                    "top_docs": {
                        "top_hits": {
                            "size": count_per_type
                        }
                    }
                }
            }
        }
    }


def query_get_by_entity_type(entity_type: str) -> dict:
    return {
        "query": {
            "term": {
                "tagged_entities.type.keyword": entity_type
            }
        },
        "size": 20,
        "_source": ["tagged_entities", "content"]
    }


def query_get_by_fine_grained_fewnerd(fine_grained_type: str | list[str]) -> dict:
    fine_grained_type = fine_grained_type if isinstance(fine_grained_type, list) else [fine_grained_type]
    return {
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "filter": [
                            {
                                "script": {
                                    "script": {
                                        "source": "doc['tagging.fine_type.keyword'].length > 1"
                                    }
                                }
                            },
                            {
                                "terms": {
                                    "tagging.fine_type.keyword": fine_grained_type
                                }
                            }
                        ]
                    }
                },
                "boost": "5",
                "random_score": {
                    "seed": 12345678910,
                    "field": '_seq_no'
                },
                "boost_mode": 'sum'
            }
        },
        "sort": [
            {"tagging.fine_type.keyword": {"order": "asc"}}
        ],
        "size": 200
    }


def query_get_by_fine_grained_fewnerd_v3_randomized(fine_grained_type: str | list[str], batch_size,
                                                    llm_layer=None) -> dict:
    fine_grained_type = fine_grained_type if isinstance(fine_grained_type, list) else [fine_grained_type]
    query = {
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "filter": [
                            {
                                "terms": {
                                    "fine_type": fine_grained_type
                                }
                            }
                        ]
                    }
                },
                "boost": "5",
                "random_score": {
                    "field": "_seq_no"
                },
                "boost_mode": "sum"
            }
        },
        "size": batch_size
    }

    if llm_layer:
        query["_source"] = all_source_except_other_embedding(llm_layer)

    return query


def all_source_except_other_embedding(llm_layer):
    return [
        "all_text",
        "coarse_type",
        "fine_type",
        "doc_id",
        "text_id",
        "phrase",
        f"embedding.{llm_layer}.start",
        f"embedding.{llm_layer}.end"
    ]


def fewnerd_random_results_per_fine_type(fine_types, instances_per_type=100):
    query = {
        "size": 0,
        "aggs": {
            "filter_types": {
                "filter": {"terms": {"fine_type": fine_types}},
                "aggs": {
                    "top_artifacts": {
                        "composite": {
                            "sources": [
                                {
                                    "artifact": {
                                        "terms": {
                                            "field": "fine_type"
                                        }
                                    }
                                }
                            ], "size": 1

                        },
                        "aggs": {
                            "hits": {
                                "top_hits": {
                                    "size": instances_per_type,
                                    "sort": [
                                        {
                                            "_script": {
                                                "type": "number",
                                                "script": {
                                                    "source": "Math.random()"
                                                },
                                                "order": "asc"
                                            }
                                        }
                                    ]
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return query


def query_get_by_fine_grained_fewnerd_v3_unrandomized(fine_grained_type: str | list[str], batch_size) -> dict:
    fine_grained_type = fine_grained_type if isinstance(fine_grained_type, list) else [fine_grained_type]
    return {
        "query": {
            "terms": {
                "fine_type": fine_grained_type
            }
        },
        "sort": [
            {"fine_type": {"order": "asc"}}
        ],
        "size": batch_size
    }

def query_search_by_similarity(embedding, layer, default_filter_query=None):
    filter_query = default_filter_query or {"match_all":{}}
    return {
        "script_score": {
            "query": filter_query,
            "script": {
                "source": f"cosineSimilarity(params.query_vector, '{layer}') + 1.0",
                "params": {
                    "query_vector": embedding
                }
            }
        }
    }


def query_get_by_fine_grained_fewnerd_v3(fine_grained_type: str | list[str], randomized=True,
                                         batch_size: int = 200, llm_layer=None) -> dict:
    fine_grained_type = fine_grained_type if isinstance(fine_grained_type, list) else [fine_grained_type]
    if randomized:
        return query_get_by_fine_grained_fewnerd_v3_randomized(fine_grained_type, batch_size, llm_layer=llm_layer)
    return query_get_by_fine_grained_fewnerd_v3_unrandomized(fine_grained_type, batch_size)


def query_get_by_coarse_grained_fewnerd(coarse_grained_type: str) -> dict:
    return {
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "filter": [
                            {
                                "script": {
                                    "script": {
                                        "source": "doc['tagging.coarse_type'].length > 1"
                                    }
                                }
                            },
                            {
                                "term": {
                                    "tagging.coarse_type": coarse_grained_type
                                }
                            }
                        ]
                    }

                },

                "boost": "5",
                "random_score": {
                    "seed": 12345678910,
                    "field": '_seq_no'
                },
                "boost_mode": 'sum',
            }
        },
        "size": 200
    }


def query_hard_negative(fine_grained_type: str | list[str],
                        coarse_grained_type: str,
                        anchor_text: str,
                        size: int,
                        llm_layer=None) -> dict:
    assert isinstance(fine_grained_type, list) or isinstance(fine_grained_type,
                                                             str), "fine_grained_type should be a string or a list of strings"
    assert isinstance(coarse_grained_type, str), "coarse_grained_type should be a string"
    assert isinstance(anchor_text, str), "term_to_be_closed_to should be a string"
    fine_grained_type = fine_grained_type if isinstance(fine_grained_type, list) else [fine_grained_type]
    query = {
        "query": {
            "bool": {
                "filter": [
                    {
                        "terms": {
                            "fine_type": fine_grained_type
                        }
                    }
                ],
                "should": [
                    {
                        "more_like_this": {
                            "fields": [
                                "all_text"
                            ],
                            "like": anchor_text,
                            "min_term_freq": 1,
                            "min_doc_freq": 1
                        }
                    },
                    {
                        "term": {
                            "coarse_type": {
                                "value": coarse_grained_type,
                                "boost": 5
                            }
                        }
                    }
                ]
            }
        },
        "size": size
    }
    if llm_layer:
        query["_source"] = all_source_except_other_embedding(llm_layer)

    return query
