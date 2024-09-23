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
                                        "source": "doc['tagging.fine_type'].length > 1"
                                    }
                                }
                            },
                            {
                                "terms": {
                                    "tagging.fine_type": fine_grained_type
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
            {"tagging.fine_type": {"order": "asc"}}
        ],
        "size": 20
    }


def query_get_by_fine_grained_fewnerd_v3_randomized(fine_grained_type: str | list[str], batch_size) -> dict:
    fine_grained_type = fine_grained_type if isinstance(fine_grained_type, list) else [fine_grained_type]
    return {
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
                    "seed": 12345678910,
                    "field": "_seq_no"
                },
                "boost_mode": "sum"
            }
        },
        "size": batch_size
    }


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


def query_get_by_fine_grained_fewnerd_v3(fine_grained_type: str | list[str], randomized=True, batch_size:int=200) -> dict:
    fine_grained_type = fine_grained_type if isinstance(fine_grained_type, list) else [fine_grained_type]
    if randomized:
        return query_get_by_fine_grained_fewnerd_v3_randomized(fine_grained_type, batch_size)
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
