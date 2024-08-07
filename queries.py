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


def query_get_by_fine_grained_fewnerd(fine_grained_type: str) -> dict:
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
                                "term": {
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
        "size": 100
    }


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
