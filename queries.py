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


def query_aggerage_by_type() -> dict:
    return {
        "size": 0,
        "aggs": {
            "x_terms": {
                "terms": {
                    "field": "tagged_entities.type.keyword",
                    "size": 10
                },
                "aggs": {
                    "top_docs": {
                        "top_hits": {
                            "size": 20
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
        "size": 1000,
        "_source": ["tagged_entities", "content"]
    }