datasets = [
    {
        "url": "https://huggingface.co/datasets/Rosenberg/fewnerd/resolve/main/dev-supervised.txt",
        "name": "dev-supervised.txt",
        "json": "dev-supervised.json",
        "env": "dev"
    },
    {
        "url": "https://huggingface.co/datasets/Rosenberg/fewnerd/resolve/main/test-supervised.txt",
        "name": "test-supervised.txt",
        "json": "test-supervised.json",
        "env": "test"
    },
    {
        "url": "https://huggingface.co/datasets/Rosenberg/fewnerd/resolve/main/train-supervised.txt",
        "name": "train-supervised.txt",
        "json": "train-supervised.json",
        "env": "train"
    }
]

elasticsearch_tests_mapping = {
    "mappings": {
        "properties": {
            "all_text": {
                "type": "text",
                "term_vector": "yes"
            },
            "coarse_type": {
                "type": "keyword"
            },
            "fine_type": {
                "type": "keyword"
            },
            "index_end": {
                "type": "integer"
            },
            "index_start": {
                "type": "integer"
            },
            "phrase": {
                "type": "text",
                "term_vector": "yes"
            },
            "text_id": {
                "type": "keyword"
            },
            "doc_id": {
                "type": "keyword"
            },
            "embedding": {
                "type": "object",
                "properties": {
                    "llama_3_17_v_proj": {
                        "type": "dense_vector",
                        "dims": 1024,
                        "index": "true",
                        "similarity": "cosine",
                        "index_options": {
                            "type": "flat"
                        }

                    },
                    "llama_3_entire_model": {
                        "type": "dense_vector",
                        "dims": 4096,
                        "index": "true",
                        "similarity": "cosine",
                        "index_options": {
                            "type": "flat"
                        }

                    },
                    "llama_3_31_v_proj": {
                        "type": "dense_vector",
                        "dims": 1024,
                        "index": "true",
                        "similarity": "cosine",
                        "index_options": {
                            "type": "flat"
                        }

                    },
                    "llama_3_3_13_k_proj": {
                        "type": "dense_vector",
                        "dims": 1024,
                        "index": "true",
                        "similarity": "cosine",
                        "index_options": {
                            "type": "flat"
                        }
                    }
                }
            }
        }
    },
    "settings": {
        "index": {
            "max_inner_result_window": 1000000
        }
    }
}

elasticsearch_storage_mapping = {
    "mappings": {
        "properties": {
            "all_text": {
                "type": "text",
                "term_vector": "yes"
            },
            "coarse_type": {
                "type": "keyword"
            },
            "fine_type": {
                "type": "keyword"
            },
            "index_end": {
                "type": "integer"
            },
            "index_start": {
                "type": "integer"
            },
            "phrase": {
                "type": "text",
                "term_vector": "yes"
            },
            "text_id": {
                "type": "keyword"
            },
            "doc_id": {
                "type": "keyword"
            },
            "embedding": {
                "type": "object",
                "properties": {
                    "llama_3_17_v_proj": {
                        "type": "dense_vector",
                        "dims": 1024,
                        "index": "true",
                        "similarity": "cosine",
                        "index_options": {
                            "type": "flat"
                        }

                    },
                    "llama_3_entire_model": {
                        "type": "dense_vector",
                        "dims": 4096,
                        "index": "true",
                        "similarity": "cosine",
                        "index_options": {
                            "type": "flat"
                        }

                    },
                    "llama_3_31_v_proj": {
                        "type": "dense_vector",
                        "dims": 1024,
                        "index": "true",
                        "similarity": "cosine",
                        "index_options": {
                            "type": "flat"
                        }

                    },
                    "llama_3_3_13_k_proj": {
                        "type": "dense_vector",
                        "dims": 1024,
                        "index": "true",
                        "similarity": "cosine",
                        "index_options": {
                            "type": "flat"
                        }
                    }
                }
            }
        }
    },
    "settings": {
        "index": {
            "number_of_replicas": 0,
            "number_of_shards": 1,
            "max_inner_result_window": 1000000
        }
    }
}

elasticsearch_mapping_for_full_sentence = {
    "mappings": {
        "properties": {
            "all_text": {
                "type": "text",
                "term_vector": "yes"
            },
            "text_id": {
                "type": "keyword"
            },
            "tagging": {
                "type": "nested",
                "properties": {
                    "coarse_type": {
                        "type": "keyword"
                    },
                    "fine_type": {
                        "type": "keyword"
                    },
                    "index_end": {
                        "type": "integer"
                    },
                    "index_start": {
                        "type": "integer"
                    },
                    "phrase": {
                        "type": "text",
                        "term_vector": "yes"
                    }
                }
            }

        }
    },
    "settings": {
        "index": {
            "max_inner_result_window": 1000000
        }
    }
}

elasticsearch_fine_type_to_embedding = {
    "mappings": {
        "properties": {
            "entity_id": {
                "type": "keyword"
            },
            "entity_name": {
                "type": "keyword"
            },
            "entity_description": {
                "type": "keyword"
            },
            "llm_layer": {
                "type": "keyword"
            },
            "embedding": {
                "type": "object"
            }
        }
    },
    "settings": {
        "index": {
            "max_inner_result_window": 1000000
        }
    }
}


def llm_and_layer_to_elastic_name(llm_id, layer=None):
    if layer is None:
        layer = "last"
    return f'{llm_id}@{layer}'.replace(".", "-").lower()


def schema_llm_layer(size, layer_name):
    return {
        layer_name: {
            "type": "dense_vector",
            "dims": size
        }
    }
