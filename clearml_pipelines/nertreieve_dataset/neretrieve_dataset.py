datasets = [
	{
		"url": "https://storage.googleapis.com/neretrieve_dataset/supervised_ner/NERetrive_sup_test.jsonl.bz2",
		"name": "test-supervised.txt",
		"json": "test-supervised.json",
		"env": "test"
	},
	{
		"url": "https://storage.googleapis.com/neretrieve_dataset/supervised_ner/NERetrive_sup_train.jsonl.bz2",
		"name": "train-supervised.txt",
		"json": "train-supervised.json",
		"env": "train"
	}
]

elasticsearch_storage_mapping = {
	"mappings": {
		"properties": {
			"all_text": {
				"type": "text"
			},
			"doc_id": {
				"type": "keyword"
			},
			"entity_type": {
				"type": "keyword"
			},
			"index_start": {
				"type": "short"
			},
			"index_end": {
				"type": "short"
			},
			"index_in_text": {
				"type": "short"
			},
			"phrase": {
				"type": "text"
			},
			"text_id": {
				"type": "keyword"
			},
			"entity_id": {
				"type": "keyword"
			},
			"embedding": {
				"type": "object",
				"properties": {
					"llama_3_17_v_proj": {
						"type": "object",
						"properties": {
							"end": {
								"type": "dense_vector",
								"dims": 1024,
								"index": False
							},
							"start": {
								"type": "dense_vector",
								"dims": 1024,
								"index": False
							}
						}

					},
					"llama_3_entire_model": {
						"type": "object",
						"properties": {
							"end": {
								"type": "dense_vector",
								"dims": 4096,
								"index": False

							},
							"start": {
								"type": "dense_vector",
								"dims": 4096,
								"index": False

							}
						}

					},
					"llama_3_31_v_proj": {
						"type": "object",
						"properties": {
							"end": {
								"type": "dense_vector",
								"dims": 1024,
								"index": False

							},
							"start": {
								"type": "dense_vector",
								"dims": 1024,
								"index": False

							}
						}

					},
					"llama_3_3_13_k_proj": {
						"type": "object",
						"properties": {
							"end": {
								"type": "dense_vector",
								"dims": 1024,
								"index": False

							},
							"start": {
								"type": "dense_vector",
								"dims": 1024,
								"index": False

							}
						}
					}
				}
			}
		}
	},
	"settings": {
		"index": {
			"max_inner_result_window": 1000000,
			"number_of_shards": 60,
			"number_of_replicas": 0
		}
	}
}

