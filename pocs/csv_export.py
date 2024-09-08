import random
from tqdm import tqdm

import dataset_provider
from dataset_provider import get_top_results_for_entities
from copy import deepcopy


def pick_best_example_for_entity(bucket):
    bucket_expected_type = bucket["key"]
    bucket_docs = bucket["top_docs"]["hits"]["hits"]
    all_good_examples = []
    for doc in bucket_docs:
        doc_source = doc["_source"]
        # I want per a single bucket a single hit contating both an entity of the expected type, and one
        # entity that is not of the expected type
        entity_with_bucket_type = random.choice([entity
                                                 for entity in doc_source["tagged_entities"]
                                                 if entity["type"] == bucket_expected_type])
        another_distinct_entity = [entity
                                   for entity in doc_source["tagged_entities"]
                                   if entity["type"] != bucket_expected_type and
                                   entity["entity_in_text"] != entity_with_bucket_type["entity_in_text"]]
        if another_distinct_entity:
            another_distinct_entity = random.choice(another_distinct_entity)
            modified_entities_document = deepcopy(doc_source)
            modified_entities_document["tagged_entities"] = [entity_with_bucket_type, another_distinct_entity]
            all_good_examples.append(modified_entities_document)

    return all_good_examples


def generate_examples_with_two_different_entities():
    write_index = "narrowed_example"
    all_entities = get_top_results_for_entities()
    for bucket in tqdm(all_entities["aggregations"]["x_terms"]["buckets"]):
        bucket_name = bucket["key"]
        best_examples = pick_best_example_for_entity(bucket)
        for example in best_examples:
            dataset_provider.write_to_index(example, write_index)


def export_csv_bucket(bucket):
    # I want to read the bucket, get the item in the bucket,
    # for the entity with the key the same as the bucket key I declare 'original=True'
    # and for the other one I declare 'original=False'
    # I export content, the two entities, and the original flag to a csv file
    bucket_key = bucket["key"]
    export_params = [
        "para_index",
        "title",
        "doc_id",
        "content"
    ]
    outputs = []
    for doc in bucket["top_docs"]["hits"]["hits"]:
        doc_source = doc["_source"]
        entity_with_bucket_type = random.choice([entity
                                                 for entity in doc_source["tagged_entities"]
                                                 if entity["type"] == bucket_key])
        other_entity = random.choice([entity
                                      for entity in doc_source["tagged_entities"]
                                      if entity["type"] != bucket_key])
        result_true = {
            "para_index": doc_source["para_index"],
            "title": doc_source["title"],
            "doc_id": doc_source["doc_id"],
            "entity_type": bucket_key,
            "entity_text": entity_with_bucket_type["entity_in_text"],
            "entity_name": entity_with_bucket_type["entity_name"],
            "ground_truth_answer": True,
            "content": doc_source["content"]
        }

        result_false = {
            "para_index": doc_source["para_index"],
            "title": doc_source["title"],
            "doc_id": doc_source["doc_id"],
            "entity_type": bucket_key,
            "entity_text": other_entity["entity_in_text"],
            "entity_name": other_entity["entity_name"],
            "ground_truth_answer": False,
            "content": doc_source["content"]
        }

        outputs.append(result_true)
        outputs.append(result_false)

    return outputs

def create_csv():
    all_entities = get_top_results_for_entities(count_entity_types=500,
                                                count_per_type=1,
                                                index="narrowed_example")
    written_header = False
    with open(f"../output.tsv", "w") as f:

        for bucket in tqdm(all_entities["aggregations"]["x_terms"]["buckets"]):
            outputs = export_csv_bucket(bucket)
            for output in outputs:
                if not written_header:
                    f.write("\t".join(output.keys()) + "\n")
                    written_header = True

                f.write("\t".join([str(value) for value in output.values()]) + "\n")

if __name__ == "__main__":
    create_csv()
