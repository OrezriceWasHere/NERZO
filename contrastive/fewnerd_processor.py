import dataset_provider
from collections import Counter
from random import shuffle
import torch
import entity_classifier
from llm_interface import LLMInterface


def yield_dataset(llm:LLMInterface, llm_hidden_layer: str) -> tuple[torch.Tensor]:
    types = [
        "language",
        "athlete",
        "mountain",
        "music",
        "company",
        "politician",
        "sportsteam",
        "award",
        "disease",
        "island"
    ]

    for sentences in dataset_provider.yield_by_fine_grained_type_fewnerd(types):
        same_type, different_type = process_batch(sentences)


        yield same_type, different_type

def most_popular_entity_in_batch(batch):
    most_popular_entity = Counter(
        [entity["fine_type"] for tagging in batch for entity in tagging["_source"]["tagging"] if
         entity["fine_type"] != "other"]).most_common(1)[0][0]

    return most_popular_entity

def process_batch(batch):
    most_popular_entity = most_popular_entity_in_batch(batch)
    same_type, different_type = [], []
    for sentence in batch:
        sentence_text = sentence["_source"]["full_text"]
        entities_of_fine_type = entity_classifier.pick_entities_of_finetype_fewnerd(sentence["_source"]["tagging"],
                                                                                    most_popular_entity)

        for entity in sentence["_source"]["tagging"]:
            entity_info = {"text": sentence_text,
                           "index_start": entity["index_start"],
                           "index_end": entity["index_end"],
                           "phrase": entity["phrase"].strip(),
                           "id": entity["text_id"]}
            if entity in entities_of_fine_type:
                same_type.append(entity_info)
            else:
                different_type.append(entity_info)

    shuffle(same_type)
    shuffle(different_type)
    different_type = different_type[:len(same_type)]
    return same_type, different_type


