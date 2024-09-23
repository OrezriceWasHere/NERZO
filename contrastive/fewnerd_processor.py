import dataset_provider
from collections import Counter
from random import shuffle
import torch
import entity_classifier


def yield_dataset(good_types, bad_types, require_even_length=False, batch_size:int=200) -> tuple[torch.Tensor]:
    extract = lambda x: extract_entities_from_es_response(x)

    for good_sentences, bad_sentences in zip(
            dataset_provider.yield_by_fine_grained_type_fewnerd_v3(good_types, randomize=False),
            dataset_provider.yield_by_fine_grained_type_fewnerd_v3(bad_types, randomize=True)):

        good_sentences = extract(good_sentences)
        bad_sentences = extract(bad_sentences)

        max_length = min(len(good_sentences), len(bad_sentences))
        good_sentences = good_sentences[:max_length]
        bad_sentences = bad_sentences[:max_length]


        # There may be a case where the good type entity switches in the middle of the batch
        # we need to find the index where the switch happens and split the batch there
        # so that we can have a batch with the same type and a batch with different types
        changing_index = [i for i, (d1, d2) in enumerate(zip(good_sentences, good_sentences[1:])) if
                          d1["fine_type"] != d2["fine_type"]]
        changing_index = [0] + changing_index + [len(good_sentences)]
        for start, end in zip(changing_index, changing_index[1:]):
            if require_even_length:
                # verify batch is of of even size
                end = end - (end - start) % 2
            if end - start < 2:
                continue
            yield good_sentences[start:end], bad_sentences[start:end]


def yield_train_dataset(batch_size: int = 200) -> tuple[torch.Tensor]:
    good_types = [
        "language",
        "athlete",
        "mountain",
        "music",
        "company"
    ]

    bad_types = [
        "politician",
        "sportsteam",
        "award",
        "disease",
        "island"
    ]

    return yield_dataset(good_types, bad_types, batch_size)


def yield_test_dataset(batch_size: int = 200) -> tuple[torch.Tensor]:
    good_types = [
        "weapon",
        "park",
        "hospital"
        "game",

    ]
    bad_types = [
        "ship",
        "library",
        "soldier",
        "medical"
    ]
    return yield_dataset(good_types, bad_types, batch_size)


def extract_entities_from_es_response(response):
    return [docu["_source"] for docu in response]


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
