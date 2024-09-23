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

    return yield_dataset(good_types, bad_types, batch_size=batch_size)


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
    return yield_dataset(good_types, bad_types, batch_size=batch_size)


def extract_entities_from_es_response(response):
    return [docu["_source"] for docu in response]
