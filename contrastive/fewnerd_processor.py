from functools import partial

import dataset_provider
import torch
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

async def yield_dataset(anchor_type, dataset_types, batch_size=50,
                        instances_per_type=100,
                        llm_layer=None,
                        should_use_hard_negative=True):
    extract = extract_entities_from_es_response

    batches  = [
        (start_index, min(start_index + batch_size, instances_per_type))
        for start_index in range(0, instances_per_type, batch_size)
    ]
    batch_sizes = [end - start for start, end in batches]

    for batch_size in batch_sizes:
        anchor = await dataset_provider.get_randomized_by_fine_type_fewnerd_v4(anchor_type,
                                                                               batch_size=1,
                                                                               llm_layer=llm_layer)
        assert len(anchor) == 1
        anchor = anchor[0]
        result_type = anchor["_source"]["fine_type"]
        coarse_type = anchor["_source"]["coarse_type"]
        text = anchor["_source"]["all_text"]
        good_batch = await dataset_provider.get_randomized_by_fine_type_fewnerd_v4(result_type,
                                                                                   batch_size=batch_size,
                                                                                   llm_layer=llm_layer)
        other_types = list(set(dataset_types) - {result_type})
        if should_use_hard_negative:
            bad_batch = await dataset_provider.get_hard_negative_fewnerd(fine_types=other_types,
                                                                         coarse_type=coarse_type,
                                                                         anchor_text=text,
                                                                         batch_size=batch_size,
                                                                         llm_layer=llm_layer)
        else:
            bad_batch = await dataset_provider.get_randomized_by_fine_type_fewnerd_v4(other_types,
                                                                                      batch_size=batch_size,
                                                                                      llm_layer=llm_layer)

        anchor  = anchor["_source"]
        chunked_good_batch = extract(good_batch)
        chunked_bad_batch = extract(bad_batch)
        yield anchor, chunked_good_batch, chunked_bad_batch

async def yield_train_dataset(anchor_type, **kwargs):
    all_train_fine_types = train_fine_types()
    assert anchor_type in all_train_fine_types
    async for a, good, bad in yield_dataset(anchor_type, all_train_fine_types, **kwargs):
        yield  a, good, bad

async def yield_test_dataset(anchor_type, **kwargs):
    all_test_fine_types = test_fine_types()
    assert anchor_type in all_test_fine_types
    kwargs = {"should_use_hard_negative": False, **kwargs}
    async for a, good, bad in yield_dataset(anchor_type, all_test_fine_types, **kwargs):
        yield  a, good, bad

def train_fine_types():
    return ['education', 'airport', 'restaurant', 'sportsleague', 'disease', 'hospital', 'painting', 'other',
            'library', 'sportsevent', 'soldier', 'game', 'educationaldegree', 'broadcastprogram', 'mountain',
            'road/railway/highway/transit', 'company', 'politician', 'attack/battle/war/militaryconflict',
            'astronomything', 'language', 'train', 'scholar', 'bodiesofwater', 'chemicalthing', 'director',
            'showorganization', 'writtenart', 'disaster', 'medical', 'music', 'airplane', 'biologything', 'theater',
            'sportsteam', 'government/governmentagency', 'livingthing', 'artist/author', 'protest', 'god']


def test_fine_types(batch_size=50, instances_per_type=100, llm_layer=None):
    return ['island', 'athlete', 'politicalparty', 'actor', 'software', 'sportsfacility', 'weapon', 'food', 'election',
            'car', 'currency', 'park', 'award', 'GPE', 'media/newspaper', 'law', 'religion', 'film', 'hotel', 'ship']


def extract_entities_from_es_response(response):
    return [docu["_source"] for docu in response]

def choose_llm_representation(end, start, input_tokens):

    assert input_tokens in __input_token_factory, f"input_tokens should be one of {list(__input_token_factory.keys())} but got {input_tokens}"
    return __input_token_factory[input_tokens](end, start)


__input_token_factory = {
    "diff": lambda end, start: (torch.tensor(end) - torch.tensor(start)),
    "end":  lambda end, start: torch.tensor(end),
    "start_end_pair": lambda end, start: torch.concat((torch.tensor(end), torch.tensor(start)))
}

def pick_llm_output_for_document(device, input_tokens, llm_layer, is_fine_tune_llm, documents: list[dict], llm=None):

    llm_representation = partial(choose_llm_representation, input_tokens=input_tokens)


    if not is_fine_tune_llm:
        end_representation = [item["embedding"][llm_layer]["end"] for item in documents]
        start_representation = [item["embedding"][llm_layer]["start"] for item in documents]
        stack = torch.stack([llm_representation(end, start) for end, start in zip(end_representation, start_representation)]).to(deivce)
        return stack

    else:
        assert llm is not None, "llm should be provided if is_fine_tune_llm is True"
        texts = [doc["all_text"] for doc in documents]
        texts_indices = [(doc["index_start"], doc["index_end"]) for doc in documents]
        tokens = llm.tokenize(texts).to(device)
        hidden_items = llm.get_llm_at_layer(tokens, llm_layer, clone=False)
        token_indices = [llm.token_indices_given_text_indices(text, text_indices) for text, text_indices in
                         zip(texts, texts_indices)]
        end_representation = [h[token_index[1]] for h, token_index in zip(hidden_items, token_indices)]
        start_representation = [h[token_index[0]] for h, token_index in zip(hidden_items, token_indices)]
        stack = torch.stack([llm_representation(end, start) for end, start in zip(end_representation, start_representation)]).to(device)
        return stack


def compute_accuracy_at_prediction(predictions: list[float], ground_truths: list[int]) -> pd.DataFrame:
    p_numpy = np.asarray(predictions)
    gt_numpy = np.asarray(ground_truths)
    accuracies = [accuracy_score(gt_numpy, p_numpy >= prediction) for prediction, ground_truth in
                  zip(predictions, ground_truths)]
    return pd.DataFrame({"prediction": p_numpy,
                         "ground_truth": gt_numpy,
                         "accuracy_if_threshold_was_here": np.asarray(accuracies)})
