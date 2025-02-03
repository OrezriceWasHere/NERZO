import math
from functools import partial
import dataset_provider
import torch
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


async def yield_dataset(anchor_type, dataset_types, batch_size=50,
                        instances_per_type=100,
                        hard_negative_ratio=0,
                        llm_layer=None):
    extract = extract_entities_from_es_response

    batches = [
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

        anchor = anchor["_source"]
        chunked_good_batch = extract(good_batch)
        chunked_bad_batch = await negative_examples(
            coarse_type=coarse_type,
            fine_types=other_types,
            batch_size=batch_size,
            llm_layer=llm_layer,
            anchor_text=text,
            hard_negative_ratio=hard_negative_ratio
        )
        yield anchor, chunked_good_batch, chunked_bad_batch


async def negative_examples(
        coarse_type,
        fine_types,
        batch_size,
        llm_layer,
        anchor_text,
        hard_negative_ratio):
    hard_negative_amount = math.ceil(hard_negative_ratio * batch_size)
    easy_negative_amount = batch_size - hard_negative_amount
    hard_negative, easy_negative = [], []
    if hard_negative_amount > 0:
        hard_negative = await dataset_provider.get_hard_negative_fewnerd(fine_types=fine_types,
                                                                         coarse_type=coarse_type,
                                                                         anchor_text=anchor_text,
                                                                         batch_size=hard_negative_amount,
                                                                         llm_layer=llm_layer)
    if easy_negative_amount > 0:
        easy_negative = await dataset_provider.get_randomized_by_fine_type_fewnerd_v4(fine_types,
                                                                                      batch_size=easy_negative_amount,
                                                                                      llm_layer=llm_layer)
    bad_batch = extract_entities_from_es_response(hard_negative) + \
    extract_entities_from_es_response(easy_negative)

    return bad_batch


async def yield_train_dataset(anchor_type, **kwargs):
    all_train_fine_types = train_fine_types()
    assert anchor_type in all_train_fine_types
    async for a, good, bad in yield_dataset(anchor_type, all_train_fine_types, **kwargs):
        yield a, good, bad


async def yield_test_dataset(anchor_type, **kwargs):
    all_test_fine_types = test_fine_types()
    assert anchor_type in all_test_fine_types
    kwargs = {"hard_negative_ratio": 0.0, **kwargs}
    async for a, good, bad in yield_dataset(anchor_type, all_test_fine_types, **kwargs):
        yield a, good, bad


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
    return [docu["_source"] for docu in response] if response else []


def choose_llm_representation(end, start, input_tokens):
    assert input_tokens in __input_token_factory, f"input_tokens should be one of {list(__input_token_factory.keys())} but got {input_tokens}"
    return __input_token_factory[input_tokens](end, start)


__input_token_factory = {
    "diff": lambda end, start: (torch.tensor(end) - torch.tensor(start)),
    "end": lambda end, start: torch.tensor(end),
    "start_end_pair": lambda end, start: torch.concat((torch.tensor(end), torch.tensor(start)))
}


def pick_llm_output_for_document(device, input_tokens, llm_layer, is_fine_tune_llm, documents: list[dict], llm=None):
    llm_representation = partial(choose_llm_representation, input_tokens=input_tokens)

    if not is_fine_tune_llm:
        end_representation = [item["embedding"][llm_layer]["end"] for item in documents]
        start_representation = [item["embedding"][llm_layer]["start"] for item in documents]
        stack = torch.stack(
            [llm_representation(end, start) for end, start in zip(end_representation, start_representation)]).to(device)
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
        stack = torch.stack(
            [llm_representation(end, start) for end, start in zip(end_representation, start_representation)]).to(device)
        return stack


def compute_accuracy_at_prediction(predictions: list[float], ground_truths: list[int]) -> pd.DataFrame:
    p_numpy = np.asarray(predictions)
    gt_numpy = np.asarray(ground_truths)
    accuracies = [accuracy_score(gt_numpy, p_numpy >= prediction) for prediction, ground_truth in
                  zip(predictions, ground_truths)]
    return pd.DataFrame({"prediction": p_numpy,
                         "ground_truth": gt_numpy,
                         "accuracy_if_threshold_was_here": np.asarray(accuracies)})


def retrieval_anchors_test():
    return {'GPE': ['fnd_967e47a3ca869cdffcb8f9c33379e9bfd02919be', 'fnd_1d6bf5d1bd81c9d50c587526d74e73fc48ac7a7c',
             'fnd_f7a19445167c64404c111b88af975849c2ae664c'],
     'actor': ['fnd_e86b0466579aa12574d18f55847367dab958f7d4', 'fnd_7f1ff4853316bff31d30ac344727bbf958139351',
               'fnd_2e2a18bfdcdff1508d8ef418ad08d63c70936eaa'],
     'athlete': ['fnd_55008cb5512fe1c70fba2224afec0773619751eb', 'fnd_a409da2f46cf5e2c3292b37d059d0a55b110e77d',
                 'fnd_48c7ce2884533983d83cfb24dde6642608f3bd18'],
     'award': ['fnd_7f5f0f61601f35ec58ac2ee6e7468aae252bc2ce', 'fnd_175d46ff533c4e1711ae6b7a26c0df739ec6dc42',
               'fnd_0b517bd4d62012ba10d24493fdbfc69a68ced33a'],
     'car': ['fnd_84ecf06c77af91dfe052a42b70823e294260a3f7', 'fnd_77a49ca79148ec22e3d59ee53d9c1dad13ed42c2',
             'fnd_da731871908819190cae129b5402e7bb56b5ce8f'],
     'currency': ['fnd_3ddd791774f523d16e876b238dfbd451c7af61b2', 'fnd_3a0db0a8ed4ce699c8b3b38fa53533027c907d48',
                  'fnd_3e75a890f555ca1efe06b0155f4f3f41a7424d4d'],
     'election': ['fnd_d087dd25ba794a1465cd1860c4b67a42726b8515', 'fnd_752bf3742b9c69f10705997b9ce6f3dd8690ff7a',
                  'fnd_39c150d389a1c8ed7cc521c25226784ae5a48957'],
     'film': ['fnd_b9e5fad97fce1238a512b8b5e7b7218ba0234e24', 'fnd_5ac8174648196ed5d76bccc29c3211f2004764f0',
              'fnd_7319d1458507ea8eb0ad9581ec8fbe7090f91d8b'],
     'food': ['fnd_8a2a54457bc1da981730e273333bb36ea44e9f9e', 'fnd_8ecab54190b8a4be83282ca3c66e4e6c9f903cf4',
              'fnd_fe9807ad5dd7e6f171dda46f508ffa7a54bbd541'],
     'hotel': ['fnd_f08cac10e475e4c0907225af34ff64503f90d2c3', 'fnd_ac901798d4a1b6ffa1066da0b17ffdabd772daff',
               'fnd_29ea32359e66e2c13ec3f06791eb82c1c05c9a18'],
     'island': ['fnd_7673b8c30192432d564cc5d5e85363b587f2515b', 'fnd_bd83781aeb6b0fa8c5032460c474dcf3eeca3678',
                'fnd_2e7cc27b9e14b560389f0a23e1696cbc8a40cbb5'],
     'law': ['fnd_117cb12f9fbd09d294b5af25721ae84baf27f152', 'fnd_0979c42d6ea161902dc38dc81022cc7d522ea2f5',
             'fnd_52a8095fdc271237bb50303019d6b23c5115fdad'],
     'media/newspaper': ['fnd_7d1ee6c9d33823aba1a9c081312c1f40c33c9261',
                         'fnd_b3e6f85de3543d83178e433a68b38a628d7d785a',
                         'fnd_e247f01d828009bc46fbf8fb4f1ebabe9f65cfa1'],
     'park': ['fnd_7e2d6363593c4e2a95f7a25bddd01693ff7f8342', 'fnd_0d01713092cb89d2f553cb60833c6eba324e5977',
              'fnd_28ac27d4a82c2f03abc6c5fbdbb472a7d26dd22c'],
     'politicalparty': ['fnd_21efe75bfab4602a936d57aa295a539929d04c62',
                        'fnd_3c0d1690fa02b1ecd951b74121fe1ea56148ebbf',
                        'fnd_f44ae07b4cb1c080fe5b523521b101757268c5cd'],
     'religion': ['fnd_1db484ae44f5977f590063ea12d77351ab10f559', 'fnd_6508339a13ee81763f96b3e684fc280a98e8abe9',
                  'fnd_7b23ea6d1fa52825c704b6bcd451e94fdf22cfc5'],
     'ship': ['fnd_d8422866ecd7524161b53f79b96dd41ff85106f3', 'fnd_3da716ac298d2d2785a8777d469f39ce6a6bc007',
              'fnd_29edc37d74b77cf51194867a72aa3b375a58ec72'],
     'software': ['fnd_f777427666ce48f6f656003e1f256ff89946e1b0', 'fnd_55c2b6cffcf452844758258f5d89eec0e4db9d88',
                  'fnd_0fe956c81957d253eefd349a71f5de99906f0a07'],
     'sportsfacility': ['fnd_316ff3589fd101bcd3f901bab7a4178beec6e4b8',
                        'fnd_d8e7750cd289df498f80386434a70a92586d999e',
                        'fnd_3c2664d0ca67bfcbf90f8990e258c808fbb898a9'],
     'weapon': ['fnd_6c321b13ee47716535a54193f3ae82b367ef83df', 'fnd_31c7a1f743d57c6d13ec75b639c6ff9e7dadcbc5',
                'fnd_1f7825e61577a9471f33dd840a79943ad3e10520']
     }
