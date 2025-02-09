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


def retrieve_anchors_for_sentence_test():
    return {"GPE": ["6c528775b3eacae0d7e5329edc876d80102957dc", "ac89780afe227bf6be3ccc770327dc290f20f02b",
                    "cba5f8226655e1b353db014dd5985e6fd91d9c84"],
            "athlete": ["fd2496a2b611d11bd3d9ca4d8059ddc00597ea9f", "10a6571a2f37b3c8ef4845e5deeba587c02fb28d",
                        "f7c38f57e917aa6bb5719ffe6b37d1a408d8feed"],
            "actor": ["44ef80dffe04094dff63a76166e4e22b6502af0a", "d48ee016b48ddd7dcc0cce7d51ec87e4f58c0bbd",
                      "631de8a1a95c88b9a7ec22d43d5cdec49427391f"],
            "media/newspaper": ["d1ffda6ea5be79a349b51d5b17731eda7b61c8c5", "59f6cef436ea3ee9dec3c8f9308e4f6f1482bbe5",
                                "75e1cb4b047e76a851ed4e6bb157f813854ce3ed"],
            "politicalparty": ["ec73777e2a7f83761bfd0bad5d052b93adf3f2e3", "489da653ca09bbdf50d4e007efbe541fa7d972dc",
                               "8a0cd1589ddf33b91796dd0f8c93abe926a916fa"],
            "award": ["582990d5ce262303fa297ddb56bd285224137f4b", "79c61007d48b7e920e76dd9e6db8655713736f1d",
                      "14b7d2493c09fe196fac060d8a45ac7c02f7af71"],
            "software": ["b2a8bbf4584637e9d7c6d7c50f6a8ea4f85414fe", "e32739dc9da365b0a52d40caa99c61d07dfde284",
                         "b99cfa25217adb505df0067fe08ee562094f6516"],
            "film": ["513e9cf3194647251c6462f60e496ce52fa93d0d", "741927c3d87856e5ab604b4c8344925945248825",
                     "ca51e32a46c518ebb25e036c673b195cac58d1aa"],
            "currency": ["f39a39d4ce4c2e747bfc4eb58f066395d535917a", "957c8c9e00de342ce52cb863a8fa2a2761e84894",
                         "0a6f9bed511d59df26d6341a1b2e3f581271cce3"],
            "car": ["ac52d604ac818a8573d917207aaba208ac92ec4a", "6efd63c7869b9c567d2561d790d4e6ed8e398175",
                    "41df0e17f6c191c383ffaea6a50116741095b709"],
            "religion": ["27e916084b0c442c2edd48c6b69c3adbfe94b95c", "eb5d6ff91fefed3ff743f5bb22c799dc35e5a4f0",
                         "469179ea739186ef242df4025847b50fed2d8503"],
            "island": ["cc8e7b06ece30555105310b49822e7d819b4a6ec", "042eea0cfd8922cba145e0d8370d21ecaa555c04",
                       "2d94bca92e25ebae0a8dc37faddb26ef9b4e1040"],
            "weapon": ["5d526e9472a7fceb9041e4aa8c6bb6e51d10ba80", "b0c0438d9c387ace145a301f56da368365e368b2",
                       "6014cf11393c54348a85069707593860568113ef"],
            "law": ["a12ef2a822267cc61ef8f5a33bb5f2d61321186a", "7b3e99932c5ad9f9fb38c26a6c4c57403e0b857c",
                    "8e6895433193aca4f6f2cc0cfbb4f2c562b9d6f9"],
            "food": ["437af723fd31f5236684a6b182d8fbd76d52cad4", "9b1cba2b6a7ea54101defb8e48ff1f7b22dd906c",
                     "559723b438a24fa157ba300bf16caaa0629b2352"],
            "park": ["90dbd6b2dc7e714adc34fa666b9af5d1bef80457", "54402454bbe3611268bd622867086274bf2da713",
                     "dd1b64190d365b6c088ae7f9c827ef5086e55a1c"],
            "sportsfacility": ["c7a6dc07841030fa9235940ba35e0063da1e68ad", "35186be1b35f4a430215f2bdc1332ff70bfa0c7f",
                               "cd8d6c8608be24d4e11e4e235655e13df9165d2a"],
            "ship": ["c0c6f37f627bcf7b2749a2a284bf6550c803c9c9", "2cf486e3357a34c2401db5c40637c04385ab528f",
                     "7faacd2f00b0c8ea09d3fe916ce9d843658e76e1"],
            "hotel": ["8b3146b3a1e7e12beea6681f294e8baa6c7db383", "1ba3793c88e8e6ed58279109a8b0b102aa0f2f85",
                      "467504acfd86fc8cfae73ff5e07da0fb47d28f40"],
            "election": ["b8c476aece2e1ca1f9ecf277b7edab05d85fb6ce", "3bb6af6478d4e7da31e0e5c99ba1b081ea5f255b",
                         "9132ff2b0b38f51e88648ae162f0119a6363237c"]}


def load_entity_name_embeddings(layer_name, entity_name_strategy) -> dict[str, torch.Tensor]:
    index = "fewnerd_entity_name_to_embedding"
    elastic_field = f'embedding.{layer_name}.{entity_name_strategy}'
    elastic_query = {
        "query": {
            "match_all": {}
        },
        "size": 100,
        "_source": [
            "entity_name",
            elastic_field,
        ]
    }

    replies = dataset_provider.search(elastic_query, index=index)
    layer_to_tensor = {
        item["_source"]["entity_name"]: torch.Tensor(item["_source"][elastic_field])
        for item in replies['hits']['hits']
    }
    return layer_to_tensor

