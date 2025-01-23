import asyncio
from collections import defaultdict

import pandas as pd
from sklearn.metrics import recall_score
from elasticsearch import AsyncElasticsearch

import clearml_poc
import queries
import runtime_args


def main():
    all_test_types = anchors()
    tasks = [handle_type(type, ids) for type, ids in all_test_types.items()]
    loop = asyncio.get_event_loop()
    result = defaultdict(list)
    task_results = loop.run_until_complete(asyncio.gather(*tasks))
    for fine_type, task_result in zip(all_test_types.keys(), task_results):
        for result_key, result_value in task_result.items():
            result[result_key].append(result_value)
        result["index"].append(fine_type)

    index_column = result.pop("index")
    df = pd.DataFrame(data=result, index=index_column)
    clearml_poc.add_table(
        iteration=0,
        series="eval",
        title="recall per fine type",
        table=df
    )

    clearml_poc.add_table(
        iteration=0,
        series="eval",
        title="average recall",
        table=df.mean().to_frame()
    )


async def handle_type(fine_type, ids):
    count_type = await get_count_fine_type(fine_type)
    result = {}
    for k in (10, 50, count_type):
        recall_list = []
        for fewnerd_id in ids:
            recall = await handle_fewnerd_id_k(fine_type, fewnerd_id, k)
            recall_list.append(recall)
        size_desc = k if k in [10, 50] else "size"
        result[f"recall@{size_desc}"] = sum(recall_list) / len(recall_list)
    return result


async def get_item_by_id(fewnerd_id, **kwargs):
    results =  await es.get(index=index, id=fewnerd_id, **kwargs)
    return results["_source"]

async def handle_fewnerd_id_k(fine_type, fewnerd_id, k):
    document = await get_item_by_id(fewnerd_id,  source=[f"embedding.{layer}"])
    embedding = document["embedding"][layer]
    similar_items = await search_similar_items(embedding, k)
    returned_fine_types = [1 if item["_source"]["fine_type"] == fine_type else 0  for item in similar_items]
    return recall_score(y_true=[1] * k, y_pred=returned_fine_types)

async def search_similar_items(embedding, k):
    results = []
    remaining_items = k + 1
    query = queries.query_search_by_similarity(embedding, f'embedding.{layer}')
    query = {
        "query":query,
        "_source": [
            "fine_type"
        ],
        "sort": [
            "_score", "fine_type"
        ],
        "size": min(remaining_items, 10000)
    }

    response = await es.search(index=index, body=query)
    hits = response.get("hits", {}).get("hits", [])
    search_after = hits[-1]["sort"] if hits else None

    while remaining_items > 0:
        results.extend(hits)
        remaining_items -= len(hits)
        query["search_after"] = search_after
        response = await es.search(index=index, body=query)
        hits = response.get("hits", {}).get("hits", [])
        search_after = hits[-1]["sort"] if hits else None

    return results[1:k + 1]


async def get_count_fine_type(fine_type):
    query = {
        "bool": {
            "must": [
                {"term": {"fine_type": fine_type}},
                {"exists": {"field": f'embedding.{layer}'}}
            ]
        }
    }
    query_results = await es.count(index=index, query=query)
    return query_results["count"]


def anchors():
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


if __name__ == "__main__":
    clearml_poc.clearml_init(task_name="calculate recall")

    layer_obj = {"layer_id": "llama_3_3_13_k_proj"}
    clearml_poc.clearml_connect_hyperparams(layer_obj, name="layer_name")
    layer = layer_obj["layer_id"]
    index = "fewnerd_tests"
    es = AsyncElasticsearch(**runtime_args.ElasticsearchConnection().model_dump())

    main()
