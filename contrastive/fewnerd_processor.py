import dataset_provider


def yield_dataset(types, batch_size=50, instances_per_type=100, llm_layer=None):
    extract = lambda x: extract_entities_from_es_response(x)

    for result in dataset_provider.random_results_per_fine_type(types, instances_per_type):
        result_type = result["_source"]["fine_type"]
        coarse_type = result["_source"]["coarse_type"]
        text = result["_source"]["all_text"]
        good_batch = dataset_provider.get_by_fine_grained_type_fewnerd_v4(result_type, randomize=True,
                                                                          batch_size=batch_size)
        other_types = [t for t in types if t != result_type]
        bad_batch = dataset_provider.get_hard_negative_fewnerd(fine_types=other_types,
                                                               coarse_type=coarse_type,
                                                               anchor_text=text,
                                                               batch_size=batch_size,
                                                               llm_layer=llm_layer)

        yield result["_source"], extract(good_batch), extract(bad_batch)


def yield_train_dataset(batch_size=50, instances_per_type=100,llm_layer=None):
    types = ['education', 'airport', 'restaurant', 'sportsleague', 'disease', 'hospital', 'painting', 'other',
             'library', 'sportsevent', 'soldier', 'game', 'educationaldegree', 'broadcastprogram', 'mountain',
             'road/railway/highway/transit', 'company', 'politician', 'attack/battle/war/militaryconflict',
             'astronomything', 'language', 'train', 'scholar', 'bodiesofwater', 'chemicalthing', 'director',
             'showorganization', 'writtenart', 'disaster', 'medical', 'music', 'airplane', 'biologything', 'theater',
             'sportsteam', 'government/governmentagency', 'livingthing', 'artist/author', 'protest', 'god']

    return yield_dataset(types, batch_size, instances_per_type, llm_layer)


def yield_test_dataset(batch_size=50, instances_per_type=100, llm_layer=None):
    types = ['island', 'athlete', 'politicalparty', 'actor', 'software', 'sportsfacility', 'weapon', 'food', 'election',
             'car', 'currency', 'park', 'award', 'GPE', 'media/newspaper', 'law', 'religion', 'film', 'hotel', 'ship']
    return yield_dataset(types, batch_size, instances_per_type, llm_layer)


def extract_entities_from_es_response(response):
    return [docu["_source"] for docu in response]
