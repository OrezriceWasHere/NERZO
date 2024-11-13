from random import shuffle
from tqdm import tqdm
import dataset_provider


def yield_dataset(types, batch_size=50, instances_per_type=100):
    extract = lambda x: extract_entities_from_es_response(x)
    anchors, good_bathches, bad_batches = [], [], []

    for result in dataset_provider.random_results_per_fine_type(types, instances_per_type):
        result_type = result["_source"]["fine_type"]
        good_batch = dataset_provider.get_by_fine_grained_type_fewnerd_v4(result_type,  randomize=True, batch_size=batch_size)
        other_types = [t for t in types if t != result_type]
        bad_batch = dataset_provider.get_by_fine_grained_type_fewnerd_v4(other_types, randomize=True, batch_size=batch_size)
        anchors.append(result["_source"])
        good_bathches.append(extract(good_batch))
        bad_batches.append(extract(bad_batch))

    order_of_results = list(range(len(anchors)))
    shuffle(order_of_results)
    for i in tqdm(order_of_results):
        yield anchors[i], good_bathches[i], bad_batches[i]

    del anchors, good_bathches, bad_batches

def yield_train_dataset(batch_size=50, instances_per_type=100):
    types = ['education', 'airport', 'restaurant', 'sportsleague', 'disease', 'hospital', 'painting', 'other',
             'library', 'sportsevent', 'soldier', 'game', 'educationaldegree', 'broadcastprogram', 'mountain',
             'road/railway/highway/transit', 'company', 'politician', 'attack/battle/war/militaryconflict',
             'astronomything', 'language', 'train', 'scholar', 'bodiesofwater', 'chemicalthing', 'director',
             'showorganization', 'writtenart', 'disaster', 'medical', 'music', 'airplane', 'biologything', 'theater',
             'sportsteam', 'government/governmentagency', 'livingthing', 'artist/author', 'protest', 'god']

    return yield_dataset(types, batch_size, instances_per_type)


def yield_test_dataset(batch_size=50, instances_per_type=100):
    types = ['island', 'athlete', 'politicalparty', 'actor', 'software', 'sportsfacility', 'weapon', 'food', 'election',
             'car', 'currency', 'park', 'award', 'GPE', 'media/newspaper', 'law', 'religion', 'film', 'hotel', 'ship']
    return yield_dataset(types, batch_size, instances_per_type)


def extract_entities_from_es_response(response):
    return [docu["_source"] for docu in response]
