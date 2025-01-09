from collections import Counter
import dataset_provider
from tqdm import tqdm
import clearml_poc
import pandas as pd
import urllib3

urllib3.disable_warnings()


all_fine_types = ['GPE', 'company', 'artist/author', 'athlete', 'politician', 'sportsteam', 'education',
                  'biologything', 'road/railway/highway/transit', 'sportsevent', 'actor', 'government/governmentagency',
                  'bodiesofwater', 'media/newspaper', 'chemicalthing', 'attack/battle/war/militaryconflict',
                  'politicalparty', 'music', 'award', 'writtenart', 'livingthing', 'sportsleague', 'software',
                  'disease', 'language', 'film', 'airplane', 'currency', 'astronomything', 'car', 'showorganization',
                  'religion', 'scholar', 'mountain', 'soldier', 'island', 'broadcastprogram', 'weapon', 'director',
                  'god', 'law', 'theater', 'food', 'medical', 'game', 'park', 'sportsfacility', 'ship',
                  'educationaldegree', 'airport', 'hospital', 'library', 'train', 'hotel', 'restaurant', 'disaster',
                  'election', 'protest', 'painting']

all_coarse_types = [
    'location', 'person', 'organization', 'product', 'event', 'building', 'art', 'other'
]

fine_to_coarse = {'GPE': 'location', 'company': 'organization', 'artist/author': 'person', 'athlete': 'person',
                  'politician': 'person', 'sportsteam': 'organization', 'education': 'organization',
                  'biologything': 'other', 'road/railway/highway/transit': 'location', 'sportsevent': 'event',
                  'actor': 'person', 'government/governmentagency': 'organization', 'bodiesofwater': 'location',
                  'media/newspaper': 'organization', 'chemicalthing': 'other',
                  'attack/battle/war/militaryconflict': 'event', 'politicalparty': 'organization', 'music': 'art',
                  'award': 'other', 'writtenart': 'art', 'livingthing': 'other', 'sportsleague': 'organization',
                  'software': 'product', 'disease': 'other', 'language': 'other', 'film': 'art', 'airplane': 'product',
                  'currency': 'other', 'astronomything': 'other', 'car': 'product', 'showorganization': 'organization',
                  'religion': 'organization', 'scholar': 'person', 'mountain': 'location', 'soldier': 'person',
                  'island': 'location', 'broadcastprogram': 'art', 'weapon': 'product', 'director': 'person',
                  'god': 'other', 'law': 'other', 'theater': 'building', 'food': 'product', 'medical': 'other',
                  'game': 'product', 'park': 'location', 'sportsfacility': 'building', 'ship': 'product',
                  'educationaldegree': 'other', 'airport': 'building', 'hospital': 'building', 'library': 'building',
                  'train': 'product', 'hotel': 'building', 'restaurant': 'building', 'disaster': 'event',
                  'election': 'event', 'protest': 'event', 'painting': 'art'}

model_threshold = 0.293
elasticsearch_threshold = (1 + model_threshold) / 2


def main():
    top_n = 5
    data = {}
    keys = [f'{k}st' for k in range(top_n)] + ['recall_fine_type', 'recall_coarse_type', 'fine_type']
    for key in keys:
        data[key] = []
    for type in tqdm(all_fine_types):
        retrieve_50_query = {
            "size": 50,
            "query": {
                "term": {
                    "fine_type": type
                }
            },
            "sort": [
                {
                    "_script": {
                        "type": "number",
                        "script": {
                            "source": "Math.random()"
                        },
                        "order": "asc"
                    }
                }
            ],
            "_source": [
                "fine_type",
                "coarse_type",
                "doc_id",
                "all_text",
                "phrase",
                "embedding.llama_3_17_v_proj.start",
                "embedding.llama_3_17_v_proj.end"
            ]
        }
        response = dataset_provider.search(index="fewnerd_v4_*", query=retrieve_50_query)
        fine_type_counter = Counter()
        coarse_type_counter = Counter()
        for item in response["hits"]["hits"]:
            doc = item["_source"]
            fine_type = doc["fine_type"]
            coarse_type = doc["coarse_type"]
            ne_embedding = doc["embedding"]["llama_3_17_v_proj"]["ne_embedding"]
            doc_id = doc["doc_id"]
            query_similarity_by_embedding = {
                "knn": {
                    "field": "embedding.llama_3_17_v_proj.ne_embedding",
                    "k": 50,
                    "num_candidates": 10000,
                    "query_vector": ne_embedding
                },
                "min_score": elasticsearch_threshold,
                "size": 50,
                "_source": ["fine_type", "coarse_type", "doc_id"]
            }
            possible_responses = dataset_provider.search(index="fewnerd_v4_*", query=query_similarity_by_embedding)
            fine_type_counter.update([f'{item["_source"]["fine_type"]}@{item["_source"]["coarse_type"]}' for item in possible_responses["hits"]["hits"]])
            coarse_type_counter.update([item["_source"]["coarse_type"] for item in possible_responses["hits"]["hits"]])

        named_fine_type = f"{type}@{fine_to_coarse[type]}"
        data['fine_type'].append(named_fine_type)
        for i,(k, v) in zip(range(top_n), fine_type_counter.most_common(top_n)):
            data[f'{i}st'].append(f'{k} ({fine_type_counter[k]})')

        true_positive = fine_type_counter[named_fine_type]
        false_negative = fine_type_counter.total() - true_positive
        recall_for_fine_type = true_positive / (true_positive + false_negative)

        # recall_for_type = recall_score(y_true=[fine_type_counter.total()], y_pred=[fine_type_counter[type]])
        data['recall_fine_type'].append(recall_for_fine_type)

        true_positive_coarse_type = coarse_type_counter[fine_to_coarse[type]]
        false_negative_coarse_type = coarse_type_counter.total() - true_positive_coarse_type
        recall_for_coarse_type = true_positive_coarse_type / (true_positive_coarse_type + false_negative_coarse_type)
        data['recall_coarse_type'].append(recall_for_coarse_type)

        # data[f'{type}@{fine_to_coarse[type]} ({fine_type_counter.total()})'] = [f'' for k, v in fine_type_counter.most_common(top_n)] + [recall_for_type]

    df = pd.DataFrame(data)
    clearml_poc.add_table(title="Top 5 most common fine types retrieved by the model",
                          series="llama 3.1 17 v proj",
                          iteration=0,
                          table=df)





    print(data)

if __name__ == "__main__":
    clearml_poc.clearml_init()

    main()
