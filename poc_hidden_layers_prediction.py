import dataset_provider
import entity_classifier
import random
import torch
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm
import clearml_poc
import llama3_interface
from random import shuffle
import numpy as np

llama3: llama3_interface.LLama3Interface = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_sentences_and_entities(fine_entity: str):
    sentences_with_entity = dataset_provider.get_by_fine_grained_type_fewnerd(fine_entity)["hits"]["hits"]
    same_type, different_type = [], []
    for sentence in sentences_with_entity:
        sentence_text = sentence["_source"]["full_text"]
        entities_of_fine_type = entity_classifier.pick_entities_of_finetype_fewnerd(sentence["_source"]["tagging"],
                                                                                    fine_entity)
        entities_of_different_type = list(
            filter(lambda entity: entity not in entities_of_fine_type, sentence["_source"]["tagging"]))
        same_type.extend(
            {"text": sentence_text, "phrase": entity["phrase"].strip(), "id": entity["text_id"]} for entity in
            entities_of_fine_type)
        different_type.extend(
            {"text": sentence_text, "phrase": entity["phrase"].strip(), "id": entity["text_id"]} for entity in
            entities_of_different_type)

    clearml_poc.add_text(f"entites: \n{sentences_with_entity}")

    return same_type, different_type


def match_entity_with_strongest_word(entity, match):
    global llama3

    tokens = llama3.tokenize([entity["text"], match["text"]])
    start1, end1 = llama3.tokens_indices_part_of_sentence(entity["text"], entity["phrase"])
    start2, end2 = llama3.tokens_indices_part_of_sentence(match["text"], match["phrase"])

    h = torch.stack(llama3.get_hidden_layers(tokens)).to(device)
    h1, h2 = torch.split(h, [1, 1], dim=1)

    h1 = h1[:, :, end1 - 1, :]
    h1 = h1 / h1.norm(dim=2, keepdim=True)

    # Normalize the tokens
    h2 = h2.squeeze(1)
    h2 = h2 / h2.norm(dim=2, keepdim=True)
    h2 = torch.permute(h2, (0, 2, 1))

    # Compute the cosine similarity
    cosine_similarities = torch.matmul(h1, h2).squeeze(1)

    most_similar_word = torch.argmax(cosine_similarities, dim=1)

    is_in_range = (most_similar_word >= start2) & (most_similar_word < end2)


    return is_in_range.float()


def main():
    clearml_poc.clearml_init()

    assert torch.cuda.is_available(), "no gpu available"
    global llama3
    llama3 = llama3_interface.LLama3Interface()

    layers_count = 33
    predictions, ground_truth = [], []
    layers_list = list(range(layers_count))



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

    counter = 0

    for type in types:

        same_type, different_type = get_sentences_and_entities(type)

        data = list(combinations(same_type, 2))
        shuffle(data)

        pbar = tqdm(data)

        for index, pairs in enumerate(pbar):
            counter += 1

            x1, x2 = pairs
            x3 = random.choice(different_type)

            predictions.append(match_entity_with_strongest_word(x1, x2).int().cpu().tolist())
            predictions.append(match_entity_with_strongest_word(x1, x3).int().cpu().tolist())

            ground_truth.append([1 for _ in range(33)])
            ground_truth.append([0 for _ in range(33)])

            x = np.asarray(predictions)
            y = np.asarray(ground_truth)

            pearson = [np.corrcoef(x.transpose()[layer], y.transpose()[layer])[0][1] for layer in range(33)]

            accuracy = np.sum(x == y, axis=0) / (2 * counter)

            clearml_poc.add_scatter(
                title="matching scores - is the strongest match word in the same part of the entity tag",
                series="same part of entity tag",
                iteration=index,
                values=accuracy
            )

            clearml_poc.add_scatter(
                title="pearson correlation",
                series="predictions and ground truth",
                iteration=index,
                values=pearson
            )

    #
    # sim1 = (all_sim1 / counter).tolist()
    # plt.plot(layers_list, sim1, label=f"similarity between two similar NER value")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()
