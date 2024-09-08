import dataset_provider
import entity_classifier
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm
import clearml_poc
import llama3_interface
from random import shuffle
import numpy as np

from torch.nn.functional import normalize

llama3: llama3_interface.LLama3Interface = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_sentences_and_entities(fine_entity: str):
    sentences_with_entity = dataset_provider.get_by_fine_grained_type_fewnerd(fine_entity)["hits"]["hits"]
    same_type = []

    for sentence in sentences_with_entity:
        sentence_text = sentence["_source"]["full_text"]
        entities_of_fine_type = entity_classifier.pick_entities_of_finetype_fewnerd(sentence["_source"]["tagging"],
                                                                                    fine_entity)
        same_type.append([
            {"text": sentence_text, "phrase": entity["phrase"].strip(), "id": entity["text_id"]} for entity in
            entities_of_fine_type])
    return same_type




def match_entity_with_strongest_word(entity, possible_matches):
    global llama3


    tokens = llama3.tokenize([entity["text"], possible_matches[0]["text"]])
    start1, end1 = llama3.tokens_indices_part_of_sentence(entity["text"], entity["phrase"])
    possible_matches_ranges = [llama3.tokens_indices_part_of_sentence(phrase["text"], phrase["phrase"])
                               for phrase in possible_matches]


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

    is_in_range = torch.zeros_like(most_similar_word)

    for ranges in possible_matches_ranges:
        is_in_range = (most_similar_word >= ranges[0]) & (most_similar_word < ranges[1])

    return is_in_range.float()



def main():
    clearml_poc.clearml_init()

    assert torch.cuda.is_available(), "no gpu available"
    global llama3
    llama3 = llama3_interface.LLama3Interface()

    layers_count = 33
    all_sim1, all_sim2, all_sim3 = torch.zeros(layers_count), torch.zeros(layers_count), torch.zeros(layers_count)
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

        same_type = get_sentences_and_entities(type)
        data = list(combinations(same_type, 2))
        shuffle(data)

        pbar = tqdm(data)

        for index, pairs in enumerate(pbar):
            counter += 1
            x1, x2 = pairs

            word_with_entity = random.choice(x1)
            possible_matches = x2

            sim1 = match_entity_with_strongest_word(word_with_entity, possible_matches).cpu()

            all_sim1 += sim1

            clearml_poc.add_scatter(
                title="matching scores - is the strongest match word in the same part of the entity tag",
                series="same part of entity tag",
                iteration=index,
                values=(all_sim1 / counter).tolist()
            )


    sim1 = (all_sim1 / counter).tolist()
    plt.plot(layers_list, sim1, label=f"similarity between two similar NER value")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
