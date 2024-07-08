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

llama3: llama3_interface.LLama3Interface = None


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


def similarities_in_layer(x1, x2, x3, h, layer):
    global llama3
    h1, h2, h3 = h[layer]
    start1, end1 = llama3.tokens_indices_part_of_sentence(x1["text"], x1["phrase"])
    start2, end2 = llama3.tokens_indices_part_of_sentence(x2["text"], x2["phrase"])
    start3, end3 = llama3.tokens_indices_part_of_sentence(x3["text"], x3["phrase"])
    h1 = torch.mean(h1[start1:end1, :], dim=0)
    h2 = torch.mean(h2[start2:end2, :], dim=0)
    h3 = torch.mean(h3[start3:end3, :], dim=0)

    sim1 = F.cosine_similarity(h1, h2, dim=-1).cpu().item()
    sim2 = F.cosine_similarity(h1, h3, dim=-1).cpu().item()
    sim3 = F.cosine_similarity(h2, h3, dim=-1).cpu().item()

    return sim1, sim2, sim3


def main():
    clearml_poc.clearml_init()

    assert torch.cuda.is_available(), "no gpu available"
    global llama3
    llama3 = llama3_interface.LLama3Interface()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_sim1, all_sim2, all_sim3 = [], [], []
    layers_list = list(range(33))

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
            desc = "processing pairs " + str([phrase["id"] for phrase in pairs])
            pbar.set_description(desc)
            x1, x2 = pairs
            x3 = random.choice(different_type)
            tokens = llama3.tokenize([x1["text"], x2["text"], x3["text"]]).to(device)
            h = llama3.get_hidden_layers(tokens)
            cosine_similarities = [similarities_in_layer(x1, x2, x3, h, layer) for layer in layers_list]
            sim1, sim2, sim3 = zip(*cosine_similarities)

            all_sim1.append(list(sim1))
            all_sim2.append(list(sim2))
            all_sim3.append(list(sim3))

            clearml_poc.add_scatter(
                title="similarity scores",
                series="same part of entity tag",
                iteration=index,
                values=torch.mean(torch.as_tensor(all_sim1), dim=0).tolist()
            )

            clearml_poc.add_scatter(
                title="similarity scores",
                series="different part of entity tag",
                iteration=index,
                values=torch.mean(torch.as_tensor(all_sim2), dim=0).tolist()
            )

            clearml_poc.add_scatter(
                title="similarity scores",
                series="another different part of entity tag",
                iteration=index,
                values=torch.mean(torch.as_tensor(all_sim3), dim=0).tolist()
            )

    sim1 = torch.mean(torch.as_tensor(all_sim1), dim=0).tolist()
    sim2 = torch.mean(torch.as_tensor(all_sim2), dim=0).tolist()
    sim3 = torch.mean(torch.as_tensor(all_sim3), dim=0).tolist()
    plt.plot(layers_list, sim1, label=f"similarity between two similar NER value")
    plt.plot(layers_list, sim2, label=f"similarity between different NER value ")
    plt.plot(layers_list, sim3, label=f"similarity between different NER value")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
