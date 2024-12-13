import dataset_provider
import entity_classifier
import random
import torch
from itertools import combinations
from tqdm import tqdm
import clearml_poc
import group_layers
import llm_interface
from random import shuffle
import numpy as np
from sklearn import metrics
import torch.nn.functional as F
import pandas as pd

llm: llm_interface.LLMInterface = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_sentences_and_entities(fine_entity: str):
    sentences_with_entity = dataset_provider.get_by_fine_grained_type_fewnerd(fine_entity)["hits"]["hits"]
    same_type, different_type = [], []
    for sentence in sentences_with_entity:
        sentence_text = sentence["_source"]["full_text"]
        entities_of_fine_type = entity_classifier.pick_entities_of_finetype_fewnerd(sentence["_source"]["tagging"],
                                                                                    fine_entity)

        for entity in sentence["_source"]["tagging"]:
            entity_info = {"text": sentence_text,
                           "index_start": entity["index_start"],
                           "index_end": entity["index_end"],
                           "phrase": entity["phrase"].strip(),
                           "id": entity["text_id"]}
            if entity in entities_of_fine_type:
                same_type.append(entity_info)
            else:
                different_type.append(entity_info)

    return same_type, different_type


def compute_word_similarity(entity, match):
    global llm

    tokens = llm.tokenize([entity["text"], match["text"]]).to(device)
    start1, end1 = llm.tokens_indices_part_of_sentence(entity["text"], entity["phrase"])

    h = torch.stack(llm.get_hidden_layers(tokens)).to(device)
    h1, h2 = torch.split(h, [1, 1], dim=1)

    h1 = h1[:, :, end1 - 1, :]
    h1 = h1 / h1.norm(dim=2, keepdim=True)

    # Normalize the tokens
    h2 = h2.squeeze(1)
    h2 = h2 / h2.norm(dim=2, keepdim=True)
    h2 = torch.permute(h2, (0, 2, 1))

    cosine_similarities = torch.matmul(h1, h2).squeeze(1)
    return cosine_similarities


def compute_word_similarity_hooked(entity, match):
    global llm

    tokens = llm.tokenize([entity["text"], match["text"]]).to(device)
    start1, end1 = llm.token_indices_given_text_indices(entity["text"], (entity["index_start"], entity["index_end"]))
    start2, end2 = llm.token_indices_given_text_indices(match["text"], (match["index_start"], match["index_end"]))

    hidden_layers_hook = llm.get_hooked_hidden_layers(tokens)
    cosine_similarities = []
    for index, layer in enumerate(hidden_layers_hook):
        h1, h2 = hidden_layers_hook[layer]
        h1 = h1[end1]
        h2 = h2[end2]
        sim = F.cosine_similarity(h1.flatten(), h2.flatten(), dim=0).item()
        cosine_similarities.append(sim)

    return cosine_similarities


def match_entity_with_strongest_word(entity, match):
    start2, end2 = llm.tokens_indices_part_of_sentence(match["text"], match["phrase"])

    cosine_similarities = compute_word_similarity(entity, match)
    most_similar_word = torch.argmax(cosine_similarities, dim=1)

    is_in_range = (most_similar_word >= start2) & (most_similar_word < end2)

    return is_in_range.float()


def predict_word_match(entity, match):
    start2, end2 = llm.tokens_indices_part_of_sentence(match["text"], match["phrase"])
    cosine_similarities = compute_word_similarity(entity, match)
    return cosine_similarities[:, end2]


predictions, ground_truth = [], []


def prediction_epoch(pairs, different_type, epoch):
    x1, x2 = pairs
    x3 = random.choice(different_type)

    predictions.append(match_entity_with_strongest_word(x1, x2).int().cpu().tolist())
    predictions.append(match_entity_with_strongest_word(x1, x3).int().cpu().tolist())

    ground_truth.append([1 for _ in range(33)])
    ground_truth.append([0 for _ in range(33)])

    x = np.asarray(predictions)
    y = np.asarray(ground_truth)

    pearson = [np.corrcoef(x.transpose()[layer], y.transpose()[layer])[0][1] for layer in range(33)]

    accuracy = np.sum(x == y, axis=0) / (2 * epoch)

    clearml_poc.add_scatter(
        title="matching scores - is the strongest match word in the same part of the entity tag",
        series="same part of entity tag",
        iteration=epoch,
        values=accuracy
    )

    clearml_poc.add_scatter(
        title="pearson correlation",
        series="predictions and ground truth",
        iteration=epoch,
        values=pearson
    )


find_threshold_prediction, find_threshold_ground_truth = [], []


def find_optimal_threshold(predictions, real_values):
    fpr, tpr, thresholds = metrics.roc_curve(y_true=real_values, y_score=predictions)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


keys, item_to_group, group_to_item, group_to_indices, group_lcp_meanings = None, None, None, None, None


def find_longest_common_prefix(words):
    # Initialize the longest common prefix to the first word
    longest_common_prefix = words[0]

    # Loop through each word in the list of words
    for word in words[1:]:
        # Loop through each character in the current longest common prefix
        for i in range(len(longest_common_prefix)):
            # If the current character is not the same as the character in the same position in the current word
            if i >= len(word) or longest_common_prefix[i] != word[i]:
                # Update the longest common prefix and break out of the loop
                longest_common_prefix = longest_common_prefix[:i]
                break

    # Return the longest common prefix
    return longest_common_prefix


def find_threshold_hooked_epoch(pairs, different_type, epoch):
    x1, x2 = pairs
    x3 = random.choice(different_type)

    predictions.append(compute_word_similarity_hooked(x1, x2))
    predictions.append(compute_word_similarity_hooked(x1, x3))

    global keys, item_to_group, group_to_item, group_to_indices, group_lcp_meanings
    if not keys:
        keys = list(llm.extractable_parts.keys())
        item_to_group = group_layers.map_item_to_group(list(keys))
        group_to_item = group_layers.group_layers(item_to_group)
        group_to_indices = {group_id: [keys.index(item) for item in group if item in keys] for group_id, group in
                            group_to_item.items()}
        group_lcp_meanings = {group_id: find_longest_common_prefix([keys[item] for item in group_to_indices[group_id]])
                              for group_id in group_to_item.keys()}

    ground_truth.append([1 for _ in range(len(keys))])
    ground_truth.append([0 for _ in range(len(keys))])

    if epoch % 1000 != 0:
        return

    x = np.asarray(predictions).transpose()
    y = np.asarray(ground_truth).transpose()

    optimal_threshold = np.asarray(
        [find_optimal_threshold(x[key].flatten(), y[key].flatten()) for key in range(len(keys))])
    optimal_threshold_seperator = optimal_threshold[:, np.newaxis]
    accuracy = np.sum(((x >= optimal_threshold_seperator) == y), axis=1) / x.shape[1]

    if epoch == 0:
        df = pd.DataFrame(
            data={"keys": keys,
                  "shape": [str(llm.extractable_parts[key].shape) for key in keys]
                  },
            index=list(range(len(keys)))
        )
        clearml_poc.add_table(
            title="guideline for model's output",
            series="llm model",
            iteration=0,
            table=df
        )

        item_to_group_df = pd.DataFrame(
            data={"item": list(item_to_group.keys()),
                  "group": list(item_to_group.values())
                  },
            index=list(range(len(item_to_group)))
        )
        clearml_poc.add_table(
            title="item to group mapping",
            series="llama model",
            iteration=0,
            table=item_to_group_df
        )

    threshold_df = pd.DataFrame(
        data={"keys": keys,
              "threshold": optimal_threshold
              },
        index=list(range(len(keys)))
    )
    clearml_poc.add_table(
        title="optimal threshold for each layer",
        series="llm model",
        iteration=epoch,
        table=threshold_df
    )
    clearml_poc.add_scatter(
        title="using cosine similarities to separate between same and different entities",
        series="same part of entity tag",
        iteration=x.shape[1] / 2,
        values=accuracy
    )
    for group_id, group in group_to_item.items():
        group_indices = group_to_indices[group_id]

        group_accuracy = np.sum(((x[group_indices] >= optimal_threshold_seperator[group_indices]) == y[group_indices]),
                                axis=1) / x.shape[1]
        clearml_poc.add_scatter(
            title="cosine similarity threshold with groups in mind",
            series=f'group {group_lcp_meanings[group_id]}',
            iteration=x.shape[1] / 2,
            values=group_accuracy)


def find_threshold_epoch(pairs, different_type, epoch):
    x1, x2 = pairs
    x3 = random.choice(different_type)

    predictions.append(predict_word_match(x1, x2).cpu().tolist())
    predictions.append(predict_word_match(x1, x3).cpu().tolist())

    ground_truth.append([1 for _ in range(33)])
    ground_truth.append([0 for _ in range(33)])

    x = np.asarray(predictions)
    y = np.asarray(ground_truth)

    optimal_threshold = find_optimal_threshold(x.flatten(), y.flatten())
    accuracy = np.sum(x > optimal_threshold, axis=0) / x.shape[0]

    clearml_poc.add_scatter(
        title="using cosine similarities to separate between same and different entities",
        series="same part of entity tag",
        iteration=x.shape[0] / 2,
        values=accuracy
    )


def main():
    clearml_poc.clearml_init()

    assert torch.cuda.is_available(), "no gpu available"
    global llm
    LLM_ID = "meta-llama/Llama-3.3-70B-Instruct"
    llm = llm_interface.LLMInterface(LLM_ID)

    types = ['island', 'athlete', 'politicalparty', 'actor', 'software', 'sportsfacility', 'weapon', 'food', 'election',
             'car', 'currency', 'park', 'award', 'GPE', 'media/newspaper', 'law', 'religion', 'film', 'hotel', 'ship']

    counter = 0

    action = "find_threshold_hooked"

    for type in types:

        same_type, different_type = get_sentences_and_entities(type)

        data = list(combinations(same_type, 2))
        shuffle(data)

        pbar = tqdm(data)

        for index, pairs in enumerate(pbar):
            counter += 1

            if action == "prediction":
                prediction_epoch(pairs, different_type, index)
            elif action == "find_threshold":
                find_threshold_epoch(pairs, different_type, index)
            elif action == "find_threshold_hooked":
                find_threshold_hooked_epoch(pairs, different_type, index)


if __name__ == "__main__":
    main()
