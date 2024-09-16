import random
from random import choice
import clearml_poc
from contrastive.mlp import ContrastiveMLP, Detector
from contrastive.loss import ContrastiveLoss
import torch
import llm_interface
from contrastive import fewnerd_processor
import torch.nn.functional as F


def forward_at_llm(batch, llm, llm_hidden_layer) -> torch.Tensor:
    tokens = llm.tokenize([x["text"] for x in batch])
    indices = [llm.token_indices_given_text_indices(x["text"], (x["index_start"], x["index_end"])) for x in batch]
    h = llm.get_llm_at_layer(tokens, llm_hidden_layer)
    h = [h[end] for h, (start, end) in zip(h, indices)]
    h = torch.stack(h).cuda().float()
    torch.cuda.empty_cache()
    return h


def main():
    clearml_poc.clearml_init()

    assert torch.cuda.is_available(), "no gpu available"
    llm_id = "meta-llama/Meta-Llama-3.1-8B"
    llm_hidden_layer = "model.layers.17.self_attn.v_proj"
    llm = llm_interface.LLMInterface(llm_id, interested_layers=[llm_hidden_layer])

    sizes = [1024, 4096, 256]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    similarity_model = ContrastiveMLP(*sizes).to(device)
    classifier_model = Detector(sizes[-1]).to(device)
    optimizer = torch.optim.Adam(list(similarity_model.parameters()) + list(classifier_model.parameters()), lr=1e-4)
    similarity_criterion = ContrastiveLoss()
    classifier_criterion = torch.nn.CrossEntropyLoss()

    for index, (same_type, different_type) in enumerate(fewnerd_processor.yield_dataset(llm, llm_hidden_layer)):
        torch.cuda.empty_cache()



        training = index % 2 == 0
        if training:
            similarity_model.train()
            classifier_model.train()

            same_type = same_type[:len(same_type) - len(same_type) % 2]
            same_type = forward_at_llm(same_type, llm, llm_hidden_layer).to(device)
            different_type = forward_at_llm(different_type, llm, llm_hidden_layer).to(device)
            good_batch_forward = similarity_model(same_type)
            bad_batch_forward = similarity_model(different_type)[:len(good_batch_forward)]
            first_half, second_half = good_batch_forward.chunk(2, dim=0)

            optimizer.zero_grad()

            good_similarity, bad_similarity, similarity_loss = compute_similarity(first_half,
                       second_half,
                       good_batch_forward,
                       bad_batch_forward,
                       device,
                       similarity_model,
                       similarity_criterion).values()
            classifier_accuracy, classifier_loss = compute_accuracy(first_half, second_half, bad_batch_forward, device,
                                                                    classifier_model, classifier_criterion).values()
            loss = similarity_loss + classifier_loss


            loss.backward()
            optimizer.step()
            clearml_poc.add_point_to_graph(title="loss", series="train", x=index, y=loss.item())
            clearml_poc.add_point_to_graph(title="good_similarity", series="train", x=index, y=good_similarity.item())
            clearml_poc.add_point_to_graph(title="bad_similarity", series="train", x=index, y=bad_similarity.item())
            clearml_poc.add_point_to_graph(title="accuracy", series="train", x=index, y=classifier_accuracy)
        else:
            similarity_model.eval()
            classifier_model.eval()

            same_type = same_type[:len(same_type) - len(same_type) % 2]
            same_type = forward_at_llm(same_type, llm, llm_hidden_layer).to(device)
            different_type = forward_at_llm(different_type, llm, llm_hidden_layer).to(device)
            good_batch_forward = similarity_model(same_type)
            bad_batch_forward = similarity_model(different_type)[:len(good_batch_forward)]
            first_half, second_half = good_batch_forward.chunk(2, dim=0)

            good_similarity, bad_similarity, similarity_loss = compute_similarity(first_half,
                                                                                  second_half,
                                                                                  good_batch_forward,
                                                                                  bad_batch_forward,
                                                                                  device,
                                                                                  similarity_model,
                                                                                  similarity_criterion).values()
            classifier_accuracy, classifier_loss = compute_accuracy(first_half, second_half, bad_batch_forward, device,
                                                                    classifier_model, classifier_criterion).values()
            loss = similarity_loss + classifier_loss

            clearml_poc.add_point_to_graph(title="loss", series="eval", x=index, y=loss.item())
            clearml_poc.add_point_to_graph(title="good_similarity", series="eval", x=index, y=good_similarity.item())
            clearml_poc.add_point_to_graph(title="bad_similarity", series="eval", x=index, y=bad_similarity.item())
            clearml_poc.add_point_to_graph(title="accuracy", series="eval", x=index, y=classifier_accuracy)


def compute_accuracy(good_first_half,
                     good_second_half,
                     bad_batch_forward,
                     device,
                     model,
                     criterion):
    similarity_from = torch.concat((good_first_half, good_first_half), dim=0).to(device)
    simialrity_to = torch.concat((good_second_half, bad_batch_forward[:len(good_first_half)]), dim=0).to(device)
    labels = torch.tensor([1] * len(good_first_half) + [0] * len(good_first_half)).to(device)
    classifcation_prediction = model(similarity_from, simialrity_to)
    classifier_loss = criterion(classifcation_prediction, labels)
    accuracy = torch.sum(torch.argmax(classifcation_prediction, dim=-1) == labels).item() / \
               classifcation_prediction.shape[0]

    return {
        "accuracy": accuracy,
        "loss": classifier_loss
    }


def compute_similarity(good_first_half,
                       good_second_half,
                       good_batch_forward,
                       bad_batch_forward,
                       device,
                       model,
                       criterion):
    good_similarity = F.cosine_similarity(good_first_half, good_second_half, dim=-1).mean()
    bad_similarity = F.cosine_similarity(good_batch_forward, bad_batch_forward[:len(good_batch_forward)], dim=-1).mean()
    loss = criterion(good_first_half, good_second_half, bad_batch_forward)

    return {
        "good_similarity": good_similarity,
        "bad_similarity": bad_similarity,
        "loss": loss
    }


if __name__ == "__main__":
    main()

# For today:
# 1. Pytorch dataset
# 2. Basic implementation of the model
# 3. Usage of MLP
# 4. Training the model
# 5. Args file

# For tomorrow:
# entrying last token and token before word
# usgae of all examples of contrastive loss - pair wise, triplet wise, etc
# using LoRA to train model instead of MLP
# optuna for hyperparameter tuning
