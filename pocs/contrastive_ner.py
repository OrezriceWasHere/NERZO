import clearml_poc
from contrastive.args import Arguments
from contrastive.mlp import ContrastiveMLP, Detector
from contrastive.loss import ContrastiveLoss
import torch
import llm_interface
from contrastive import fewnerd_processor
import torch.nn.functional as F
from tqdm import tqdm

def main():
    for index, (same_type, different_type) in enumerate(tqdm(fewnerd_processor.yield_dataset())):
        torch.cuda.empty_cache()

        same_type = same_type[:len(same_type) - len(same_type) % 2]
        same_type = forward_at_llm(same_type).to(device)
        different_type = forward_at_llm(different_type).to(device)

        if index % 2 == 0:
            train(index, same_type, different_type)
        else:
            evaluate(index, same_type, different_type)


def train(index, same_type, different_type):
    similarity_model.train()
    classifier_model.train()

    good_batch_forward, bad_batch_forward, first_half, second_half = generate_pairs(same_type, different_type)

    optimizer.zero_grad()

    good_similarity, bad_similarity, similarity_loss = compute_similarity(first_half,
                                                                          second_half,
                                                                          good_batch_forward,
                                                                          bad_batch_forward).values()
    classifier_accuracy, classifier_loss = compute_accuracy(first_half, second_half, bad_batch_forward).values()
    loss = similarity_loss + classifier_loss

    loss.backward()
    optimizer.step()
    log_training_metrics(index, loss.item(), good_similarity.item(), bad_similarity.item(), classifier_accuracy,
                         series="train")


def evaluate(index, same_type, different_type):
    similarity_model.eval()
    classifier_model.eval()

    good_batch_forward, bad_batch_forward, first_half, second_half = generate_pairs(same_type, different_type)

    good_similarity, bad_similarity, similarity_loss = compute_similarity(first_half,
                                                                          second_half,
                                                                          good_batch_forward,
                                                                          bad_batch_forward).values()
    classifier_accuracy, classifier_loss = compute_accuracy(first_half, second_half, bad_batch_forward).values()
    loss = similarity_loss + classifier_loss

    log_training_metrics(index, loss.item(), good_similarity.item(), bad_similarity.item(), classifier_accuracy,
                         series="eval")


def forward_at_llm(batch) -> torch.Tensor:
    tokens = llm.tokenize([x["text"] for x in batch])
    indices = [llm.token_indices_given_text_indices(x["text"], (x["index_start"], x["index_end"])) for x in batch]
    h = llm.get_llm_at_layer(tokens, llm_hidden_layer)
    h = [h[end] for h, (start, end) in zip(h, indices)]
    h = torch.stack(h).cuda().float()
    torch.cuda.empty_cache()
    return h


def generate_pairs(same_type, different_type):
    good_batch_forward = similarity_model(same_type)
    bad_batch_forward = similarity_model(different_type)[:len(good_batch_forward)]
    first_half, second_half = good_batch_forward.chunk(2, dim=0)
    return good_batch_forward, bad_batch_forward, first_half, second_half


def compute_accuracy(good_first_half,
                     good_second_half,
                     bad_batch_forward):
    similarity_from = torch.concat((good_first_half, good_first_half), dim=0).to(device)
    similarity_to = torch.concat((good_second_half, bad_batch_forward[:len(good_first_half)]), dim=0).to(device)
    labels = torch.tensor([1] * len(good_first_half) + [0] * len(good_first_half)).to(device)
    classification_prediction = classifier_model(similarity_from, similarity_to)
    classifier_loss = classifier_criterion(classification_prediction, labels)
    accuracy = torch.sum(torch.argmax(classification_prediction, dim=-1) == labels).item() / \
               classification_prediction.shape[0]

    return {
        "accuracy": accuracy,
        "loss": classifier_loss
    }


def compute_similarity(good_first_half,
                       good_second_half,
                       good_batch_forward,
                       bad_batch_forward):
    good_similarity = F.cosine_similarity(good_first_half, good_second_half, dim=-1).mean()
    bad_similarity = F.cosine_similarity(good_batch_forward, bad_batch_forward[:len(good_batch_forward)], dim=-1).mean()
    loss = similarity_criterion(good_first_half, good_second_half, bad_batch_forward)

    return {
        "good_similarity": good_similarity,
        "bad_similarity": bad_similarity,
        "loss": loss
    }


def log_training_metrics(index, similarity_loss, good_similarity, bad_similarity, accuracy, series):
    clearml_poc.add_point_to_graph(title="similarity_loss", series=series, x=index, y=similarity_loss)
    clearml_poc.add_point_to_graph(title="good_similarity", series=series, x=index, y=good_similarity)
    clearml_poc.add_point_to_graph(title="bad_similarity", series=series, x=index, y=bad_similarity)
    clearml_poc.add_point_to_graph(title="accuracy", series=series, x=index, y=accuracy)


if __name__ == "__main__":
    clearml_poc.clearml_init()
    assert torch.cuda.is_available(), "no gpu available"
    args: Arguments = Arguments()
    llm = llm_interface.LLMInterface(args.llm_id, interested_layers=[args.llm_hidden_layer])
    llm_hidden_layer = args.llm_hidden_layer

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    similarity_model = ContrastiveMLP(*args.contrastive_mlp_sizes).to(device)
    classifier_model = Detector(args.contrastive_mlp_sizes[-1]).to(device)
    optimizer = torch.optim.Adam(list(similarity_model.parameters()) + list(classifier_model.parameters()), lr=args.lr)
    similarity_criterion = ContrastiveLoss()
    classifier_criterion = torch.nn.CrossEntropyLoss()

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
