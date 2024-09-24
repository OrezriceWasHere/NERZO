import random
import clearml_poc
from contrastive.args import Arguments
from contrastive.mlp import ContrastiveMLP, Detector
from contrastive.loss import ContrastiveLoss
import torch
from contrastive import fewnerd_processor
import torch.nn.functional as F
from tqdm import trange


def main():
    for e in trange(args.epochs):
        train(e)
        evaluate(e)


def avg(l):
    return sum(l) / len(l)


def train(epoch):
    similarity_model.train()
    classifier_model.train()

    losses = []
    good_similarities = []
    bad_similarities = []
    classifier_accuracies = []

    for same_type, different_type in fewnerd_processor.yield_train_dataset():
        optimizer.zero_grad()
        good_batch, bad_batch = pick_llm_output(same_type, different_type)

        good_similarity, bad_similarity, similarity_loss = compute_similarity(
            good_batch,
            bad_batch).values()
        classifier_accuracy, classifier_loss = compute_accuracy(good_batch, bad_batch).values()

        optimizer.step()

        losses.append(classifier_loss + similarity_loss)
        good_similarities.append(good_similarity)
        bad_similarities.append(bad_similarity)
        classifier_accuracies.append(classifier_accuracy)

    log_training_metrics(epoch, avg(losses), avg(good_similarities), avg(bad_similarities),
                         avg(classifier_accuracies),
                         series="train")


def evaluate(index):
    similarity_model.eval()
    classifier_model.eval()

    losses = []
    good_similarities = []
    bad_similarities = []
    classifier_accuracies = []

    for same_type, different_type in fewnerd_processor.yield_test_dataset(batch_size=args.batch_size):
        good_batch, bad_batch = pick_llm_output(same_type, different_type)
        good_batch.requires_grad = False
        bad_batch.requires_grad = False

        good_similarity, bad_similarity, similarity_loss = compute_similarity(
            good_batch,
            bad_batch).values()
        classifier_accuracy, classifier_loss = compute_accuracy(good_batch, bad_batch).values()
        # similarity_loss.backward()

        losses.append(classifier_loss + similarity_loss)
        good_similarities.append(good_similarity)
        bad_similarities.append(bad_similarity)
        classifier_accuracies.append(classifier_accuracy)

    log_training_metrics(index, avg(losses), avg(good_similarities), avg(bad_similarities),
                         avg(classifier_accuracies),
                         series="eval")


def pick_llm_output(same_type, different_type):
    tensorify = lambda batch: torch.stack(
        [torch.tensor(item["embedding"]["llama_3_17_v_proj"]["end"]) for item in batch]).to(device)
    good_batch = tensorify(same_type)
    bad_batch = tensorify(different_type)
    return good_batch, bad_batch


def compute_accuracy(good_batch, bad_batch):
    with torch.no_grad():
        good_batch_forward = similarity_model(good_batch).clone().detach()
        bad_batch_forward = similarity_model(bad_batch).clone().detach()

    accuracies, losses = [], []
    for query, positive_examples, negative_examples in generate_triplets(good_batch_forward, bad_batch_forward, 1):
        labels = torch.tensor([1] * len(positive_examples) + [0] * len(negative_examples)).to(device)
        true_classification_prediction = classifier_model(query, positive_examples)
        false_classfication_prediction = classifier_model(query, negative_examples)
        classification_predication = torch.cat((true_classification_prediction, false_classfication_prediction), dim=0)
        classifier_loss = classifier_criterion(classification_predication, labels)
        classifier_loss.backward()
        accuracy = torch.sum(torch.argmax(classification_predication, dim=-1) == labels).item() / \
                   classification_predication.shape[0]
        accuracies.append(accuracy)
        losses.append(classifier_loss.item())

    return {
        "accuracy": avg(accuracies),
        "loss": avg(losses)
    }


def generate_triplets(good_examples, bad_examples, amount=1) -> tuple[torch.Tensor]:
    amount = min(amount, len(good_examples))
    possible_indices = list(range(good_examples.shape[0]))
    for selected_index in random.choices(possible_indices, k=amount):
        query = good_examples[selected_index].unsqueeze(0)
        positive_examples = torch.cat((good_examples[:selected_index], good_examples[selected_index + 1:]), dim=0)
        negative_examples = bad_examples[:len(positive_examples)]
        yield query, positive_examples, negative_examples


def compute_similarity(
        good_batch,
        bad_batch):
    good_similarities = []
    bad_similarities = []
    losses = []

    number_of_examples = min(args.number_of_examples, good_batch.shape[0] // 2)
    for query, positive_examples, negative_examples in generate_triplets(good_batch, bad_batch, number_of_examples):
        query = similarity_model(query)
        positive_examples = similarity_model(positive_examples)
        negative_examples = similarity_model(negative_examples)
        loss = similarity_criterion(query, positive_examples, negative_examples)
        loss.backward()

        losses.append(loss.item())
        good_similarities.append(F.cosine_similarity(query, positive_examples, dim=-1).mean().item())
        bad_similarities.append(F.cosine_similarity(query, negative_examples, dim=-1).mean().item())

    return {
        "good_similarity": avg(good_similarities),
        "bad_similarity": avg(bad_similarities),
        "loss": avg(losses)
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
    clearml_poc.clearml_connect_hyperparams(args)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    similarity_model = ContrastiveMLP(args).to(device)
    classifier_model = Detector(args.contrastive_mlp_sizes[-1]).to(device)
    optimizer = torch.optim.Adam(list(similarity_model.parameters()) + list(classifier_model.parameters()), lr=args.lr)
    similarity_criterion = ContrastiveLoss(loss_fn=args.loss_fn)
    classifier_criterion = torch.nn.CrossEntropyLoss()

    main()
