import clearml_poc
from contrastive.args import Arguments
from contrastive.mlp import ContrastiveMLP, Detector
from contrastive.loss import ContrastiveLoss
import torch
from contrastive import fewnerd_processor
import torch.nn.functional as F
from tqdm import trange
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


def main():
    for e in trange(args.epochs):
        # train(e)
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

    for anchor, same_type, different_type in fewnerd_processor.yield_train_dataset(batch_size=args.batch_size, instances_per_type=args.instances_per_type):
        optimizer.zero_grad()
        anchor, good_batch, bad_batch = pick_llm_output(anchor, same_type, different_type)

        good_similarity, bad_similarity, similarity_loss = compute_similarity(
            anchor,
            good_batch,
            bad_batch).values()
        classifier_accuracy, classifier_loss = compute_accuracy(anchor, good_batch, bad_batch).values()
        optimizer.step()

        losses.append(classifier_loss + similarity_loss)
        good_similarities.append(good_similarity)
        bad_similarities.append(bad_similarity)
        classifier_accuracies.append(classifier_accuracy)

    log_training_metrics(epoch, avg(losses), avg(good_similarities), avg(bad_similarities),
                         avg(classifier_accuracies),
                         series="train")


def compute_accuracy_at_prediction(predictions: list[float], ground_truths:list[int]) -> pd.DataFrame:
    p_numpy = np.asarray(predictions)
    gt_numpy = np.asarray(ground_truths)
    accuracies = [accuracy_score(gt_numpy, p_numpy >= prediction) for prediction, ground_truth in zip(predictions, ground_truths)]
    return pd.DataFrame({"prediction": p_numpy,
                          "ground_truth": gt_numpy,
                         "accuracy_if_threshold_was_here": np.asarray(accuracies)})



def evaluate(epoch):
    similarity_model.eval()
    classifier_model.eval()

    losses = []
    good_similarities = []
    bad_similarities = []
    classifier_accuracies = []
    predictions = []
    ground_truth = []

    for anchor, same_type, different_type in fewnerd_processor.yield_test_dataset(batch_size=args.batch_size, instances_per_type=args.instances_per_type):
        anchor, good_batch, bad_batch = pick_llm_output(anchor, same_type, different_type)

        good_similarity, bad_similarity, similarity_loss = compute_similarity(
            anchor,
            good_batch,
            bad_batch).values()
        classifier_accuracy, classifier_loss = compute_accuracy(anchor, good_batch, bad_batch).values()

        losses.append(classifier_loss + similarity_loss)
        good_similarities.append(good_similarity)
        bad_similarities.append(bad_similarity)
        classifier_accuracies.append(classifier_accuracy)


        with torch.no_grad():
            anchor_mlp = similarity_model(anchor)
            good_batch_mlp = similarity_model(good_batch)
            bad_batch_mlp = similarity_model(bad_batch)

        predictions.extend(torch.cosine_similarity(anchor_mlp, good_batch_mlp, dim=1).cpu().tolist())
        predictions.extend(torch.cosine_similarity(anchor_mlp, bad_batch_mlp, dim=1).cpu().tolist())

        ground_truth.extend([1] * len(good_batch_mlp))
        ground_truth.extend([0] * len(bad_batch_mlp))

    accuracy_at_prediction = compute_accuracy_at_prediction(predictions, ground_truth)
    best_accuracy = accuracy_at_prediction["accuracy_if_threshold_was_here"].max()


    log_training_metrics(epoch, avg(losses), avg(good_similarities), avg(bad_similarities),
                         avg(classifier_accuracies),
                         series="eval",
                         accuracy_at_prediction=compute_accuracy_at_prediction(predictions, ground_truth),
                         best_accuracy=best_accuracy)


def pick_llm_output(*items):
    tensorify = lambda item: torch.concat((torch.tensor(item["embedding"]["llama_3_17_v_proj"]["end"]), torch.tensor(item["embedding"]["llama_3_17_v_proj"]["start"]))).to(device)
    stack = lambda batch: torch.stack([tensorify(item) for item in batch]) if isinstance(batch, list) else tensorify(
        batch).unsqueeze(0)

    return list(map(stack, items))


def compute_accuracy(anchor, good_batch, bad_batch):
    with torch.no_grad():
        anchor_forward = similarity_model(anchor).clone().detach()
        good_batch_forward = similarity_model(good_batch).clone().detach()
        bad_batch_forward = similarity_model(bad_batch).clone().detach()

    accuracies, losses = [], []
    labels = torch.tensor([1] * len(good_batch_forward) + [0] * len(bad_batch_forward)).to(device)
    true_classification_prediction = classifier_model(anchor_forward, good_batch_forward)
    false_classfication_prediction = classifier_model(anchor_forward, bad_batch_forward)
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

def compute_similarity(
        anchor,
        positive_examples,
        negative_examples):
    good_similarities = []
    bad_similarities = []
    losses = []

    anchor = similarity_model(anchor)
    positive_examples = similarity_model(positive_examples)
    negative_examples = similarity_model(negative_examples)
    loss = similarity_criterion(anchor, positive_examples, negative_examples)
    loss.backward()

    losses.append(loss.item())
    good_similarities.append(F.cosine_similarity(anchor, positive_examples, dim=-1).mean().item())
    bad_similarities.append(F.cosine_similarity(anchor, negative_examples, dim=-1).mean().item())

    return {
        "good_similarity": avg(good_similarities),
        "bad_similarity": avg(bad_similarities),
        "loss": avg(losses)
    }


def log_training_metrics(index, similarity_loss, good_similarity, bad_similarity, accuracy, series, **kwargs):
    clearml_poc.add_point_to_graph(title="similarity_loss", series=series, x=index, y=similarity_loss)
    clearml_poc.add_point_to_graph(title="good_similarity", series=series, x=index, y=good_similarity)
    clearml_poc.add_point_to_graph(title="bad_similarity", series=series, x=index, y=bad_similarity)
    clearml_poc.add_point_to_graph(title="accuracy", series=series, x=index, y=accuracy)
    if "accuracy_at_prediction" in kwargs:
        clearml_poc.add_table(title="accuracy at prediction", series=series, iteration=index, table=kwargs["accuracy_at_prediction"])
    if "best_accuracy" in kwargs:
        clearml_poc.add_point_to_graph(title="best_accuracy", series=series, x=index, y=kwargs["best_accuracy"])


if __name__ == "__main__":
    clearml_poc.clearml_init()
    assert torch.cuda.is_available(), "no gpu available"
    args: Arguments = Arguments()
    clearml_poc.clearml_connect_hyperparams(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    similarity_model = ContrastiveMLP(args).to(device)
    classifier_model = Detector().to(device)
    optimizer = torch.optim.Adam(list(similarity_model.parameters()) + list(classifier_model.parameters()), lr=args.lr)
    similarity_criterion = ContrastiveLoss(loss_fn=args.loss_fn, margin=args.triplet_loss_margin)
    classifier_criterion = torch.nn.CrossEntropyLoss()

    main()
