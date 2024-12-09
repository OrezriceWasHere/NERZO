import json

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
from llm_interface import LLMInterface
from peft import LoraConfig, PeftType

from runtime_args import RuntimeArgs


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

        anchor_mlp = forward_similarity_model(anchor, compute_grad=False,detach=False)
        good_batch_mlp = forward_similarity_model(good_batch, compute_grad=False,detach=False)
        bad_batch_mlp = forward_similarity_model(bad_batch, compute_grad=False,detach=False)


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


def tensorify_db_document(dataset_document: dict | list[dict]) -> list[torch.Tensor]:
    if isinstance(dataset_document, dict):
        dataset_document = [dataset_document]
    texts = [doc["all_text"] for doc in dataset_document]
    texts_indices = [(doc["index_start"], doc["index_end"]) for doc in dataset_document]
    tokens = llm.tokenize(texts).to(device)
    hidden_items = llm.get_llm_at_layer(tokens, layer, clone=False)
    token_indices = [llm.token_indices_given_text_indices(text, text_indices) for text, text_indices in zip(texts, texts_indices)]
    if args.input_tokens == "start_end_pair":
        desired_tokens = [torch.concat((h[token_index[1]], h[token_index[0]])) for h, token_index in zip(hidden_items, token_indices)]
    elif args.input_tokens == "end":
        desired_tokens = [h[token_index[1]] for h, token_index in zip(hidden_items, token_indices)]
    elif args.input_tokens == "diff":
        desired_tokens = [h[token_index[1]] - h[token_index[0]] for h, token_index in zip(hidden_items, token_indices)]
    else:
        raise ValueError(f"input_tokens should be one of ['end', 'diff', 'start_end_pair'] but got {args.input_tokens}")
    return desired_tokens


def pick_llm_output(*items):
    if args.fine_tune_llm:
        tensorify = lambda item: tensorify_db_document(item)
        stack = lambda batch: torch.stack(tensorify(batch))

    elif args.input_tokens == "end":
        tensorify = lambda item: torch.tensor(item["embedding"][args.llm_layer]["end"]).to(device)
        stack = lambda batch: torch.stack([tensorify(item) for item in batch]) if isinstance(batch, list) else tensorify(
            batch).unsqueeze(0)

    elif args.input_tokens == "diff":
        tensorify = lambda item: (torch.tensor(item["embedding"][args.llm_layer]["end"]) - torch.tensor(
            item["embedding"][args.llm_layer]["start"])).to(device)
        stack = lambda batch: torch.stack([tensorify(item) for item in batch]) if isinstance(batch,
                                                                                             list) else tensorify(
            batch).unsqueeze(0)

    elif args.input_tokens == "start_end_pair":
        tensorify = lambda item: torch.concat((torch.tensor(item["embedding"][args.llm_layer]["end"]),
                                               torch.tensor(item["embedding"][args.llm_layer]["start"]))).to(
            device)
        stack = lambda batch: torch.stack([tensorify(item) for item in batch]) if isinstance(batch,
                                                                                             list) else tensorify(
            batch).unsqueeze(0)
    else:
        raise ValueError(f"input_tokens should be one of ['end', 'diff', 'start_end_pair'] but got {args.input_tokens}")

    return list(map(stack, items))


def forward_similarity_model(x, compute_grad=False,detach=True):
    if not args.fine_tune_llm:
        if compute_grad:
                x = similarity_model(x)
        else:
            with torch.no_grad():
                    x = similarity_model(x)
    else:
        # when fine tuning llm, the similiraity model is the llm model
        pass

    if detach:
        x = x.clone().detach()
    return x


def compute_accuracy(anchor, good_batch, bad_batch):
    anchor_forward = forward_similarity_model(anchor, compute_grad=False,detach=True).float()
    good_batch_forward = forward_similarity_model(good_batch, compute_grad=False,detach=True).float()
    bad_batch_forward = forward_similarity_model(bad_batch, compute_grad=False,detach=True).float()

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

    # anchor = similarity_model(anchor)
    # positive_examples = similarity_model(positive_examples)
    # negative_examples = similarity_model(negative_examples)
    anchor = forward_similarity_model(anchor, compute_grad=True,detach=False)
    positive_examples = forward_similarity_model(positive_examples, compute_grad=True,detach=False)
    negative_examples = forward_similarity_model(negative_examples, compute_grad=True,detach=False)
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
    print('args are: ', json.dumps(args.__dict__, indent=4))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    similarity_criterion = ContrastiveLoss(loss_fn=args.loss_fn, margin=args.triplet_loss_margin)
    classifier_criterion = torch.nn.CrossEntropyLoss()
    classifier_model = Detector().to(device)


    if not args.fine_tune_llm:

        similarity_model = ContrastiveMLP(args).to(device)
        optimizer = torch.optim.Adam(list(similarity_model.parameters()) + list(classifier_model.parameters()),
                                     lr=args.lr)

    else:

        LLM_ID = "meta-llama/Meta-Llama-3.1-8B"
        llm_id, layer = "meta-llama/Meta-Llama-3.1-8B", "model.layers.17.self_attn.v_proj"
        if RuntimeArgs.debug_llm:
            lora_config = None
        else:
        # Configure LoRA
            lora_config = LoraConfig(
                r=2,  # Rank of the LoRA matrices
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,  # Scaling factor
                lora_dropout=0.1,  # Dropout probability
                peft_type=PeftType.LORA,
            )

            clearml_poc.clearml_connect_hyperparams(lora_config, name="lora_optim")

        llm = LLMInterface(llm_id=llm_id, interested_layers=[layer], lora_config=lora_config)
        # Make sure only LoRA parameters are trainable
        if RuntimeArgs.debug_llm:
            for param in llm.model.parameters():
                param.requires_grad = False

            # Enable gradient computation for LoRA-specific parameters
            for name, param in llm.model.named_parameters():
                if "lora" in name:  # Adjust to your naming scheme
                    param.requires_grad = True

        optimizer = torch.optim.Adam([p for p in llm.model.parameters() if p.requires_grad] + list(classifier_model.parameters()),
                                     lr=args.lr)
        similarity_model = llm.model


    main()
