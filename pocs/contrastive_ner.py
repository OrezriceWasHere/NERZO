import asyncio
import json

import clearml_poc
from contrastive.args import Arguments, FineTuneLLM
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

    upload_models()


def avg(l):
    return sum(l) / len(l)


def upload_models():
    torch.save(similarity_model.state_dict(), f"similarity_model.pt")
    similarity_model_clearml = clearml_poc.generate_tracked_model(name="similarity_model", framework="PyTorch")
    clearml_poc.upload_model_to_clearml(similarity_model_clearml, "similarity_model.pt")

    if not args.fine_tune_llm:
        torch.save(classifier_model.state_dict(), f"classifier_model.pt")
        classifier_model_clearml = clearml_poc.generate_tracked_model(name="classifier_model", framework="PyTorch")
        clearml_poc.upload_model_to_clearml(classifier_model_clearml, "classifier_model.pt")

def tensorify(*document_items):
    for document_item in document_items:
        yield fewnerd_processor.pick_llm_output_for_document(
            device=device,
            input_tokens=args.input_tokens,
            llm_layer=args.llm_layer,
            is_fine_tune_llm=args.fine_tune_llm,
            llm=llm or None,
            documents=document_item if isinstance(document_item, list) else [document_item]
        )

async def one_type_epoch_training(fine_type):
    good_similarities = []
    bad_similarities = []
    losses = []
    classifier_accuracies = []
    classifier_losses = []
    async for anchor, same_type, different_type in fewnerd_processor.yield_train_dataset(
            anchor_type=fine_type,
            batch_size=args.batch_size,
            instances_per_type=args.instances_per_type,
            llm_layer=args.llm_layer):
        optimizer.zero_grad()
        anchor, good_batch, bad_batch = tensorify(anchor, same_type, different_type)
        good_similarity, bad_similarity, similarity_loss = compute_similarity(
            anchor,
            good_batch,
            bad_batch).values()
        classifier_accuracy, classifier_loss = compute_accuracy(anchor, good_batch, bad_batch).values()
        optimizer.step()

        good_similarities.append(good_similarity)
        bad_similarities.append(bad_similarity)
        losses.append(similarity_loss)
        classifier_accuracies.append(classifier_accuracy)
        classifier_losses.append(classifier_loss)

    return {
        "good_similarity": good_similarities,
        "bad_similarity": bad_similarities,
        "similarity_loss": losses,
        "classifier_accuracy": classifier_accuracies,
        "classifier_loss": classifier_losses
    }


def train(epoch):
    similarity_model.train()
    classifier_model.train()

    losses = []
    good_similarities = []
    bad_similarities = []
    classifier_accuracies = []

    all_train_types = fewnerd_processor.train_fine_types()
    tasks = [one_type_epoch_training(fine_type) for fine_type in all_train_types]
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(asyncio.gather(*tasks))
    for result in results:
        good_similarities.extend(result["good_similarity"])
        bad_similarities.extend(result["bad_similarity"])
        losses.extend(result["similarity_loss"])
        classifier_accuracies.extend(result["classifier_accuracy"])

    log_training_metrics(epoch, avg(losses), avg(good_similarities), avg(bad_similarities),
                         avg(classifier_accuracies),
                         series="train")

    del losses, good_similarities, bad_similarities, classifier_accuracies


def compute_accuracy_at_prediction(predictions: list[float], ground_truths: list[int]) -> pd.DataFrame:
    p_numpy = np.asarray(predictions)
    gt_numpy = np.asarray(ground_truths)
    accuracies = [accuracy_score(gt_numpy, p_numpy >= prediction) for prediction, ground_truth in
                  zip(predictions, ground_truths)]
    return pd.DataFrame({"prediction": p_numpy,
                         "ground_truth": gt_numpy,
                         "accuracy_if_threshold_was_here": np.asarray(accuracies)})


async def one_type_epoch_evaluation(fine_type):
    good_similarities = []
    bad_similarities = []
    losses = []
    classifier_accuracies = []
    predictions = []
    ground_truth = []
    async for anchor, same_type, different_type in fewnerd_processor.yield_test_dataset(anchor_type=fine_type,
                                                                                        batch_size=args.batch_size,
                                                                                        instances_per_type=args.instances_per_type,
                                                                                        llm_layer=args.llm_layer):
        anchor, good_batch, bad_batch = tensorify(anchor, same_type, different_type)

        good_similarity, bad_similarity, similarity_loss = compute_similarity(
            anchor,
            good_batch,
            bad_batch).values()
        classifier_accuracy, classifier_loss = compute_accuracy(anchor, good_batch, bad_batch).values()

        losses.append(classifier_loss + similarity_loss)
        good_similarities.append(good_similarity)
        bad_similarities.append(bad_similarity)
        classifier_accuracies.append(classifier_accuracy)

        anchor_mlp = forward_similarity_model(anchor, compute_grad=False, detach=False)
        good_batch_mlp = forward_similarity_model(good_batch, compute_grad=False, detach=False)
        bad_batch_mlp = forward_similarity_model(bad_batch, compute_grad=False, detach=False)

        predictions.extend(torch.cosine_similarity(anchor_mlp, good_batch_mlp, dim=1).cpu().tolist())
        predictions.extend(torch.cosine_similarity(anchor_mlp, bad_batch_mlp, dim=1).cpu().tolist())

        ground_truth.extend([1] * len(good_batch_mlp))
        ground_truth.extend([0] * len(bad_batch_mlp))

    return {
        "good_similarity": good_similarities,
        "bad_similarity": bad_similarities,
        "loss": losses,
        "classifier_accuracy": classifier_accuracies,
        "predictions": predictions,
        "ground_truth": ground_truth
    }


def evaluate(epoch):
    similarity_model.eval()
    classifier_model.eval()

    losses = []
    good_similarities = []
    bad_similarities = []
    classifier_accuracies = []
    predictions = []
    ground_truth = []

    all_test_types = fewnerd_processor.test_fine_types()
    tasks = [one_type_epoch_evaluation(fine_type) for fine_type in all_test_types]
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(asyncio.gather(*tasks))
    for result in results:
        good_similarities.extend(result["good_similarity"])
        bad_similarities.extend(result["bad_similarity"])
        losses.extend(result["loss"])
        classifier_accuracies.extend(result["classifier_accuracy"])
        predictions.extend(result["predictions"])
        ground_truth.extend(result["ground_truth"])

    if RuntimeArgs.upload_all_predictions:
        accuracy_at_prediction = compute_accuracy_at_prediction(predictions, ground_truth)
        best_accuracy = accuracy_at_prediction["accuracy_if_threshold_was_here"].max()

    else:
        accuracy_at_prediction = None
        best_accuracy = None

    log_training_metrics(epoch, avg(losses), avg(good_similarities), avg(bad_similarities),
                         avg(classifier_accuracies),
                         series="eval",
                         accuracy_at_prediction=accuracy_at_prediction,
                         best_accuracy=best_accuracy)
    del losses, good_similarities, bad_similarities, classifier_accuracies



def forward_similarity_model(x, compute_grad=False, detach=True):
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
    anchor_forward = forward_similarity_model(anchor, compute_grad=False, detach=True).float()
    good_batch_forward = forward_similarity_model(good_batch, compute_grad=False, detach=True).float()
    bad_batch_forward = forward_similarity_model(bad_batch, compute_grad=False, detach=True).float()

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

    anchor = forward_similarity_model(anchor, compute_grad=True, detach=False)
    positive_examples = forward_similarity_model(positive_examples, compute_grad=True, detach=False)
    negative_examples = forward_similarity_model(negative_examples, compute_grad=True, detach=False)
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
    if kwargs.get("accuracy_at_prediction", None):
        clearml_poc.add_table(title="accuracy at prediction", series=series, iteration=index,
                              table=kwargs["accuracy_at_prediction"])
    if kwargs.get("best_accuracy", None):
        clearml_poc.add_point_to_graph(title="best_accuracy", series=series, x=index, y=kwargs["best_accuracy"])


if __name__ == "__main__":
    clearml_poc.clearml_init()
    assert torch.cuda.is_available(), "no gpu available"
    args: Arguments = Arguments()
    clearml_poc.clearml_connect_hyperparams(args)
    print('args are: ', json.dumps(args.__dict__, indent=4))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    similarity_criterion = ContrastiveLoss(loss_fn=args.loss_fn, margin=args.triplet_loss_margin)
    contrastive_criterion = ContrastiveLoss(loss_fn="contrastive_loss", margin=args.triplet_loss_margin)
    triplet_criterion = ContrastiveLoss(loss_fn="triplet_loss", margin=args.triplet_loss_margin)

    classifier_criterion = torch.nn.CrossEntropyLoss()
    classifier_model = Detector().to(device)

    if not args.fine_tune_llm:

        similarity_model = ContrastiveMLP(args).to(device)
        optimizer = torch.optim.Adam(list(similarity_model.parameters()) + list(classifier_model.parameters()),
                                     lr=args.lr)

    else:
        fine_tune_llm_args = FineTuneLLM()
        clearml_poc.clearml_connect_hyperparams(fine_tune_llm_args, name="fine_tune_llm_args")
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

        llm = LLMInterface(llm_id=fine_tune_llm_args.llm_id,
                           interested_layers=[fine_tune_llm_args.layer],
                           max_llm_layer=fine_tune_llm_args.max_llm_layer,
                           lora_config=lora_config)
        # Make sure only LoRA parameters are trainable
        if RuntimeArgs.debug_llm:
            for param in llm.model.parameters():
                param.requires_grad = False

            # Enable gradient computation for LoRA-specific parameters
            for name, param in llm.model.named_parameters():
                if "lora" in name:  # Adjust to your naming scheme
                    param.requires_grad = True

        optimizer = torch.optim.Adam(
            [p for p in llm.model.parameters() if p.requires_grad] + list(classifier_model.parameters()),
            lr=args.lr)
        similarity_model = llm.model

    main()
