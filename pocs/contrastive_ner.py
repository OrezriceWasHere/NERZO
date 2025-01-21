import asyncio
import json
from collections import defaultdict
import pandas as pd
import numpy as np
import clearml_poc
from contrastive.args import Arguments, FineTuneLLM
from contrastive.mlp import ContrastiveMLP, Detector
from contrastive.loss import ContrastiveLoss
import torch
from contrastive import fewnerd_processor, helper
import torch.nn.functional as F
from tqdm import trange
from llm_interface import LLMInterface
from peft import LoraConfig, PeftType
from sklearn import metrics
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
            llm=llm if args.fine_tune_llm else None,
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
        losses.append(similarity_loss + classifier_loss)
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

    results_so_far = defaultdict(list)

    all_train_types = fewnerd_processor.train_fine_types()
    tasks = [one_type_epoch_training(fine_type) for fine_type in all_train_types]
    loop = asyncio.get_event_loop()
    task_results = loop.run_until_complete(asyncio.gather(*tasks))

    for result in task_results:
        for key, value in result.items():
            results_so_far[key].extend(value)

    log_training_metrics(epoch,
                         similarity_loss=avg(results_so_far["similarity_loss"]),
                         good_similarity=avg(results_so_far["good_similarity"]),
                         bad_similarity=avg(results_so_far["bad_similarity"]),
                         accuracy=avg(results_so_far["classifier_accuracy"]),
                         series="train"
                         )


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

        anchor_mlp = forward_similarity_model(anchor, compute_grad=False, detach=False)
        good_batch_mlp = forward_similarity_model(good_batch, compute_grad=False, detach=False)
        bad_batch_mlp = forward_similarity_model(bad_batch, compute_grad=False, detach=False)

        predictions.extend(torch.cosine_similarity(anchor_mlp, good_batch_mlp, dim=1).cpu().tolist())
        predictions.extend(torch.cosine_similarity(anchor_mlp, bad_batch_mlp, dim=1).cpu().tolist())

        ground_truth.extend([1] * len(good_batch_mlp))
        ground_truth.extend([0] * len(bad_batch_mlp))

        losses.append(classifier_loss + similarity_loss)
        good_similarities.append(good_similarity)
        bad_similarities.append(bad_similarity)
        classifier_accuracies.append(classifier_accuracy)

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

    results_so_far = defaultdict(list)
    auc_threshold_graph = defaultdict(list)

    all_test_types = fewnerd_processor.test_fine_types()
    tasks = [one_type_epoch_evaluation(fine_type) for fine_type in all_test_types]
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(asyncio.gather(*tasks))
    for fine_type, result in zip(all_test_types, results):
        for key, value in result.items():
            results_so_far[key].extend(value)
        type_auc = metrics.roc_auc_score(y_score=result["predictions"], y_true=result["ground_truth"])
        type_optimal_threshold, type_accuracy = helper.find_optimal_threshold(y_score=result["predictions"],
                                                               y_true=result["ground_truth"])
        auc_threshold_graph["type"].append(fine_type)
        auc_threshold_graph["threshold"].append(type_optimal_threshold)
        auc_threshold_graph["accuracy"].append(type_accuracy)
        auc_threshold_graph["auc"].append(type_auc)

    auc = metrics.roc_auc_score(y_score=results_so_far["predictions"], y_true=results_so_far["ground_truth"])
    tpr, fpr, _ = metrics.roc_curve(y_score=results_so_far["predictions"], y_true=results_so_far["ground_truth"])

    if RuntimeArgs.upload_all_predictions:
        accuracy_at_prediction = fewnerd_processor.compute_accuracy_at_prediction(
            predictions=results_so_far["predictions"],
            ground_truths=results_so_far["ground_truth"])
        best_accuracy = accuracy_at_prediction["accuracy_if_threshold_was_here"].max()

    else:
        accuracy_at_prediction = None
        best_accuracy = None

    log_training_metrics(epoch,
                         similarity_loss=avg(results_so_far["loss"]),
                         good_similarity=avg(results_so_far["good_similarity"]),
                         bad_similarity=avg(results_so_far["bad_similarity"]),
                         accuracy=avg(results_so_far["classifier_accuracy"]),
                         series="eval",
                         accuracy_at_prediction=accuracy_at_prediction,
                         best_accuracy=best_accuracy,
                         auc=auc)

    clearml_poc.add_scatter(title="roc_curve",
                            series="eval",
                            iteration=epoch,
                            values=np.vstack((tpr, fpr)).transpose())

    index_column = auc_threshold_graph.pop("type")
    clearml_poc.add_table(title="per fine type",
                          series="eval",
                          iteration=epoch,
                          table=pd.DataFrame(data=auc_threshold_graph, index=index_column))


def forward_similarity_model(x, compute_grad=False, detach=True):
    if args.disable_similarity_training:
        return x

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
    if not args.disable_similarity_training:
        loss.backward()

    losses.append(loss.item())
    good_similarities.append(F.cosine_similarity(anchor, positive_examples, dim=-1).mean().item())
    bad_similarities.append(F.cosine_similarity(anchor, negative_examples, dim=-1).mean().item())

    return {
        "good_similarity": avg(good_similarities),
        "bad_similarity": avg(bad_similarities),
        "loss": avg(losses)
    }


def log_training_metrics(index, series, **kwargs):
    for metric, value in kwargs.items():
        if value:
            clearml_poc.add_point_to_graph(title=metric, series=series, x=index, y=value)


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
