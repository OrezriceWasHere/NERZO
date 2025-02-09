import asyncio
import json
import math
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
from abc import ABC, abstractmethod


def avg(l):
    return sum(l) / len(l)


def log_training_metrics(index, series, **kwargs):
    for metric, value in kwargs.items():
        if value:
            clearml_poc.add_point_to_graph(title=metric, series=series, x=index, y=value)


class ContrastiveNER(ABC):


    def __init__(self):
        clearml_poc.clearml_init()
        self.max_benchmark = 0
        self.current_benchmark = 0
        assert torch.cuda.is_available(), "no gpu available"
        self.args: Arguments = Arguments()
        clearml_poc.clearml_connect_hyperparams(self.args)
        print('args are: ', json.dumps(self.args.__dict__, indent=4))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.similarity_criterion = ContrastiveLoss(loss_fn=self.args.loss_fn, margin=self.args.triplet_loss_margin)

        self.classifier_criterion = torch.nn.CrossEntropyLoss()
        self.classifier_model = Detector().to(self.device)

        self.similarity_model_clearml = clearml_poc.generate_tracked_model(name="similarity_model", framework="PyTorch")
        self.classifier_model_clearml = clearml_poc.generate_tracked_model(name="classifier_model", framework="PyTorch")

        if not self.args.fine_tune_llm:

            self.similarity_model = ContrastiveMLP(self.args).to(self.device)
            self.optimizer = torch.optim.Adam(list(self.similarity_model.parameters()) + list(self.classifier_model.parameters()),
                                         lr=self.args.lr)

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

            self.llm = LLMInterface(llm_id=fine_tune_llm_args.llm_id,
                               interested_layers=[fine_tune_llm_args.layer],
                               max_llm_layer=fine_tune_llm_args.max_llm_layer,
                               lora_config=lora_config)
            # Make sure only LoRA parameters are trainable
            if RuntimeArgs.debug_llm:
                for param in self.llm.model.parameters():
                    param.requires_grad = False

                # Enable gradient computation for LoRA-specific parameters
                for name, param in self.llm.model.named_parameters():
                    if "lora" in name:  # Adjust to your naming scheme
                        param.requires_grad = True

            self.optimizer = torch.optim.Adam(
                [p for p in self.llm.model.parameters() if p.requires_grad] + list(self.classifier_model.parameters()),
                lr=self.args.lr)
            self.similarity_model = self.llm.model


    def main(self):
        for e in trange(self.args.epochs):
            self.train(e)
            self.evaluate(e)
            self.upload_models(e)


    def upload_models(self, epoch):
        if math.isclose(self.current_benchmark, self.max_benchmark):
            print(f"recognized better benchmark value {self.max_benchmark}. Uploading model")

            torch.save(self.similarity_model.state_dict(), f"similarity_model.pt")
            clearml_poc.upload_model_to_clearml(self.similarity_model_clearml, "similarity_model.pt")
            clearml_poc.add_tags([f"similarity model {str(self.similarity_model_clearml.id) if self.similarity_model_clearml else ''}" ])
            if not self.args.fine_tune_llm:
                torch.save(self.classifier_model.state_dict(), f"classifier_model.pt")
                clearml_poc.upload_model_to_clearml(self.classifier_model_clearml, "classifier_model.pt")
                clearml_poc.add_tags([f"classifier model {str(self.classifier_model_clearml.id) if self.classifier_model_clearml else ''}" ])

    def tensorify(self, *document_items):
        for document_item in document_items:
            yield fewnerd_processor.pick_llm_output_for_document(
                device=self.device,
                input_tokens=self.args.input_tokens,
                llm_layer=self.args.llm_layer,
                is_fine_tune_llm=self.args.fine_tune_llm,
                llm=self.llm if self.args.fine_tune_llm else None,
                documents=document_item if isinstance(document_item, list) else [document_item]
            )

    @abstractmethod
    async def one_type_epoch_training(self, fine_type):
        raise NotImplementedError()

    @abstractmethod
    async def one_type_epoch_evaluation(self, fine_type):
        raise NotImplementedError()

    def train(self, epoch):
        self.similarity_model.train()
        self.classifier_model.train()

        results_so_far = defaultdict(list)

        all_train_types = fewnerd_processor.train_fine_types()
        tasks = [self.one_type_epoch_training(fine_type) for fine_type in all_train_types]
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


    def evaluate(self, epoch):
        self.similarity_model.eval()
        self.classifier_model.eval()

        results_so_far = defaultdict(list)
        auc_threshold_graph = defaultdict(list)

        all_test_types = fewnerd_processor.test_fine_types()
        tasks = [self.one_type_epoch_evaluation(fine_type) for fine_type in all_test_types]
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

        if RuntimeArgs.upload_all_predictions:
            accuracy_at_prediction = fewnerd_processor.compute_accuracy_at_prediction(
                predictions=results_so_far["predictions"],
                ground_truths=results_so_far["ground_truth"])
            best_accuracy = accuracy_at_prediction["accuracy_if_threshold_was_here"].max()
            tpr, fpr, _ = metrics.roc_curve(y_score=results_so_far["predictions"], y_true=results_so_far["ground_truth"])

            clearml_poc.add_scatter(title="roc_curve",
                                    series="eval",
                                    iteration=epoch,
                                    values=np.vstack((tpr, fpr)).transpose())


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


        index_column = auc_threshold_graph.pop("type")
        clearml_poc.add_table(title="per fine type",
                              series="eval",
                              iteration=epoch,
                              table=pd.DataFrame(data=auc_threshold_graph, index=index_column))

        self.current_benchmark = auc
        self.max_benchmark = max(self.current_benchmark, self.max_benchmark)


    def forward_similarity_model(self, x, compute_grad=False, detach=True):
        if self.args.disable_similarity_training:
            return x

        if not self.args.fine_tune_llm:
            if compute_grad:
                x = self.similarity_model(x)
            else:
                with torch.no_grad():
                    x = self.similarity_model(x)
        else:
            # when fine tuning llm, the similiraity model is the llm model
            pass

        if detach:
            x = x.clone().detach()
        return x


    def compute_accuracy(self, anchor, good_batch, bad_batch):
        anchor_forward = self.forward_similarity_model(anchor, compute_grad=False, detach=True).float()
        good_batch_forward = self.forward_similarity_model(good_batch, compute_grad=False, detach=True).float()
        bad_batch_forward = self.forward_similarity_model(bad_batch, compute_grad=False, detach=True).float()

        accuracies, losses = [], []
        labels = torch.tensor([1] * len(good_batch_forward) + [0] * len(bad_batch_forward)).to(self.device)
        true_classification_prediction = self.classifier_model(anchor_forward, good_batch_forward)
        false_classfication_prediction = self.classifier_model(anchor_forward, bad_batch_forward)
        classification_predication = torch.cat((true_classification_prediction, false_classfication_prediction), dim=0)
        classifier_loss = self.classifier_criterion(classification_predication, labels)
        classifier_loss.backward()
        accuracy = torch.sum(torch.argmax(classification_predication, dim=-1) == labels).item() / \
                   classification_predication.shape[0]
        accuracies.append(accuracy)
        losses.append(classifier_loss.item())

        return {
            "accuracy": avg(accuracies),
            "loss": avg(losses)
        }


    def compute_similarity(self,
            anchor,
            positive_examples,
            negative_examples):
        good_similarities = []
        bad_similarities = []
        losses = []

        anchor = self.forward_similarity_model(anchor, compute_grad=True, detach=False)
        positive_examples = self.forward_similarity_model(positive_examples, compute_grad=True, detach=False)
        negative_examples = self.forward_similarity_model(negative_examples, compute_grad=True, detach=False)
        loss = self.similarity_criterion(anchor, positive_examples, negative_examples)
        if not self.args.disable_similarity_training:
            loss.backward()

        losses.append(loss.item())
        good_similarities.append(F.cosine_similarity(anchor, positive_examples, dim=-1).mean().item())
        bad_similarities.append(F.cosine_similarity(anchor, negative_examples, dim=-1).mean().item())

        return {
            "good_similarity": avg(good_similarities),
            "bad_similarity": avg(bad_similarities),
            "loss": avg(losses)
        }
