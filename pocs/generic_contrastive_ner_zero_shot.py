import asyncio
import dataclasses
import json
from collections import defaultdict
from functools import partial
import pandas as pd
import clearml_poc
from contrastive.args import Arguments, FineTuneLLM
from contrastive.mlp import ContrastiveMLP
from contrastive.loss import ContrastiveLoss
import torch
from contrastive import helper
import torch.nn.functional as F
from tqdm import trange, tqdm
from sklearn import metrics
from abc import ABC, abstractmethod
from typing import AsyncIterator, Tuple, List, Dict, Any


class AbstractDataProvider(ABC):
    @abstractmethod
    async def yield_train_dataset(self, anchor_type: str, batch_size: int, instances_per_type: int,
                                  hard_negative_ratio: int, llm_layer: int,
                                  similarity_strategy: str) -> AsyncIterator[Tuple[Any, Any, Any]]:
        """Yields batches of training data."""
        pass

    @abstractmethod
    async def yield_test_dataset(self, anchor_type: str, batch_size: int, instances_per_type: int,
                                 llm_layer: int, similarity_strategy: str) -> AsyncIterator[Tuple[Any, Any, Any]]:
        """Yields batches of test data."""
        pass

    @abstractmethod
    def pick_llm_output_for_document(self, device: torch.device, input_tokens: int, llm_layer: int,
                                     is_fine_tune_llm: bool, documents: List[Any]) -> torch.Tensor:
        """Extracts LLM output for a document."""
        pass

    @abstractmethod
    def train_fine_types(self) -> List[str]:
        """Returns a list of fine-grained entity types for training."""
        pass

    @abstractmethod
    def test_fine_types(self) -> List[str]:
        """Returns a list of fine-grained entity types for testing."""
        pass

    @abstractmethod
    def load_entity_name_embeddings(self, layer_name: str, entity_name_strategy: str) -> Dict[str, torch.Tensor]:
        """Loads embeddings for entity names."""
        pass

    @abstractmethod
    def llm_and_layer_to_elastic_name(self, llm_id: str, layer: str) -> str:
        """Maps LLM ID and layer to an elastic name."""
        pass


class FewnerdDataProvider(AbstractDataProvider):
    def __init__(self):
        from clearml_pipelines.fewnerd_pipeline import fewnerd_dataset
        from contrastive import fewnerd_processor
        self.fewnerd_dataset = fewnerd_dataset
        self.fewnerd_processor = fewnerd_processor

    async def yield_train_dataset(self, anchor_type: str, batch_size: int, instances_per_type: int,
                                  hard_negative_ratio: int, llm_layer: int,
                                  similarity_strategy: str) -> AsyncIterator[Tuple[Any, Any, Any]]:
        async for anchor, same_type, different_type in self.fewnerd_processor.yield_train_dataset(
                anchor_type=anchor_type,
                batch_size=batch_size,
                instances_per_type=instances_per_type,
                hard_negative_ratio=hard_negative_ratio,
                llm_layer=llm_layer,
                similarity_strategy=similarity_strategy
        ):
            yield anchor, same_type, different_type

    async def yield_test_dataset(self, anchor_type: str, batch_size: int, instances_per_type: int,
                                 llm_layer: int, similarity_strategy: str) -> AsyncIterator[Tuple[Any, Any, Any]]:
        async for anchor, same_type, different_type in self.fewnerd_processor.yield_test_dataset(
                anchor_type=anchor_type,
                batch_size=batch_size,
                instances_per_type=instances_per_type,
                llm_layer=llm_layer,
                similarity_strategy=similarity_strategy
        ):
            yield anchor, same_type, different_type

    def pick_llm_output_for_document(self, device: torch.device, input_tokens: int, llm_layer: int,
                                     is_fine_tune_llm: bool, documents: List[Any]) -> torch.Tensor:
        return self.fewnerd_processor.pick_llm_output_for_document(
            device=device,
            input_tokens=input_tokens,
            llm_layer=llm_layer,
            is_fine_tune_llm=is_fine_tune_llm,
            documents=documents if isinstance(documents, list) else [documents]
        )

    def train_fine_types(self) -> List[str]:
        return self.fewnerd_processor.train_fine_types()

    def test_fine_types(self) -> List[str]:
        return self.fewnerd_processor.test_fine_types()

    def load_entity_name_embeddings(self, layer_name: str, entity_name_strategy: str) -> Dict[str, torch.Tensor]:
        return self.fewnerd_processor.load_entity_name_embeddings(
            layer_name=layer_name,
            entity_name_strategy=entity_name_strategy
        )

    def llm_and_layer_to_elastic_name(self, llm_id: str, layer: str) -> str:
        return self.fewnerd_dataset.llm_and_layer_to_elastic_name(
            llm_id=llm_id,
            layer=layer
        )


class AbstractMLPTrainer:
    def __init__(self, mlp_args: Arguments, llm_args: FineTuneLLM, clearml_poc, data_provider: AbstractDataProvider, device: torch.device):
        self.mlp_args = mlp_args
        self.llm_args = llm_args
        self.clearml_poc = clearml_poc
        self.data_provider = data_provider
        self.device = device
        model = ContrastiveMLP(mlp_args).to(device).double()
        self.instances_model = model
        self.types_model = model
        self.similarity_criterion = ContrastiveLoss(loss_fn=mlp_args.loss_fn, margin=mlp_args.triplet_loss_margin)
        self.instances_optimizer = torch.optim.Adam(self.instances_model.parameters(), lr=mlp_args.lr)
        self.types_optimizer = torch.optim.Adam(self.types_model.parameters(), lr=mlp_args.lr)
        self.max_benchmark = 0
        self.current_benchmark = 0
        self.semaphore = asyncio.Semaphore(mlp_args.semaphore_size)
        self.pbar = None
        self.instances_model_clearml = clearml_poc.generate_tracked_model(
            name="instances_model",
            framework="PyTorch",
            config_dict=dataclasses.asdict(mlp_args)
        )
        self.layer_to_tensor_test = {
            entity_type: tensor.to(device)
            for entity_type, tensor in self._load_entity_name_embeddings().items()
            if entity_type in self.data_provider.test_fine_types()
        }
        self.layer_to_tensor_train = {
            entity_type: tensor.to(device)
            for entity_type, tensor in self._load_entity_name_embeddings().items()
            if entity_type in self.data_provider.train_fine_types()
        }

    def _load_entity_name_embeddings(self) -> Dict[str, torch.Tensor]:
        layer_name = self.data_provider.llm_and_layer_to_elastic_name(
            llm_id=self.llm_args.llm_id,
            layer=self.llm_args.layer
        )
        return self.data_provider.load_entity_name_embeddings(
            layer_name=layer_name,
            entity_name_strategy=self.mlp_args.entity_name_embeddings
        )

    def tensorify(self, *document_items):
        for document_item in document_items:
            result = self.data_provider.pick_llm_output_for_document(
                device=self.device,
                input_tokens=self.mlp_args.input_tokens,
                llm_layer=self.mlp_args.llm_layer,
                is_fine_tune_llm=self.mlp_args.fine_tune_llm,
                documents=document_item if isinstance(document_item, list) else [document_item]
            )
            yield result

    def upload_models(self, epoch):
        file_path = f'instances_model.pt'
        torch.save(self.instances_model.state_dict(), file_path)
        self.clearml_poc.upload_model_to_clearml(self.instances_model_clearml, file_path)
        self.clearml_poc.add_tags([str(self.instances_model_clearml.id)])

    async def iterate_over_train(self, fine_type, similarity_strategy):
        async for anchor, same_type, different_type in self.data_provider.yield_train_dataset(
                anchor_type=fine_type,
                batch_size=self.mlp_args.batch_size,
                instances_per_type=self.mlp_args.instances_per_type,
                hard_negative_ratio=self.mlp_args.hard_negative_ratio,
                llm_layer=self.mlp_args.llm_layer,
                similarity_strategy=similarity_strategy
        ):
            good_batch, bad_batch = self.tensorify(same_type, different_type)
            if similarity_strategy == 'instance':
                anchor = next(self.tensorify(anchor))
            yield anchor, good_batch, bad_batch

    async def one_type_epoch_training(self, fine_type, epoch):
        async with self.semaphore:
            results = defaultdict(list)
            async for anchor, good_batch, bad_batch in self.iterate_over_train(fine_type, similarity_strategy='instance'):
                self.instances_optimizer.zero_grad()
                self.types_optimizer.zero_grad()
                similarity = partial(self.compute_similarity_base, positive_examples=good_batch, negative_examples=bad_batch, epoch=epoch)

                for metric, value in similarity(self.instances_model, anchor=anchor).items():
                    results[f'instances_{metric}'].extend(value)

                self.instances_optimizer.step()
                self.types_optimizer.step()

            async for anchor, good_batch, bad_batch in self.iterate_over_train(fine_type, similarity_strategy='type'):
                self.instances_optimizer.zero_grad()
                self.types_optimizer.zero_grad()
                similarity = partial(self.compute_similarity_base, positive_examples=good_batch, negative_examples=bad_batch, epoch=epoch)
                anchor_tensor = self.layer_to_tensor_train[fine_type]
                for metric, value in similarity(self.types_model, anchor=anchor_tensor).items():
                    results[f'types_{metric}'].extend(value)

                self.instances_optimizer.step()
                self.types_optimizer.step()

            self.pbar.update(1)
            return results

    def compute_similarity_base(self, model, anchor, positive_examples, negative_examples, epoch):
        anchor = torch.atleast_2d(anchor)
        good_similarities = []
        bad_similarities = []
        losses = []
        anchor = model(anchor)
        positive_examples = self.instances_model(positive_examples)
        negative_examples = self.instances_model(negative_examples)
        loss = self.similarity_criterion(anchor, positive_examples, negative_examples, epoch=epoch)
        loss.backward()

        losses.append(loss.item())
        good_similarities.append(F.cosine_similarity(anchor, positive_examples, dim=-1).mean().item())
        bad_similarities.append(F.cosine_similarity(anchor, negative_examples, dim=-1).mean().item())

        return {
            "good_similarity": good_similarities,
            "bad_similarity": bad_similarities,
            "loss": losses
        }

    def train_epoch(self, epoch):
        self.instances_model.train()
        self.types_model.train()

        results_so_far = defaultdict(list)

        all_train_types = self.data_provider.train_fine_types()
        tasks = [self.one_type_epoch_training(fine_type, epoch) for fine_type in all_train_types]
        self.pbar = tqdm(total=len(all_train_types))
        loop = asyncio.get_event_loop()
        task_results = loop.run_until_complete(asyncio.gather(*tasks))

        for result in task_results:
            for key, value in result.items():
                results_so_far[key].extend(value)

        results = {metric: self.avg(value) for metric, value in results_so_far.items()}

        self.log_training_metrics(epoch, series="train", **results)
        torch.cuda.empty_cache()

    async def iterate_over_test(self, fine_type, similarity_strategy):
        async for anchor, same_type, different_type in self.data_provider.yield_test_dataset(
                anchor_type=fine_type,
                batch_size=self.mlp_args.batch_size,
                instances_per_type=self.mlp_args.instances_per_type,
                llm_layer=self.mlp_args.llm_layer,
                similarity_strategy=similarity_strategy
        ):
            good_batch, bad_batch = self.tensorify(same_type, different_type)
            if similarity_strategy == 'instance':
                anchor = next(self.tensorify(anchor))
            yield anchor, good_batch, bad_batch

    async def one_type_epoch_evaluation(self, fine_type, epoch):
        async with self.semaphore:
            results = defaultdict(list)
            async for anchor, good_batch, bad_batch in self.iterate_over_test(fine_type, similarity_strategy='instance'):
                similarity = partial(self.compute_similarity_base, positive_examples=good_batch, negative_examples=bad_batch, epoch=epoch)
                prediction = partial(self.eval_predict, good_batch=good_batch, bad_batch=bad_batch)

                for metric, value in similarity(self.instances_model, anchor=anchor).items():
                    results[f'instances_{metric}'].extend(value)

                for metric, value in prediction(self.instances_model, anchor=anchor).items():
                    results[f'instances_{metric}'].extend(value)

            async for anchor, good_batch, bad_batch in self.iterate_over_test(fine_type, similarity_strategy='type'):
                similarity = partial(self.compute_similarity_base, positive_examples=good_batch, negative_examples=bad_batch, epoch=epoch)
                prediction = partial(self.eval_predict, good_batch=good_batch, bad_batch=bad_batch)

                anchor_tensor = self.layer_to_tensor_test[fine_type]

                for metric, value in similarity(self.types_model, anchor=anchor_tensor).items():
                    results[f'types_{metric}'].extend(value)

                for metric, value in prediction(self.types_model, anchor=anchor_tensor).items():
                    results[f'types_{metric}'].extend(value)
            self.pbar.update(1)
            return results

    def eval_predict(self, model, anchor, good_batch, bad_batch):
        predictions, ground_truth = [], []
        with torch.no_grad():
            anchor_mlp = model(anchor)
            good_batch_mlp = self.instances_model(good_batch)
            bad_batch_mlp = self.instances_model(bad_batch)

        predictions.extend(torch.cosine_similarity(anchor_mlp, good_batch_mlp, dim=1).cpu().tolist())
        predictions.extend(torch.cosine_similarity(anchor_mlp, bad_batch_mlp, dim=1).cpu().tolist())
        ground_truth.extend([1] * len(good_batch_mlp))
        ground_truth.extend([0] * len(bad_batch_mlp))
        return {"predictions": predictions, "ground_truth": ground_truth}

    def evaluate_epoch(self, epoch):
        self.instances_model.eval()
        self.types_model.eval()

        results_so_far = defaultdict(list)
        auc_threshold_graph = defaultdict(list)

        all_test_types = self.data_provider.test_fine_types()
        tasks = [self.one_type_epoch_evaluation(fine_type, epoch) for fine_type in all_test_types]
        self.pbar = tqdm(total=len(all_test_types))
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(asyncio.gather(*tasks))
        for fine_type, result in zip(all_test_types, results):
            for key, value in result.items():
                results_so_far[key].extend(value)
            type_auc = metrics.roc_auc_score(y_score=result["types_predictions"], y_true=result["types_ground_truth"])
            type_optimal_threshold, type_accuracy = helper.find_optimal_threshold(
                y_score=result["types_predictions"],
                y_true=result["types_ground_truth"]
            )
            auc_threshold_graph["type"].append(fine_type)
            auc_threshold_graph["threshold"].append(type_optimal_threshold)
            auc_threshold_graph["accuracy"].append(type_accuracy)
            auc_threshold_graph["auc"].append(type_auc)

        types_auc = metrics.roc_auc_score(
            y_score=results_so_far.pop("types_predictions"), y_true=results_so_far.pop("types_ground_truth")
        )
        instances_auc = metrics.roc_auc_score(
            y_score=results_so_far.pop("instances_predictions"), y_true=results_so_far.pop("instances_ground_truth")
        )
        results_so_far = {metric: self.avg(value) for metric, value in results_so_far.items()}

        results = {
            **results_so_far,
            "types_auc": types_auc,
            "instances_auc": instances_auc,
        }
        self.log_training_metrics(
            epoch,
            series="eval",
            **results
        )

        index_column = auc_threshold_graph.pop("type")
        self.clearml_poc.add_table(
            title="per fine type",
            series="eval",
            iteration=epoch,
            table=pd.DataFrame(data=auc_threshold_graph, index=index_column)
        )

        self.current_benchmark = types_auc
        self.max_benchmark = max(self.current_benchmark, self.max_benchmark)
        torch.cuda.empty_cache()

    def log_training_metrics(self, index, series, **kwargs):
        for metric, value in kwargs.items():
            if value:
                self.clearml_poc.add_point_to_graph(title=metric, series=series, x=index, y=value)

    def avg(self, l):
        return sum(l) / len(l) if l else 0

    def main_loop(self):
        for e in trange(self.mlp_args.epochs):
            self.train_epoch(e)
            self.evaluate_epoch(e)
            self.upload_models(e)


if __name__ == "__main__":
    clearml_poc.clearml_init(queue_name='a100_gpu')
    mlp_args = Arguments()
    llm_args = FineTuneLLM()
    clearml_poc.clearml_connect_hyperparams(mlp_args, "mlp_args")
    clearml_poc.clearml_connect_hyperparams(llm_args, "llm_args")
    print('args are: ', json.dumps(mlp_args.__dict__, indent=4))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert torch.cuda.is_available(), "no gpu available"

    # Instantiate the Fewnerd data provider
    fewnerd_data_provider = FewnerdDataProvider()

    # Instantiate the abstract trainer with the Fewnerd data provider
    trainer = AbstractMLPTrainer(mlp_args, llm_args, clearml_poc, fewnerd_data_provider, device)

    # Run the training loop
    trainer.main_loop()

    print(f"Max benchmark achieved: {trainer.max_benchmark}")