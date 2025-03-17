import asyncio
import dataclasses
import json
import uuid
from collections import defaultdict
from functools import partial
import math
import pandas as pd
import clearml_poc
from clearml_pipelines.fewnerd_pipeline import fewnerd_dataset
from contrastive.args import Arguments, FineTuneLLM
from contrastive.mlp import ContrastiveMLP, Detector
from contrastive.loss import ContrastiveLoss
import torch
from contrastive import fewnerd_processor, helper
import torch.nn.functional as F
from tqdm import trange
from sklearn import metrics


def main():
	for e in trange(mlp_args.epochs):
		train(e)
		evaluate(e)
		upload_models(e)


def avg(l):
	return sum(l) / len(l)


def tensorify(*document_items):
	for document_item in document_items:
		yield fewnerd_processor.pick_llm_output_for_document(
			device=device,
			input_tokens=mlp_args.input_tokens,
			llm_layer=mlp_args.llm_layer,
			is_fine_tune_llm=mlp_args.fine_tune_llm,
			documents=document_item if isinstance(document_item, list) else [document_item]
		)

def upload_models(epoch):
    global max_benchmark, current_benchmark, instances_model, instances_model_clearml
    # if math.isclose(current_benchmark, max_benchmark):
    #     print(f"recognized better benchmark value {max_benchmark}. Uploading model")
    # if epoch == 0:
	#     update_output_model(model_path=model_path)

    file_path = f'instances_model.pt'
    torch.save(instances_model.state_dict(), file_path)
    clearml_poc.upload_model_to_clearml(instances_model_clearml, file_path)
    clearml_poc.add_tags([str(instances_model_clearml.id)])


async def iterate_over_train(fine_type, similarity_strategy):
	async for anchor, same_type, different_type in fewnerd_processor.yield_train_dataset(
			anchor_type=fine_type,
			batch_size=mlp_args.batch_size,
			instances_per_type=mlp_args.instances_per_type,
			hard_negative_ratio=mlp_args.hard_negative_ratio,
			llm_layer=mlp_args.llm_layer,
			similarity_strategy=similarity_strategy
	):
		good_batch, bad_batch = tensorify(same_type, different_type)
		if similarity_strategy == 'instance':
			anchor = next(tensorify(anchor))
		yield anchor, good_batch, bad_batch


async def one_type_epoch_training(fine_type, epoch):
	results = defaultdict(list)
	async for anchor, good_batch, bad_batch in iterate_over_train(fine_type, similarity_strategy='instance'):
		instances_optimizer.zero_grad()
		types_optimizer.zero_grad()
		similarity = partial(compute_similarity_base, positive_examples=good_batch, negative_examples=bad_batch, epoch=epoch)

		for metric, value in similarity(instances_model, anchor=anchor).items():
			results[f'instances_{metric}'].extend(value)

		instances_optimizer.step()
		types_optimizer.step()

	async for anchor, good_batch, bad_batch in iterate_over_train(fine_type, similarity_strategy='type'):
		instances_optimizer.zero_grad()
		types_optimizer.zero_grad()
		similarity = partial(compute_similarity_base, positive_examples=good_batch, negative_examples=bad_batch, epoch=epoch)
		anchor_tensor = layer_to_tensor_train[fine_type]
		anchor_tensor = torch.concat((anchor_tensor, anchor_tensor), dim=-1)
		for metric, value in similarity(types_model, anchor=anchor_tensor).items():
			results[f'types_{metric}'].extend(value)

		instances_optimizer.step()
		types_optimizer.step()

	return results


def compute_similarity_base(
		model,
		anchor,
		positive_examples,
		negative_examples,
		epoch
):
	anchor = torch.atleast_2d(anchor)
	good_similarities = []
	bad_similarities = []
	losses = []
	anchor = model(anchor)
	positive_examples = instances_model(positive_examples)
	negative_examples = instances_model(negative_examples)
	loss = similarity_criterion(anchor, positive_examples, negative_examples, epoch=epoch)
	loss.backward()

	losses.append(loss.item())
	good_similarities.append(F.cosine_similarity(anchor, positive_examples, dim=-1).mean().item())
	bad_similarities.append(F.cosine_similarity(anchor, negative_examples, dim=-1).mean().item())

	return {
		"good_similarity": good_similarities,
		"bad_similarity": bad_similarities,
		"loss": losses
	}


def train(epoch):
	instances_model.train()
	types_model.train()

	results_so_far = defaultdict(list)

	all_train_types = fewnerd_processor.train_fine_types()
	tasks = [one_type_epoch_training(fine_type, epoch) for fine_type in all_train_types]
	loop = asyncio.get_event_loop()
	task_results = loop.run_until_complete(asyncio.gather(*tasks))

	for result in task_results:
		for key, value in result.items():
			results_so_far[key].extend(value)

	results = {metric: avg(value) for metric, value in results_so_far.items()}

	log_training_metrics(epoch, series="train", **results)


async def iterate_over_test(fine_type, similarity_strategy):
	async for anchor, same_type, different_type in fewnerd_processor.yield_test_dataset(
			anchor_type=fine_type,
			batch_size=mlp_args.batch_size,
			instances_per_type=mlp_args.instances_per_type,
			llm_layer=mlp_args.llm_layer,
			similarity_strategy=similarity_strategy
	):
		good_batch, bad_batch = tensorify(same_type, different_type)
		if similarity_strategy == 'instance':
			anchor = next(tensorify(anchor))
		yield anchor, good_batch, bad_batch


async def one_type_epoch_evaluation(fine_type, epoch):
	results = defaultdict(list)
	async for anchor, good_batch, bad_batch in iterate_over_test(fine_type, similarity_strategy='instance'):
		similarity = partial(compute_similarity_base, positive_examples=good_batch, negative_examples=bad_batch, epoch=epoch)
		prediction = partial(eval_predict, good_batch=good_batch, bad_batch=bad_batch)

		for metric, value in similarity(instances_model, anchor=anchor).items():
			results[f'instances_{metric}'].extend(value)

		for metric, value in prediction(instances_model, anchor=anchor).items():
			results[f'instances_{metric}'].extend(value)

	async for anchor, good_batch, bad_batch in iterate_over_test(fine_type, similarity_strategy='type'):
		similarity = partial(compute_similarity_base, positive_examples=good_batch, negative_examples=bad_batch, epoch=epoch)
		prediction = partial(eval_predict, good_batch=good_batch, bad_batch=bad_batch)

		anchor_tensor = layer_to_tensor_test[fine_type]
		anchor_tensor = torch.concat((anchor_tensor, anchor_tensor), dim=-1)

		for metric, value in similarity(types_model, anchor=anchor_tensor).items():
			results[f'types_{metric}'].extend(value)

		for metric, value in prediction(types_model, anchor=anchor_tensor).items():
			results[f'types_{metric}'].extend(value)


	return results


def eval_predict(model, anchor, good_batch, bad_batch):
	predictions, ground_truth = [], []
	with torch.no_grad():
		anchor_mlp = model(anchor)
		good_batch_mlp = instances_model(good_batch)
		bad_batch_mlp = instances_model(bad_batch)

	predictions.extend(torch.cosine_similarity(anchor_mlp, good_batch_mlp, dim=1).cpu().tolist())
	predictions.extend(torch.cosine_similarity(anchor_mlp, bad_batch_mlp, dim=1).cpu().tolist())
	ground_truth.extend([1] * len(good_batch_mlp))
	ground_truth.extend([0] * len(bad_batch_mlp))
	return {"predictions": predictions, "ground_truth": ground_truth}


def evaluate(epoch):
	instances_model.eval()
	types_model.eval()

	results_so_far = defaultdict(list)
	auc_threshold_graph = defaultdict(list)

	all_test_types = fewnerd_processor.test_fine_types()
	tasks = [one_type_epoch_evaluation(fine_type, epoch) for fine_type in all_test_types]
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
	results_so_far = {metric: avg(value) for metric, value in results_so_far.items()}

	results = {
		**results_so_far,
		"types_auc": types_auc,
		"instances_auc": instances_auc,
	}
	log_training_metrics(
		epoch,
		series="eval",
		**results
	)

	index_column = auc_threshold_graph.pop("type")
	clearml_poc.add_table(
		title="per fine type",
		series="eval",
		iteration=epoch,
		table=pd.DataFrame(data=auc_threshold_graph, index=index_column)
	)

	global max_benchmark, current_benchmark
	current_benchmark = types_auc
	max_benchmark = max(current_benchmark, max_benchmark)


def log_training_metrics(index, series, **kwargs):
	for metric, value in kwargs.items():
		if value:
			clearml_poc.add_point_to_graph(title=metric, series=series, x=index, y=value)


if __name__ == "__main__":
	clearml_poc.clearml_init(queue_name='dsicsgpu')
	max_benchmark = 0
	current_benchmark = 0
	assert torch.cuda.is_available(), "no gpu available"
	mlp_args = Arguments()
	llm_args = FineTuneLLM()
	clearml_poc.clearml_connect_hyperparams(mlp_args, "mlp_args")
	clearml_poc.clearml_connect_hyperparams(llm_args, "llm_args")
	print('args are: ', json.dumps(mlp_args.__dict__, indent=4))
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model = ContrastiveMLP(mlp_args).to(device).double()

	similarity_criterion = ContrastiveLoss(loss_fn=mlp_args.loss_fn, margin=mlp_args.triplet_loss_margin)

	instances_model_clearml = clearml_poc.generate_tracked_model(name="instances_model",
	                                                             framework="PyTorch",
	                                                             config_dict=dataclasses.asdict(mlp_args))



	instances_model = model
	types_model = model


	instances_optimizer = torch.optim.Adam(instances_model.parameters(), lr=mlp_args.lr)
	types_optimizer = torch.optim.Adam(types_model.parameters(), lr=mlp_args.lr)

	type_to_tensor = fewnerd_processor.load_entity_name_embeddings(
		layer_name=fewnerd_dataset.llm_and_layer_to_elastic_name(
			llm_id=llm_args.llm_id,
			layer=llm_args.layer
		),
		entity_name_strategy=mlp_args.entity_name_embeddings
	)

	layer_to_tensor_test = {entity_type: tensor.to(device) for entity_type, tensor in type_to_tensor.items() if
	                        entity_type in fewnerd_processor.test_fine_types()}
	layer_to_tensor_train = {type: tensor.to(device) for type, tensor in type_to_tensor.items() if
	                         type in fewnerd_processor.train_fine_types()}

	main()
