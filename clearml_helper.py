import torch
from clearml import Task, Model
from contrastive.args import FineTuneLLM, Arguments, dataclass_decoder
from contrastive.mlp import ContrastiveMLP

def get_mlp_by_id(mlp_id: str, device=None) -> ContrastiveMLP:
	if device is None:
		device = torch.device("cpu")
		print(f"Using device: {device}")
	model = Model(mlp_id)
	assert model
	task = Task.get_task(model.task)
	print(f'generating mlp from task:\t {task.data.name} with id\t {model.id}.\n task id {task.id}.\nlink to task {task.get_output_log_web_page()}')
	args_of_task = task.get_parameters(cast=False)
	args_dict = {key.replace("general/", ""): value for key, value in args_of_task.items() if "general" in key}
	if not args_dict:
		args_dict = {key.replace("mlp_args/", ""): value for key, value in args_of_task.items() if "mlp_args" in key}
	args: Arguments = dataclass_decoder(dct=args_dict, cls=Arguments)
	local_mlp_head_path = model.get_local_copy(raise_on_error=True)

	similarity_model = ContrastiveMLP(args).to(device)
	assert local_mlp_head_path, "could not download mlp head model"
	local_mlp_head_model = torch.load(
		local_mlp_head_path,
		weights_only=True,
		map_location=device
		)
	similarity_model.load_state_dict(local_mlp_head_model)
	return similarity_model

def get_args_by_mlp_id(mlp_id: str) -> Arguments:
	model = Model(mlp_id)
	args_of_task = Task.get_task(model.task).get_parameters(cast=False)
	args_dict = {key.replace("general/", ""): value for key, value in args_of_task.items() if "general" in key}
	args: Arguments = dataclass_decoder(dct=args_dict, cls=Arguments)
	return args