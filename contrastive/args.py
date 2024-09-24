from clearml.task_parameters import TaskParameters


class Arguments(TaskParameters):
    lr: float = 5e-6
    llm_id: str = "meta-llama/Meta-Llama-3.1-8B"
    llm_hidden_layer: str = "model.layers.17.self_attn.v_proj"
    contrastive_mlp_sizes: list[int] = [1024, 10, 50]
    batch_size: int = 20
    activation: str = "relu"
    noise: str = "dropout"
    dropout: float = 0.1
    epochs: int = 200
    loss_fn: str = "triplet_loss"
    # In contrastive, how many items to compare
    number_of_examples: int = 20

