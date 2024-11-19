from clearml.task_parameters import TaskParameters


class Arguments(TaskParameters):
    lr: float = 5e-6

    contrastive_mlp_sizes: list[int] = [1024, 50, 100]
    batch_size: int = 15
    instances_per_type: int = 100

    triplet_loss_margin: float = 1.0
    activation: str = "silu"
    noise: str = "dropout"
    dropout: float = 0.2
    epochs: int = 200
    loss_fn: str = "triplet_loss"

