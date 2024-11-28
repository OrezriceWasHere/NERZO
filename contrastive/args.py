from dataclasses import dataclass, field


@dataclass
class Arguments:
    lr: float = 5e-6

    contrastive_mlp_sizes: list[int] = field(default_factory=lambda:[1024, 50, 100])
    batch_size: int = 10
    instances_per_type: int = 100

    input_tokens = "start_end_pair"  # [possible_values: "end", "start_end_pair", "diff"]
    fine_tune_llm: bool = True

    triplet_loss_margin: float = 0.5
    activation: str = "silu"
    noise: str = "dropout"
    dropout: float = 0.2
    epochs: int = 200
    loss_fn: str = "triplet_loss" # possible_values: "triplet_loss", "dpr_loss"

    compute_queue: str = "runai_gpu"

