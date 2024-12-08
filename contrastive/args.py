from dataclasses import dataclass, field


@dataclass
class Arguments:
    lr: float = 5e-6

    input_layer: int = 1024
    hidden_layer: int = 50
    output_layer: int = 100

    is_hidden_layer: bool = True
    batch_size: int = 50
    instances_per_type: int = 100

    llm_layer: str = "llama_3_17_v_proj"

    input_tokens: str = "start_end_pair"  # [possible_values: "end", "start_end_pair", "diff"]
    fine_tune_llm: bool = True

    triplet_loss_margin: float = 0.5
    activation: str = "silu"
    noise: str = "dropout"
    dropout: float = 0.2
    epochs: int = 35
    loss_fn: str = "dpr_loss" # possible_values: "triplet_loss", "dpr_loss"

    compute_queue: str = "a100_gpu"
