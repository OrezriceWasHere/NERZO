from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Arguments:
    lr: float = 1e-6

    input_layer: int = 1024
    hidden_layer: int = 100
    output_layer: int = 250

    is_hidden_layer: bool = False
    batch_size: int = 20
    instances_per_type: int = 300

    llm_layer: str = "llama_3_3_13_k_proj"

    input_tokens: str = "end"  # [possible_values: "end", "start_end_pair", "diff"]
    fine_tune_llm: bool = False

    triplet_loss_margin: float = 0.5
    activation: str = "silu"
    noise: str = "identity"
    dropout: float = 0
    epochs: int = 150
    enable_gate: bool = True
    loss_fn: str = "triplet_loss" # possible_values: "triplet_loss", "dpr_loss", "contrastive_loss", triplet_contrastive_loss

@dataclass
class FineTuneLLM:
    llm_id: str = "meta-llama/Llama-3.3-70B-Instruct"
    layer: str = "model.layers.13.self_attn.k_proj"
    mlp_head_model_id_from_clearml: str = "df186502700540649bfc012cb7f0a3a6"
    max_llm_layer: Optional[int] = 14