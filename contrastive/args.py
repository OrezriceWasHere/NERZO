from dataclasses import dataclass, fields
from typing import Optional,Type, Any


@dataclass
class Arguments:
    lr: float = 2e-6

    input_layer: int = 1024
    hidden_layer: int = 500
    output_layer: int = 500

    is_hidden_layer: bool = True
    batch_size: int = 50
    instances_per_type: int = 50

    llm_layer: str = "llama_3_17_v_proj"

    input_tokens: str = "start_eos_pair"  # [possible_values: "end", "start_end_pair", "diff","start_eos_pair"]
    entity_name_embeddings: str = "end_eos"

    fine_tune_llm: bool = False
    disable_similarity_training: bool = False

    hard_negative_ratio: float = 0.1

    triplet_loss_margin: float = 0.5
    activation: str = "silu"
    noise: str = "dropout"
    dropout: float = 0.1
    epochs: int = 50
    enable_gate: bool = True
    loss_fn: str = "triplet_loss" # possible_values: "triplet_loss", "dpr_loss", "contrastive_loss", triplet_contrastive_loss

def convert_value(value: Any, target_type: Type) -> Any:
    """Converts a value to the specified target type with special handling for bool."""
    if target_type is bool:
        # Handle string representations of booleans
        if isinstance(value, str):
            return value.lower() in ["true", "1", "yes"]
        return bool(value)
    return target_type(value)

def dataclass_decoder(dct: dict, cls: Type[Any]) -> Any:
    """Custom decoder to convert a dictionary to a dataclass instance."""
    field_types = {field.name: field.type for field in fields(cls)}
    return cls(**{key: convert_value(value, field_types[key]) for key, value in dct.items() if key in field_types})

# @dataclass
# class FineTuneLLM:
#     llm_id: str = 'intfloat/e5-mistral-7b-instruct'
#     layer: str = "model.layers.17.self_attn.v_proj"
#     mlp_head_model_id_from_clearml: str = "a18145c711b046cbb9fbb86e38ac3e47"
#     max_llm_layer: Optional[int] = 18

@dataclass
class FineTuneLLM:
    llm_id: str = "meta-llama/Meta-Llama-3.1-8B"
    layer: str = "model.layers.17.self_attn.v_proj"
    mlp_head_model_id_from_clearml: str = "df186502700540649bfc012cb7f0a3a6"
    max_llm_layer: Optional[int] = 18