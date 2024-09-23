

class Arguments:
    lr: float = 1e-5
    llm_id: str = "meta-llama/Meta-Llama-3.1-8B"
    llm_hidden_layer: str = "model.layers.17.self_attn.v_proj"
    contrastive_mlp_sizes: list[int] = [1024, 4096, 50]
    accumulation_steps: int = 5
