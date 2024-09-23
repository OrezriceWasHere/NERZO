

class Arguments:
    lr: float = 1e-5
    llm_id: str = "meta-llama/Meta-Llama-3.1-8B"
    llm_hidden_layer: str = "model.layers.17.self_attn.v_proj"
    contrastive_mlp_sizes: list[int] = [1024, 4096, 50]
    batch_size: int = 200
    activation: str = "silu"
    noise: str = "dropout"
    dropout: float = 0.1
    epochs: int = 100
    loss_fn: str = "triplet_loss"
    # In contrastive, how many items to compare
    number_of_examples: int = 20

