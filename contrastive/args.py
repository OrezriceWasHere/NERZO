from dataclasses import dataclass, field


@dataclass
class Arguments:
    lr: float = 5e-6

    contrastive_mlp_sizes: list[int] = field(default_factory=lambda:[1024, 50, 100])
    is_hidden_layer = True
    batch_size: int = 50
    instances_per_type: int = 100


    input_tokens = "start_end_pair"  # [possible_values: "end", "start_end_pair", "diff"]
    fine_tune_llm: bool = False

    triplet_loss_margin: float = 0.5
    activation: str = "silu"
    noise: str = "dropout"
    dropout: float = 0.2
    epochs: int = 35
    loss_fn: str = "dpr_loss" # possible_values: "triplet_loss", "dpr_loss"

    compute_queue: str = "gpu"

def optuna_suggest_hyperparams(trial) -> Arguments:
    # Create an instance of the HyperParams class.
    args = Arguments()

    args.input_layer = 1024

    args.input_methods = trial.suggest_categorical('input_methods', ['start_end_pair', 'end', 'diff'])
    if args.input_methods == 'start_end_pair':
        args.input_layer = 1024 * 2

    output_size = trial.suggest_int('output_size', 20, 500)


    args.is_hidden_layer = trial.suggest_categorical('is_hidden_layer', [True, False])
    if args.is_hidden_layer:
        hidden_size = trial.suggest_int('hidden_size', 20, 500)
        args.contrastive_mlp_sizes = [args.input_layer,hidden_size, output_size]
    else:
        args.contrastive_mlp_sizes = [args.input_layer, output_size]

    args.lr = trial.suggest_float('lr', 1e-6, 1e-4, log=True)
    args.dropout = trial.suggest_categorical('dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5])
    args.activation = trial.suggest_categorical('activation', ['silu', 'leaky_relu', 'relu'])
    args.noise = trial.suggest_categorical('noise', ['dropout', 'identity'])
    args.loss_fn = trial.suggest_categorical('loss_fn', ['triplet_loss', 'dpr_loss'])

    args.fine_tune_llm = trial.suggest_categorical('fine_tune_llm', [True, False])


    return args

