import asyncio

import torch
import clearml_poc
from contrastive.args import Arguments
from contrastive.mlp import ContrastiveMLP


def main():
    pass

if __name__ == "__main__":
    clearml_poc.clearml_init()
    assert torch.cuda.is_available(), "no gpu available"
    device = torch.device("cuda:0")

    mlp_args = Arguments()
    clearml_poc.clearml_connect_hyperparams(mlp_args, "mlp_args")
    mlp_types = ContrastiveMLP(mlp_args)
    mlp_instances = ContrastiveMLP(mlp_args)
    optimizer = torch.optim.Adam(list(mlp_instances.parameters()) + list(mlp_instances.parameters()), lr=mlp_args.lr)


    loop = asyncio.get_event_loop()
    main()
    loop.close()
