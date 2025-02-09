import asyncio
import torch
from tqdm import trange
import clearml_poc
import dataset_provider
from clearml_pipelines.fewnerd_pipeline import fewnerd_dataset
from contrastive import fewnerd_processor
from contrastive.args import Arguments, FineTuneLLM
from contrastive.mlp import ContrastiveMLP


def main():
    for e in trange(mlp_args.epochs):
        train(e)
        evaluate(e)


def train(epoch):
    mlp_types.train()
    mlp_instances.train()





def evaluate(epoch):
    mlp_types.eval()
    mlp_instances.eval()


if __name__ == "__main__":
    clearml_poc.clearml_init(
        task_name="contrastive ner zero shot",
    )
    assert torch.cuda.is_available(), "no gpu available"
    device = torch.device("cuda:0")

    mlp_args = Arguments()
    llm_args = FineTuneLLM()
    clearml_poc.clearml_connect_hyperparams(mlp_args, "mlp_args")
    clearml_poc.clearml_connect_hyperparams(mlp_args, "llm_args")

    mlp_types = ContrastiveMLP(mlp_args)
    mlp_instances = ContrastiveMLP(mlp_args)
    optimizer = torch.optim.Adam(list(mlp_instances.parameters()) + list(mlp_instances.parameters()), lr=mlp_args.lr)

    type_to_tensor = fewnerd_processor.load_entity_name_embeddings(
        layer_name=fewnerd_dataset.llm_and_layer_to_elastic_name(llm_id=llm_args.llm_id, layer=llm_args.layer),
        entity_name_strategy=mlp_args.entity_name_embeddings
    )

    layer_to_tensor_test = {type:tensor for type, tensor in type_to_tensor.items() if type in fewnerd_processor.test_fine_types()}
    layer_to_tensor_train = {type:tensor for type, tensor in type_to_tensor.items() if type in fewnerd_processor.train_fine_types()}


    loop = asyncio.get_event_loop()
    main()
    loop.close()
