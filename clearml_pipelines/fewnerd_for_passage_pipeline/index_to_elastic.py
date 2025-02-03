import asyncio
import os

import ijson
from clearml import Dataset
from tqdm import tqdm

import clearml_poc
import dataset_provider
from clearml_pipelines.fewnerd_pipeline import fewnerd_dataset
from contrastive.args import FineTuneLLM


def schema_llm_layer(size):
    return {
        f'embedding.{llm_id}': {
            "type": "dense_vector",
            "dims": size,
            "index": "true",
            "similarity": "cosine",
            "index_options": {
                "type": "flat"
            }
        }
    }


async def index_task(dataset):
    env = dataset["env"]
    index_name = f'fewnerd_full_sentence_{env}'
    # Load the artifact
    dataset_dir = Dataset.get(dataset_name=f'entire_setence_{dataset["json"]}',
                              dataset_tags=[llm_id],
                              dataset_project="fewnerd_pipeline").get_local_copy()

    first_time = True

    with open(os.path.join(dataset_dir, dataset["json"])) as file:
        for document in tqdm(ijson.items(file, "item")):
            if first_time:
                llm_layer_size = len(document["embedding"][llm_id])
                elastic_mapping = fewnerd_dataset.elasticsearch_mapping_for_full_sentence
                field_mapping = schema_llm_layer(llm_layer_size)
                await dataset_provider.ensure_existing_index(index_name, elastic_mapping)
                await dataset_provider.ensure_field(index_name, field_mapping)
                first_time = False
            await dataset_provider.upsert(document, document["text_id"], index_name)


async def main():
    tasks = [index_task(dataset) for dataset in fewnerd_dataset.datasets]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    llm_args = FineTuneLLM()
    clearml_poc.clearml_init(
        project_name="fewnerd_pipeline",
        task_name="index to db"
    )
    clearml_poc.clearml_connect_hyperparams(llm_args, "llm_args")
    llm_id = llm_args.llm_id

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
