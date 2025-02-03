import json
import hashlib
import torch
from tqdm import tqdm
from clearml import StorageManager, Dataset
import clearml_poc
from clearml_pipelines.fewnerd_pipeline import fewnerd_dataset
from contrastive.args import Arguments, FineTuneLLM
from sentence_embedder import SentenceEmbedder


def split_into_document(dataset_file):
    pbar = tqdm()
    with open(dataset_file, "r") as f:
        # For document in file
        while True:
            pbar.update(1)
            line_buffer = []
            while True:
                line = f.readline()
                if not line or line == "\n":
                    break
                line_buffer.append(line.replace("\n", ""))

            if not line_buffer:
                break

            yield line_buffer


def space_when_necessary(prev_word, current_word):
    space = " "
    no_space = ""
    list_words_without_space = ["(", ")", "[", "]", "{", "}", ":", ";", ",", ".", "!", "?", "'", "\"", "`", "'s", "''",
                                "%"]

    if any((not prev_word,
            not current_word,
            current_word in list_words_without_space)):
        return no_space
    return space


def decide_word_tagging(tagging):
    if tagging == "O":
        return "O", "O"
    return tagging.split("-")

def create_embedding(text):
    embedding = llm.forward_passage(text).tolist()

    embeddings = {
        llm_id: embedding
    }

    return embeddings


def process_document(document):
    prev_word = None
    prev_tagging = None
    full_text = ""
    tagging_array = []
    for line in document:
        word, tagging = line.split("\t")
        coarse_tag, fine_tag = decide_word_tagging(tagging)
        addition = space_when_necessary(prev_word, word) + word

        not_yet_entity = (prev_tagging == "O" or prev_tagging is None) and tagging == "O"
        start_entity = (prev_tagging != tagging and tagging != "O") or (prev_tagging == "O" and tagging != "O")
        in_entity = prev_tagging == tagging and tagging != "O"
        end_entity = prev_tagging != "O" and tagging == "O"

        if not_yet_entity or end_entity:
            pass
        elif start_entity:
            tagging_array.append({
                "phrase": word,
                "coarse_type": coarse_tag,
                "fine_type": fine_tag,
                "index_start": len(full_text) + (1 if prev_word and len(word) != len(addition) else 0),
                "index_end": len(full_text) + len(addition),
            })
        elif in_entity:
            tagging_array[-1]["phrase"] += addition
            tagging_array[-1]["index_end"] += len(addition)

        full_text += addition
        prev_tagging = tagging
        prev_word = word

    text_id =  str(hashlib.sha1(str.encode(full_text)).hexdigest())

    tagging_object = {
        "all_text": full_text,
        "text_id": text_id,
        "tagging": tagging_array,
        "embedding": create_embedding(full_text)
    }

    return tagging_object


def process_dataset(dataset_url):
    dataset_file = StorageManager.get_local_copy(remote_url=dataset_url)
    documents = split_into_document(dataset_file)
    processed_documents = []
    for document in tqdm(documents):
        processed_document = process_document(document)
        processed_documents.append(processed_document)
    return processed_documents


def main_process(dataset):
    file_dir = dataset["json"]
    with open(file_dir, "w") as file:
        processed_documents = process_dataset(dataset["url"])
        tags = [llm_id] + [dataset["env"]]
        clearml_poc.add_tags(tags)

        file.write(json.dumps(processed_documents))
        clearml_dataset = Dataset.create(dataset_name=f'entire_setence_{file_dir}', dataset_project="fewnerd_pipeline")
        clearml_dataset.add_files(path=file_dir)
        clearml_dataset.add_tags(tags)

        # Dataset is uploaded to the ClearML Server by default
        clearml_dataset.upload()
        clearml_dataset.finalize()


if __name__ == "__main__":

    clearml_poc.clearml_init(project_name="fewnerd_pipeline",
                             task_name="prepare dataset",
                             requirements=["sentence_transformers", "datasets", "einops"])

    args = Arguments()
    clearml_poc.clearml_connect_hyperparams(args, "general")
    llm_args = FineTuneLLM()
    clearml_poc.clearml_connect_hyperparams(llm_args, "llm_args")

    llm_id = llm_args.llm_id
    llm = SentenceEmbedder(llm_id=llm_id,
                           max_llm_layer=llm_args.max_llm_layer,
                           interested_layers=llm_args.layer
                           )
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    for dataset in fewnerd_dataset.datasets:
        main_process(dataset)