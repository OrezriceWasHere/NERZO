import json
from uuid import uuid4
import torch
from clearml import Task
from tqdm import tqdm
from clearml import StorageManager, Dataset
from clearml_pipelines.fewnerd_pipeline import fewnerd_dataset
from contrastive.args import FineTuneLLM, Arguments
from llm_interface import LLMInterface

# Connecting ClearML with the current process,
# from here on everything is logged automatically
Task.add_requirements("-rrequirements.txt")
task = Task.init(project_name="fewnerd_pipeline", task_name="Pipeline step 2 jsonify dataset", reuse_last_task_id=False)
task.execute_remotely()


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

def create_embedding(text, indices):
    tokens = llm.tokenize(text).to(device)
    llm_indices = llm.token_indices_given_text_indices(text, indices)
    embeddings = {}
    for model_layer, db_name  in layers_and_keys_pairs:
        h = llm.get_llm_at_layer(tokens, model_layer)[0]
        start = h[llm_indices[0] - 1]
        end = h[llm_indices[1]]
        embeddings[db_name] = {
                "start": start.tolist(),
                "end": end.tolist()
       }

    return embeddings


def process_document(document):
    prev_word = None
    prev_tagging = None
    full_text = ""
    tagging_array = []
    text_id = str(uuid4())
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

    for tagging in tagging_array:
        tagging["all_text"] = full_text
        tagging["text_id"] = text_id
        assert tagging["all_text"][tagging["index_start"]:tagging["index_end"]] == tagging["phrase"]
        embedding = create_embedding(full_text, (tagging["index_start"], tagging["index_end"]))
        tagging["embedding"] = embedding

    return tagging_array


def process_dataset(dataset_url):
    dataset_file = StorageManager.get_local_copy(remote_url=dataset_url)
    documents = split_into_document(dataset_file)
    processed_documents = []
    for document in tqdm(documents):
        processed_document = process_document(document)
        processed_documents.extend(processed_document)
    return processed_documents


def main_process(dataset):
    file_dir = dataset["json"]
    with open(file_dir, "w") as file:
        processed_documents = process_dataset(dataset["url"])
        tags = db_layer_name + [dataset["env"]]
        task.add_tags(tags)

        file.write(json.dumps(processed_documents))
        clearml_dataset = Dataset.create(dataset_name=file_dir, dataset_project="fewnerd_pipeline")
        clearml_dataset.add_files(path=file_dir)
        clearml_dataset.add_tags(tags)

        # Dataset is uploaded to the ClearML Server by default
        clearml_dataset.upload()
        clearml_dataset.finalize()



if __name__ == "__main__":
    args = Arguments()
    fine_tune_llm = FineTuneLLM()

    task.connect(args)
    task.connect(fine_tune_llm, name="fine_tune_llm")

    model_layer_name = fine_tune_llm.layer
    db_layer_name = args.llm_layer

    layers_and_keys_pairs = list(zip([model_layer_name], [db_layer_name]))
    llm = LLMInterface(llm_id=fine_tune_llm.llm_id,
                       interested_layers=[db_layer_name],
                       max_llm_layer=fine_tune_llm.max_llm_layer)
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    for dataset in fewnerd_dataset.datasets:
        main_process(dataset)



print('Notice, artifacts are uploaded in the background')
print('Done')
