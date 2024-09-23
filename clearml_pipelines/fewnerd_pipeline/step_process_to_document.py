import json
from uuid import uuid4
from clearml import Task, StorageManager
from tqdm import tqdm
from clearml import StorageManager, Dataset

from llm_interface import LLMInterface


# Connecting ClearML with the current process,
# from here on everything is logged automatically
Task.add_requirements("-rrequirements.txt")
task = Task.init(project_name="fewnerd_pipeline", task_name="Pipeline step 2 jsonify dataset", reuse_last_task_id=False)
task.execute_remotely()

llm_id, layer = "meta-llama/Meta-Llama-3.1-8B", "model.layers.17.self_attn.v_proj"
db_key = "llama_3_17_v_proj"
llm = LLMInterface(llm_id=llm_id, interested_layers=[layer])


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
    list_words_without_space = ["(", ")", "[", "]", "{", "}", ":", ";", ",", ".", "!", "?", "'", "\"", "`", "'s","''", "%"]

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
    tokens = llm.tokenize(text)
    llm_indices = llm.token_indices_given_text_indices(text, indices)
    h = llm.get_llm_at_layer(tokens, layer)[0]
    start = h[llm_indices[0]]
    end = h[llm_indices[1]]
    return start, end

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
                "index_end": len(full_text)  +  len(addition),
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
        start, end = create_embedding(full_text, (tagging["index_start"], tagging["index_end"]))
        tagging["embedding"] = {
            db_key: {
                "start": start.tolist(),
                "end": end.tolist()
            }
        }

    return tagging_array


def process_dataset(dataset_url):
    dataset_file = StorageManager.get_local_copy(remote_url=dataset_url)
    documents = split_into_document(dataset_file)
    processed_documents = []
    for document in tqdm(documents):
        processed_document = process_document(document)
        processed_documents.extend(processed_document)
    return processed_documents


for dataset in tqdm(fewnerd_dataset.datasets):
    file_dir = dataset["json"]
    with open(file_dir, "w") as file:
        processed_documents = process_dataset(dataset["url"])

        file.write(json.dumps(processed_documents))
        clearml_dataset = Dataset.create(dataset_name=file_dir, dataset_project="fewnerd_pipeline")
        clearml_dataset.add_files(path=file_dir)

        # Dataset is uploaded to the ClearML Server by default
        clearml_dataset.upload()
        clearml_dataset.finalize()


    # task.upload_artifact(f'json-dataset-{dataset["env"]}',
    #                      artifact_object=file_dir)



print('Notice, artifacts are uploaded in the background')
print('Done')