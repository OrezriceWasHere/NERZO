from elasticsearch import AsyncElasticsearch
import asyncio
import uuid
from tqdm import tqdm


def create_document(lines):
    if not lines:
        return None
    generated_id = str(uuid.uuid4())


    word_and_label = [line.replace("\n", "").split("\t") for line in lines ]
    full_text = " ".join([word_label[0] for word_label in word_and_label])

    # filter "O" labels
    current_label = word_and_label[0][1]
    coarse_type, fine_type = current_label.split("-") if current_label != "O" else ("O", "O")
    tagging_array = [{
        "phrase": "",
        "coarse_type": coarse_type,
        "fine_type": fine_type,
        "offset_in_word": 0,
        "text_id": generated_id
    }]

    for (word, label) in word_and_label:
        if label == current_label:
            tagging_array[-1]["phrase"] += " " + word
        else:
            current_label = label
            coarse_type, fine_type = current_label.split("-") if current_label != "O" else ("O", "O")
            tagging_array.append({
                "phrase": word,
                "coarse_type": coarse_type,
                "fine_type": fine_type,
                "offset_in_word": len(tagging_array),
                "text_id": generated_id
            })


    return {
        "text_id": generated_id,
        "full_text": full_text,
        "tagging": tagging_array
    }
def group_fewnerd_together(document):
    if not document:
        return None
    current_coarse_type, current_fine_type = document["tagging"][0]["coarse_type"], document["tagging"][0]["fine_type"]
    new_document = {
        "text_id": document["text_id"],
        "full_text": document["full_text"],
        "tagging": [{"phrase": "",
                     "coarse_type": current_coarse_type,
                     "fine_type": current_fine_type,
                     "offset_in_word": 0,
                     "text_id": document["text_id"]}]
    }
    for tagging in document["tagging"]:
        if tagging["fine_type"] == current_fine_type:
            new_document["tagging"][-1]["phrase"] += " " + tagging["phrase"]
        else:
            current_fine_type = tagging["fine_type"]
            current_coarse_type = tagging["coarse_type"]
            new_document["tagging"].append({"phrase": tagging["phrase"],
                                            "coarse_type": current_coarse_type,
                                            "fine_type": current_fine_type,
                                            "offset_in_word": len(new_document["tagging"]),
                                            "text_id": document["text_id"]})

    return new_document

async def index_documents(file_name, es: AsyncElasticsearch):
    with open(file_name, "r") as f:
        pbar = tqdm()

        # For document in file
        while True:
            pbar.update(1)
            line_buffer = []
            while True:
                line = f.readline()
                if line == "\n":
                    break
                line_buffer.append(line)

            if not line_buffer:
                break
            docuemnt = create_document(line_buffer)
            if docuemnt:
                groupped_fewnerd = group_fewnerd_together(docuemnt)
                if groupped_fewnerd:
                    await index_document(groupped_fewnerd, es)


async def index_document(docuemnt, es: AsyncElasticsearch):
    x = await es.index(
        index="fewnerd_dev",
        body=docuemnt,
        id=docuemnt["text_id"],
    )
    pass


if __name__ == "__main__":
    """"create async function for indexing documents"""
    es = AsyncElasticsearch(
        hosts=["https://*****:9200"],
        basic_auth=("****", "*****"),
        verify_certs=False,
        http_compress=False
    )

    loop = asyncio.get_event_loop()
    asyncio.ensure_future(index_documents("/home/orsh/data/fewnerd/supervised/dev.txt", es))
    loop.run_forever()
    loop.close()
