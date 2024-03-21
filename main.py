import asyncio
import random
import dataset_provider
import entity_classifier


async def handle_type(entity_type: str):
    documents_with_type = dataset_provider.get_by_entity_type(entity_type)["hits"]["hits"]
    for dataset_document in documents_with_type:
        document = dataset_document["_source"]
        entities = entity_classifier.pick_entities_of_type(document, entity_type)
        chosen_entity = random.choice(entities)
        prompt = entity_classifier.build_prompt(entity_type=entity_type,
                                                entity_text=chosen_entity["entity_in_text"],
                                                sentence=document["content"])
        print(prompt)
        print("*******************")


async def main():
    intersting_types= [
        "Butterfly",
        "District_of_London"
    ]
    for type in intersting_types:
        await handle_type(type)
    pass


if __name__ == "__main__":
    # Start async loop
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
