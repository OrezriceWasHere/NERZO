from sklearn import metrics
from clearml import Logger
import chat_gpt_client
import asyncio
import random
import dataset_provider
import entity_classifier




async def handle_type(entity_type: str):
    documents_with_type = dataset_provider.get_by_entity_type(entity_type)["hits"]["hits"]
    expected_responses = ['yes' for _ in documents_with_type]
    received_response = []
    for index, dataset_document in enumerate(documents_with_type):
        document = dataset_document["_source"]
        entities = entity_classifier.pick_entities_of_type(document, entity_type)
        chosen_entity = random.choice(entities)
        prompt = entity_classifier.build_classification_prompt(entity_type=entity_type,
                                                               entity_text=chosen_entity["entity_in_text"],
                                                               sentence=document["content"])
        print(prompt)
        chat_gpt_response = chat_gpt_client.send_message(prompt)
        parsed_response = chat_gpt_client.parse_answer_from_response(chat_gpt_response)
        received_response.append(parsed_response[0].lower())
        if len(parsed_response) != 1:
            print(f'more than one ansewer in {parsed_response}. taking first answer {parsed_response[0]}')
        print('chat gpt respone ', parsed_response)
        print('   ')
        if expected_responses[index] != parsed_response[0].lower():
            print(f'expected {expected_responses[index]} but got {parsed_response[0]}')
    return expected_responses, received_response


async def main():
    intersting_types= [
        "Butterfly",
        "District_of_London"
    ]
    for type in intersting_types:
        expected_responses, received_response = await handle_type(type)
        confusion_matrix(expected_responses, received_response, type)
    pass


def confusion_matrix(expected_responses, received_responses, entity_type, iteration=1):
    #  clearml display confusion matrix
    confusion = metrics.confusion_matrix(expected_responses, received_responses)
    Logger.current_logger().report_confusion_matrix(
        f"confusion metrix for entity type {entity_type}",
        "ignored",
        iteration=iteration,
        matrix=confusion,
        xaxis="answer expected",
        yaxis="answer received",
    )

if __name__ == "__main__":
    # Start async loop
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
