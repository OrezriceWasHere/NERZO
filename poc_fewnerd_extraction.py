from sklearn import metrics
from clearml import Logger
import chat_gpt_client
import asyncio
import random
import dataset_provider
import entity_classifier
import clearml_poc
import re

NUMBER_DOT_AT_START_REGEX = r'^\d+\.'


def parse_answer_item(answer_item: str) -> str:
    x = answer_item \
        .replace("- ", "") \
        .replace('``', '') \
        .replace("''", '') \
        .replace('"', '') \
        .replace("  ", " ")

    x = re.sub(NUMBER_DOT_AT_START_REGEX, '', x)

    if len(x) > 1 and x[0] == x[-1] == '"':
        x = x[1:-1]
    x = x.strip()
    return x


def parse_extracted_entities(response: str) -> list[str]:
    if response == "none":
        return []
    return [parse_answer_item(item) for item in response.split("\n")]


def send_chat_gpt(prompt) -> str:
    print("prompt" + prompt)
    chat_gpt_response = chat_gpt_client.send_message(prompt)
    parsed_response = chat_gpt_client.parse_answer_from_response(chat_gpt_response)
    result = parsed_response[0].lower()
    if len(parsed_response) != 1:
        print(f'more than one ansewer in {parsed_response}. taking first answer {parsed_response[0]}')
    print('chat gpt respone ', parsed_response)
    print('   ')

    # Remove answer prefix
    answer_prefix = "answer:"
    if result.find(answer_prefix) > 0:
        result = result[len(answer_prefix) + result.find(answer_prefix):]

    return result


async def handle_coarse_type(entity_type: str):
    documents_with_type = dataset_provider.get_by_coarse_grained_type_fewnerd(entity_type)["hits"]["hits"]
    expected_responses = []
    received_response = []
    entity_type_no_backslash = " or ".join(entity_type.split("/"))
    for index, dataset_document in enumerate(documents_with_type):

        document = dataset_document["_source"]

        # For entities within document, extract all entities of the given type
        all_with_entity = entity_classifier.pick_entities_of_coarsetype_fewnerd(document["tagging"], entity_type)
        ground_truth = list(map(lambda x: parse_answer_item(x["phrase"].lower()), all_with_entity))

        prompt = entity_classifier.build_extraction_prompt(entity_type_no_backslash, document["full_text"])
        chat_gpt_result = send_chat_gpt(prompt)
        parsed_response = parse_extracted_entities(chat_gpt_result)

        yes_ground_truth = ["yes" for _ in range(len(ground_truth))]
        yes_predictions = ["yes" if entity in parsed_response else "no" for entity in ground_truth]

        if yes_ground_truth != yes_predictions:
            print(f'expected {ground_truth} but got {parsed_response}.\nprompt is: {prompt}')

        # For entities not in document, extract all entities of the given type
        all_without_entity = list(filter(lambda x: x not in all_with_entity, dataset_document["_source"]["tagging"]))
        ground_truth = list(map(lambda x: parse_answer_item(x["phrase"].lower()), all_without_entity))

        prompt = entity_classifier.build_extraction_prompt(entity_type_no_backslash, document["full_text"])
        chat_gpt_result = send_chat_gpt(prompt)
        parsed_response = parse_extracted_entities(chat_gpt_result)

        no_ground_truth = ["no" for _ in range(len(ground_truth))]
        no_predictions = ["yes" if entity in parsed_response else "no" for entity in ground_truth]

        if no_ground_truth != no_predictions:
            print(f'expected {ground_truth} but got {parsed_response}.\nprompt is: {prompt}')

        expected_responses.extend(yes_ground_truth)
        expected_responses.extend(no_ground_truth)
        received_response.extend(yes_predictions)
        received_response.extend(no_predictions)

    return expected_responses, received_response




async def handle_fine_type(entity_type: str):
    documents_with_type = dataset_provider.get_by_fine_grained_type_fewnerd(entity_type)["hits"]["hits"]
    expected_responses = []
    received_response = []
    entity_type_no_backslash = " or ".join(entity_type.split("/"))
    for index, dataset_document in enumerate(documents_with_type):

        document = dataset_document["_source"]

        # For entities within document, extract all entities of the given type
        all_with_entity = entity_classifier.pick_entities_of_finetype_fewnerd(document["tagging"], entity_type)
        ground_truth = list(map(lambda x: parse_answer_item(x["phrase"].lower()), all_with_entity))

        prompt = entity_classifier.build_extraction_prompt(entity_type_no_backslash, document["full_text"])
        chat_gpt_result = send_chat_gpt(prompt)
        parsed_response = parse_extracted_entities(chat_gpt_result)

        yes_ground_truth = ["yes" for _ in range(len(ground_truth))]
        yes_predictions = ["yes" if entity in parsed_response else "no" for entity in ground_truth]

        if yes_ground_truth != yes_predictions:
            print(f'expected {ground_truth} but got {parsed_response}.\nprompt is: {prompt}')

        # For entities not in document, extract all entities of the given type
        all_without_entity = list(filter(lambda x: x not in all_with_entity, dataset_document["_source"]["tagging"]))
        ground_truth = list(map(lambda x: parse_answer_item(x["phrase"].lower()), all_without_entity))

        prompt = entity_classifier.build_extraction_prompt(entity_type_no_backslash, document["full_text"])
        chat_gpt_result = send_chat_gpt(prompt)
        parsed_response = parse_extracted_entities(chat_gpt_result)

        no_ground_truth = ["no" for _ in range(len(ground_truth))]
        no_predictions = ["yes" if entity in parsed_response else "no" for entity in ground_truth]

        if no_ground_truth != no_predictions:
            print(f'expected {ground_truth} but got {parsed_response}.\nprompt is: {prompt}')

        expected_responses.extend(yes_ground_truth)
        expected_responses.extend(no_ground_truth)
        received_response.extend(yes_predictions)
        received_response.extend(no_predictions)

    return expected_responses, received_response

async def main_fine_type():
    intersting_types = ["GPE", "company", "artist/author", "politician", "athlete", "sportsteam", "education",
                        "government/governmentagency", "sportsevent", "road/railway/highway/transit",
                        "attack/battle/war/militaryconflict", "media/newspaper", "bodiesofwater", "actor",
                        "biologything", "award", "writtenart", "music", "politicalparty", "sportsleague", "scholar",
                        "film", "showorganization", "soldier", "airplane", "language", "disease", "island", "religion",
                        "currency", "chemicalthing", "director", "mountain", "broadcastprogram", "software",
                        "livingthing", "law", "car", "park", "astronomything", "theater", "sportsfacility", "weapon",
                        "game", "ship", "hospital", "god", "airport", "library", "educationaldegree", "medical",
                        "hotel", "food", "train", "disaster", "restaurant", "election", "protest", "painting"]
    all_expected, all_received = [], []
    for type in intersting_types:
        expected_responses, received_response = await handle_fine_type(type)
        all_expected.extend(expected_responses)
        all_received.extend(received_response)
        confusion_matrix(expected_responses, received_response, type)

    confusion_matrix(all_expected, all_received, "all types")
    # Print accuracy, recall for all answers
    print(metrics.classification_report(all_expected, all_received))


async def main_coarse_type():
    intersting_types = ["person",
                        "location",
                        "organization",
                        "building",
                        "product",
                        "event",
                        "art"]
    all_expected, all_received = [], []
    for type in intersting_types:
        expected_responses, received_response = await handle_coarse_type(type)
        all_expected.extend(expected_responses)
        all_received.extend(received_response)
        confusion_matrix(expected_responses, received_response, type)

    confusion_matrix(all_expected, all_received, "all types")
    # Print accuracy, recall for all answers
    print(metrics.classification_report(all_expected, all_received))


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
    loop.run_until_complete(main_fine_type())
    loop.close()

""""
In the following sentence, write in a list the words or word phrases that represent a art: 
Animal Jam is a children 's television show created by John Derevlany ( who also wrote most of the episodes ) and produced by The Jim Henson Company which first aired on February 24 , 2003 .

Is Animal Jam an art?

In the following sentence, write in a list the words or word phrases that represent a building: 
Rockland Coaches provides service on the 14ET route to the
Port Authority Bus Terminal 
and on the 14K route to the George Washington Bridge Bus Station .
 if there are no words or word phrases that represent a building please write 'none'.

Port Authority Bus Terminal or Bus Station


In the following sentence, write in a list the words or word phrases that represent a person:
 Born on 24 July 1763, Rey joined the old royal army of 
 Louis XIV of France
  and became a lieutenant in 1791 .
 if there are no words or word phrases that represent a person please write 'none'.

Louis XIV of France vs Louis XIV




Andrey the dreadful - bad that it is a person or not?

jim henson company vs the jim henson company

"""
