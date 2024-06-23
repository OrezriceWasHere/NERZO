from sklearn import metrics
from clearml import Logger
import chat_gpt_client
import asyncio
import random
import dataset_provider
import entity_classifier
import clearml_poc


async def handle_fine_type(entity_type: str):
    documents_with_type = dataset_provider.get_by_fine_grained_type_fewnerd(entity_type)["hits"]["hits"]
    expected_responses = []
    received_response = []
    for index, dataset_document in enumerate(documents_with_type):
        document = dataset_document["_source"]
        # Verify two types of entites in the document
        if len(set([entity["fine_type"] for entity in document["tagging"]])) == 1:
            print("we need two types of entities in the document")
            continue

        # For yes answer
        yes_entities = entity_classifier.pick_entities_of_finetype_fewnerd(document["tagging"], entity_type)

        chosen_entity = random.choice(yes_entities)
        prompt = entity_classifier.build_classification_prompt(entity_type=entity_type,
                                                               entity_text=chosen_entity["phrase"],
                                                               sentence=document["full_text"])
        print(prompt)
        chat_gpt_response = chat_gpt_client.send_message(prompt)
        parsed_response = chat_gpt_client.parse_answer_from_response(chat_gpt_response)
        if len(parsed_response) != 1:
            print(f'more than one ansewer in {parsed_response}. taking first answer {parsed_response[0]}')

        answer = parsed_response[0].lower().replace(".", "")
        received_response.append(answer)
        expected_responses.append("yes")
        print('chat gpt respone ', parsed_response)
        print('   ')
        if answer != 'yes':
            print(f'expected yes but got {answer}')

        # For no answer
        no_entities = list(filter(lambda x: x not in yes_entities, dataset_document["_source"]["tagging"]))
        chosen_entity = random.choice(no_entities)
        entity_type_no_backslash = " or ".join(entity_type.split("/"))
        prompt = entity_classifier.build_classification_prompt(entity_type=entity_type_no_backslash,
                                                               entity_text=chosen_entity["phrase"],
                                                               sentence=document["full_text"])
        print(prompt)
        chat_gpt_response = chat_gpt_client.send_message(prompt)
        parsed_response = chat_gpt_client.parse_answer_from_response(chat_gpt_response)
        if len(parsed_response) != 1:
            print(f'more than one ansewer in {parsed_response}. taking first answer {parsed_response[0]}')

        answer = parsed_response[0].lower().replace(".", "")
        received_response.append(answer)
        expected_responses.append("no")
        print('chat gpt respone ', parsed_response)
        print('   ')
        if answer != 'no':
            print(f'expected no but got {answer}')

    return expected_responses, received_response

async def handle_coarse_type(entity_type: str):
    documents_with_type = dataset_provider.get_by_coarse_grained_type_fewnerd(entity_type)["hits"]["hits"]
    expected_responses = []
    received_response = []
    for index, dataset_document in enumerate(documents_with_type):
        document = dataset_document["_source"]
        # Verify two types of entites in the document
        if len(set([entity["coarse_type"] for entity in document["tagging"]])) <= 1:
            print("we need two types of entities in the document")
            continue

        # For yes answer
        yes_entities = entity_classifier.pick_entities_of_coarsetype_fewnerd(document["tagging"], entity_type)

        chosen_entity = random.choice(yes_entities)
        prompt = entity_classifier.build_classification_prompt(entity_type=entity_type,
                                                               entity_text=chosen_entity["phrase"],
                                                               sentence=document["full_text"])
        print(prompt)
        chat_gpt_response = chat_gpt_client.send_message(prompt)
        parsed_response = chat_gpt_client.parse_answer_from_response(chat_gpt_response)
        if len(parsed_response) != 1:
            print(f'more than one ansewer in {parsed_response}. taking first answer {parsed_response[0]}')

        answer = parsed_response[0].lower().replace(".", "")
        received_response.append(answer)
        expected_responses.append("yes")
        print('chat gpt respone ', parsed_response)
        print('   ')
        if answer != 'yes':
            print(f'expected yes but got {answer}')

        # For no answer
        no_entities = list(filter(lambda x: x not in yes_entities, dataset_document["_source"]["tagging"]))
        chosen_entity = random.choice(no_entities)
        entity_type_no_backslash = " or ".join(entity_type.split("/"))
        prompt = entity_classifier.build_classification_prompt(entity_type=entity_type_no_backslash,
                                                               entity_text=chosen_entity["phrase"],
                                                               sentence=document["full_text"])
        print(prompt)
        chat_gpt_response = chat_gpt_client.send_message(prompt)
        parsed_response = chat_gpt_client.parse_answer_from_response(chat_gpt_response)
        if len(parsed_response) != 1:
            print(f'more than one ansewer in {parsed_response}. taking first answer {parsed_response[0]}')

        answer = parsed_response[0].lower().replace(".", "")
        received_response.append(answer)
        expected_responses.append("no")
        print('chat gpt respone ', parsed_response)
        print('   ')
        if answer != 'no':
            print(f'expected no but got {answer}')

    return expected_responses, received_response


def group_fewnerd_together(document):
    if not document:
        return None
    current_label = document["tagging"][0]["label"]
    new_document = {
        "text_id": document["text_id"],
        "full_text": document["full_text"],
        "tagging": [{"word": "",
                     "label": current_label,
                     "offset_in_word": 0,
                     "text_id": document["text_id"]}]
    }
    for tagging in document["tagging"]:
        if tagging["label"] == current_label:
            new_document["tagging"][-1]["word"] += " " + tagging["word"]
        else:
            current_label = tagging["label"]
            new_document["tagging"].append({"word": tagging["word"],
                                            "label": current_label,
                                            "offset_in_word": len(new_document["tagging"]),
                                            "text_id": document["text_id"]})

    return new_document


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
    loop.run_until_complete(main_coarse_type())
    loop.close()

""""

1. Is helicopter an airplane:
In the following sentence, specify if the word or word phrase 'CH-53 Sea Stallion  Yas'ur ''' represent a airplane:
Tendler was killed during the 2006 Israel-Lebanon conflict , along with four other crew members , when their helicopter 
, a CH-53 Sea Stallion `` Yas'ur '' ,was shot down upon lifting off in Lebanon .
 Please answer 'yes' or 'no'.


2. How to handle part of information:

In the following sentence, specify if the word or word phrase 'Zeppelin LZ 4' represent a airplane: 
Zeppelin LZ 4 caught fire and burned out in Echterdingen in August 1908 .
Please answer 'yes' or 'no'.

In the following sentence, specify if the word or word phrase 'Herta' represent a politician: 
They were held in Most and later brought to Berlin, 
where Herta was sentenced to death for high treason in November 1942 . Please answer 'yes' or 'no'.

In the following sentence, specify if the word or word phrase 'Johnson' represent a athlete: 
Elliott would return to victory lane that year at Darlington,
 but left Johnson at season 's end to form his own team .
Please answer 'yes' or 'no'.


3. Should we check for high level (coarse) type and for fine type?


4. extraction - how should we handle the mission:
should we pick a word and see if it is mentioned, 
or should we let the model extact all words and count matching according to the data we have. 

lets say we know that the dataset has label 'people' for phrases: A B C D
and the model says: A B E F

should we pick random word A or should we let the model state all predictions and than to let it count the correct ones.



"""
