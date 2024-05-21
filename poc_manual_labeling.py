import asyncio
import os.path
import clearml_poc
import pandas as pd
from sklearn import metrics
from clearml import Logger
import chat_gpt_client
import entity_classifier
from pathlib import Path


def send_chat_gpt(prompt) -> str:
    print("prompt" + prompt)
    chat_gpt_response = chat_gpt_client.send_message(prompt)
    parsed_response = chat_gpt_client.parse_answer_from_response(chat_gpt_response)
    result = parsed_response[0].lower()
    if len(parsed_response) != 1:
        print(f'more than one ansewer in {parsed_response}. taking first answer {parsed_response[0]}')
    print('chat gpt respone ', parsed_response)
    print('   ')
    answer_prefix = "answer:"
    if result.find(answer_prefix) > 0:
        result = result[len(answer_prefix) + result.find(answer_prefix):]
    result = result.replace(".", "")
    result = result.strip()
    return result


async def handle_classification_mission(data: list[dict]):
    expected_responses, received_responses = [], []
    for row in data:
        entity_type, entity_text, ground_truth_label, content = row.values()
        prompt = entity_classifier.build_classification_prompt(entity_type, entity_text, content)
        chat_gpt_result = send_chat_gpt(prompt)
        expected_responses.append(ground_truth_label)
        received_responses.append(chat_gpt_result)
        if ground_truth_label != chat_gpt_result:
            print(f'expected {ground_truth_label} but got {chat_gpt_result}')
    return expected_responses, received_responses


def parse_extracted_entities(response: str) -> list[str]:
    if response == "none":
        return []
    return response.split("\n")


async def handle_extraction_mission(data: list[dict]):
    expected_responses, received_responses = [], []
    for row in data:
        entity_type, entity_text, ground_truth_label, content = row.values()
        prompt = entity_classifier.build_extraction_prompt(entity_type, content)
        chat_gpt_result = send_chat_gpt(prompt)
        parsed_response = parse_extracted_entities(chat_gpt_result)
        expected_responses.append(ground_truth_label)
        prediction = "no"
        for entity in parsed_response:
            if entity_text in entity:
                prediction = "yes"
                break
        received_responses.append(prediction)
        if ground_truth_label != prediction:
            print(f'\nexpected {entity_text} but got {parsed_response}.'
                  f' Is the term expected to appear - {ground_truth_label}\n')
    return expected_responses, received_responses


async def main():
    path = Path(__file__).parent.joinpath("data").joinpath("manual_analysis.tsv").resolve()
    df = pd.read_csv(path, sep="\t")
    df["ground_truth_answer"].replace({1: "yes", 0: "no"}, inplace=True)
    data = df.to_dict('records')
    expected, received = await handle_extraction_mission(data)
    confusion_matrix(expected, received, "classification")


def confusion_matrix(expected_responses, received_responses, entity_type, iteration=1):
    #  clearml display confusion matrix
    confusion = metrics.confusion_matrix(expected_responses, received_responses)
    Logger.current_logger().report_confusion_matrix(
        f"confusion metrix",
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
