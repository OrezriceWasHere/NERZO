import asyncio
import pandas as pd
from sklearn import metrics
from clearml import Logger
import clearml_poc
import chat_gpt_client
import entity_classifier


def send_chat_gpt(prompt) -> str:
    print("prompt" + prompt)
    chat_gpt_response = chat_gpt_client.send_message(prompt)
    parsed_response = chat_gpt_client.parse_answer_from_response(chat_gpt_response)
    result = parsed_response[0].lower()
    if len(parsed_response) != 1:
        print(f'more than one ansewer in {parsed_response}. taking first answer {parsed_response[0]}')
    print('chat gpt respone ', parsed_response)
    print('   ')
    return result


async def handle_classification_mission(data: list[dict]):
    expected_responses, received_responses = [], []
    for row in data:
        entity_type, entity_text, ground_truth_label, content = row.values()
        prompt = entity_classifier.build_classification_prompt(entity_type, entity_text, content)
        chat_gpt_result = send_chat_gpt(prompt)
        expected_responses.append(ground_truth_label)
        received_responses.append(chat_gpt_result)
    return expected_responses, received_responses


async def handle_extraction_mission(data: list[dict]):
    expected_responses, received_responses = [], []
    for row in data:
        entity_type, entity_text, ground_truth_label, content = row.values()
        prompt = entity_classifier.build_extraction_prompt(entity_type, entity_text)
        chat_gpt_result = send_chat_gpt(prompt)
        expected_responses.append(ground_truth_label)
        received_responses.append(chat_gpt_result)
    return expected_responses, received_responses


async def main():
    df = pd.read_csv("data\manual_analysis.tsv", sep="\t")
    data = df.to_dict('records')
    expected, received = await handle_classification_mission(data)
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
