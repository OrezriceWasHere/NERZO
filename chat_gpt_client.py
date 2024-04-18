import os
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def send_message(message: str | list[str],
                 role="system",
                 model="gpt-3.5-turbo") -> ChatCompletion | Stream[ChatCompletionChunk]:
    if isinstance(message, str):
        messages = [{
            "role": role,
            "content": message
        }]
    else:
        messages = [
            {
                "role": role,
                "content": single_message,
            }
            for single_message in message
        ]

    return client.chat.completions.create(
        model=model,
        messages=messages
    )


def parse_answer_from_response(response) -> list[str]:
    return [item.message.content for item in response.choices]
