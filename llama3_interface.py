from os import environ as env
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import BitsAndBytesConfig
from huggingface_hub import login


def find(input_sentence, word):
    assert word in input_sentence, f"{word} not in {input_sentence}"
    assert len(word) > 0, "word must be at least one character long"
    possible_characters_after_word = [
        " ", ".", ",", "!", "?", ":", ";", ")", "]", "}", "'", '"', "EOS"
    ]
    possible_characters_before_word = [
        " ", "(", "[", "{", "'", '"', "SOS"
    ]

    index_start = 0
    while index_start < len(input_sentence):
        part_of_setence = input_sentence[index_start:]
        possible_match = part_of_setence.find(word)

        if possible_match == -1:
            break
        prev_char = part_of_setence[possible_match - 1] if possible_match > 0 else "SOS"
        next_char = part_of_setence[possible_match + len(word)] if possible_match + len(word) < len(
            input_sentence) else "EOS"

        if prev_char in possible_characters_before_word and next_char in possible_characters_after_word:
            return possible_match + index_start
        else:
            index_start = possible_match + len(word)

    raise ValueError(f"Could not find {word} in {input_sentence}")


class LLama3Interface:

    def __init__(self):
        HUGGINGFACE_TOKEN = env.get("HUGGINGFACE_TOKEN")
        LLAMA_3_ID = "meta-llama/Meta-Llama-3-8B"

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        login(token=HUGGINGFACE_TOKEN)

        self.tokenizer = AutoTokenizer.from_pretrained(LLAMA_3_ID)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = (AutoModelForCausalLM.from_pretrained(LLAMA_3_ID,
                                                           device_map="auto",
                                                           quantization_config=nf4_config))

    def tokenize(self, prompt: str | list[str]) -> torch.Tensor:
        return self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)

    def tokens_indices_part_of_sentence(self, sentence, part_of_sentence):
        """
        Given a sentence and a part of the sentence, return the indices of the part of the sentence in the tokenized sentence.
        For example:
        sentence = "The quick brown fox jumps over the lazy dog."
        part_of_sentence = "quick brown fox"
        tokens_indices_part_of_sentence(sentence, part_of_sentence) -> (2, 5)

        :param sentence: a raw input to llama 3 model
        :param part_of_sentence: a part of the sentence that we want to find in the tokenized sentence
        :return: a range of indices in the tokenized sentence
        """

        # The LLM's tokenizer splits the sentence into tokens, and this is the input for the model. We need to find the indices of the part of the sentence in order to learn its hidden states.
        assert part_of_sentence in sentence

        try:

            token_starts_offsets = self.tokenizer(sentence,
                                                  return_offsets_mapping=True,
                                                  return_tensors="pt").offset_mapping[0].transpose(0, -1)[0].tolist()
            start_offset = find(sentence, part_of_sentence)

            if start_offset > 0 and sentence[start_offset - 1] == " ":
                # space in tokenization algorithm is included before the word
                start_token_index = token_starts_offsets.index(start_offset - 1)
            else:
                start_token_index = token_starts_offsets.index(start_offset)
            # The start of sentence has a special token with no textual representation
            start_token_index = max(1, start_token_index)

            end_sentence_offset = start_offset + len(part_of_sentence)
            if end_sentence_offset == len(sentence):
                end_token_index = len(token_starts_offsets)
            else:
                end_token_index = token_starts_offsets.index(end_sentence_offset)

        except Exception as e:
            print(f'for sentence: {sentence} and part of sentence: {part_of_sentence} the error is {e}')
            raise e

        return start_token_index, end_token_index

    def get_hidden_layers(self, inputs) -> tuple[torch.Tensor]:
        with torch.no_grad():
            outputs = self.model.forward(**inputs, output_hidden_states=True)
        return outputs["hidden_states"]
