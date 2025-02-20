from os import environ as env
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import BitsAndBytesConfig
from huggingface_hub import login
from transformers import T5EncoderModel
from peft import LoraConfig, get_peft_model, PeftType

import llama3_tokenizer
from runtime_args import RuntimeArgs


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


def load_model_tokenizer(llm_id: str, tokenizer_llm_id: str = None, lora_config=None, max_llm_layer=None):
    HUGGINGFACE_TOKEN = env.get("HUGGINGFACE_TOKEN")
    login(token=HUGGINGFACE_TOKEN)

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    match tokenizer_llm_id:
        case "meta-llama/Meta-Llama-3.1-8B" | "meta-llama/Llama-3.3-70B-Instruct":
            tokenizer = llama3_tokenizer.CustomLlama3Tokenizer(tokenizer_llm_id)
            tokenizer.tokenizer.pad_token = tokenizer.tokenizer.eos_token
        case _:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_llm_id)

    match llm_id:
        case "google-t5/t5-11b":
            T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
            model = T5EncoderModel.from_pretrained(llm_id,
                                                   quantization_config=nf4_config,
                                                   device_map="auto")

        case _:
            model = (AutoModelForCausalLM.from_pretrained(llm_id,
                                                          device_map="auto",
                                                          quantization_config=nf4_config
                                                          ))
    if max_llm_layer:
        del model.model.layers[max_llm_layer:]
        torch.cuda.empty_cache()

    model = model.eval()

    if lora_config:
        model = get_peft_model(model, lora_config)
        model = model.train()

    return model, tokenizer


class LLMInterface:

    def __init__(self, llm_id="meta-llama/Meta-Llama-3.1-8B",
                 interested_layers=None,
                 lora_config=None,
                 tokenizer_llm_id=None,
                 max_llm_layer=None,
                 **kwargs):
        tokenizer_llm_id = tokenizer_llm_id or llm_id

        print(f'LLM ID: {llm_id}')
        self.model, self.tokenizer = load_model_tokenizer(llm_id, tokenizer_llm_id, lora_config, max_llm_layer=max_llm_layer)

        self.extractable_parts = {}
        self.interested_layers = interested_layers or kwargs.get("layer") or []
        self.register_hooks(self.model)

    def hook_fn(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.extractable_parts[name] = output
            else:
                self.extractable_parts[name] = output[0]

        return hook

    def register_hooks(self, model):
        ignore = ["model.rotary_emb", "lm_head", "encoder.block.0.layer.0.SelfAttention.relative_attention_bias"]
        ignore = ignore + [f"model.layers.{layer}.self_attn.rotary_emb" for layer in range(32)]

        for name, module in model.named_modules():
            if name in ignore:
                continue
            if any([not self.interested_layers, name in self.interested_layers]):
                module.register_forward_hook(self.hook_fn(name))

    def tokenize(self, prompt: str | list[str]) -> torch.Tensor:
        return self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)

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

        use_offset_mapping = False
        try:

            if not use_offset_mapping:

                token_starts_offsets = self.tokenizer(sentence,
                                                      return_offsets_mapping=True,
                                                      return_tensors="pt").offset_mapping[0].transpose(0, -1)[
                    0].tolist()
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
            else:
                token_starts_offsets, token_end_offsets = self.tokenizer(sentence,
                                                                         return_offsets_mapping=True,
                                                                         return_tensors="pt").offset_mapping.transpose(
                    0, -1)
                token_starts_offsets, token_end_offsets = token_starts_offsets.squeeze(
                    1).tolist(), token_end_offsets.squeeze(1).tolist()
                start_offset = find(sentence, part_of_sentence)
                start_token_index = token_starts_offsets.index(start_offset)
                end_offsets = start_offset + len(part_of_sentence)
                end_token_index = token_end_offsets.index(end_offsets)
                return start_token_index, end_token_index



        except Exception as e:
            print(f'for sentence: {sentence} and part of sentence: {part_of_sentence} the error is {e}')
            raise e

        return start_token_index, end_token_index

    def token_indices_given_text_indices(self, sentence, text_indices):
        # offsets = self.tokenizer(sentence,
        #                          return_offsets_mapping=True,
        #                          return_tensors="pt").offset_mapping[0].transpose(0, -1)
        #
        # Tokenize the sentence and get word-to-token mappings (word_ids)
        # tokens = self.tokenizer.tokenize(sentence)
        encoding = self.tokenizer(sentence, return_offsets_mapping=True)

        # Get the offsets for each token (start, end positions in original sentence)
        offsets = encoding['offset_mapping']

        start_idx, end_idx = text_indices

        # Find the first and last tokens that correspond to the given word's string indices
        first_token_idx, last_token_idx = None, None

        for i, (token_start, token_end) in enumerate(offsets):
            if token_start <= start_idx < token_end:  # First token that overlaps with start_idx
                first_token_idx = i
            if token_start < end_idx <= token_end:  # Last token that overlaps with end_idx
                last_token_idx = i
                break

        assert first_token_idx is not None and last_token_idx is not None, f"Could not find token indices for text indices {text_indices} in sentence {sentence}"

        return first_token_idx, last_token_idx

    def get_hidden_layers(self, inputs) -> tuple[torch.Tensor]:
        with torch.no_grad():
            outputs = self.model.forward(**inputs, output_hidden_states=True)
        return outputs["hidden_states"]

    def get_llm_at_layer(self, inputs, layer, retain_layers_dict=False, clone=True) -> torch.Tensor:
        parts = self.get_hooked_hidden_layers(inputs, no_grad=clone)
        h = parts[layer]
        if clone:
            h = h.detach().clone()
        if not retain_layers_dict:
            self.extractable_parts.clear()
            parts.clear()
        return h

    def get_hooked_hidden_layers(self, inputs, no_grad=True) -> dict[str, torch.Tensor]:
        if no_grad:
            with torch.no_grad():
                self.extractable_parts.clear()
                _ = self.model.forward(**inputs)
        else:
            self.extractable_parts.clear()
            _ = self.model.forward(**inputs)
        return self.extractable_parts

    def get_attention_layers(self, inputs) -> tuple[torch.Tensor]:
        with torch.no_grad():
            outputs = self.model.forward(**inputs, output_attentions=True)
        return outputs["attentions"]