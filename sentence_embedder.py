from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from os import environ  as env

class SentenceEmbedder:

    def __init__(self, llm_id, passage_prompt="", query_prompt=""):
        self.model = self.build_llm_by_id(llm_id)
        self.passage_prompt = passage_prompt
        self.query_prompt = query_prompt



    def build_llm_by_id(self, llm_id):
        HUGGINGFACE_TOKEN = env.get("HUGGINGFACE_TOKEN")
        login(token=HUGGINGFACE_TOKEN)
        match llm_id:
            case 'nvidia/NV-Embed-v1':
                model = SentenceTransformer(llm_id,  trust_remote_code=True)
                model.max_seq_length = 32768
                model.tokenizer.padding_side="right"

            case _:
                model = SentenceTransformer(llm_id)

        return model

    def add_eos(self, input_examples):
        input_examples = [input_examples] if not isinstance(input_examples, list) else input_examples
        input_examples = [input_example + self.model.tokenizer.eos_token for input_example in input_examples]
        return input_examples

    def forward_passage(self, passage):
        passage_embeddings = self.model.encode(passage, batch_size=1)
        return passage_embeddings


    def forward_query(self, query):
        query_embeddings = self.model.encode(self.add_eos(query), batch_size=1, prompt=self.query_prompt,
                                        normalize_embeddings=True)
        return query_embeddings


