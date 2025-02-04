from abc import ABC
from sentence_transformers import SentenceTransformer
from llm_interface import LLMInterface



class AbstractSentenceEmbedder(ABC):

    def __init__(self, llm_id, passage_prompt="", query_prompt="", **kwargs):
        self.llm_id = llm_id
        self.passage_prompt = passage_prompt
        self.query_prompt = query_prompt

    def forward_passage(self, passage):
        raise NotImplementedError()

    def forward_query(self, query):
        raise NotImplementedError()

class SentenceEmbedder(AbstractSentenceEmbedder):

    def __init__(self, llm_id, passage_prompt="", query_prompt="", **kwargs):
        super().__init__(llm_id, passage_prompt, query_prompt)
        factory = {
            "intfloat/e5-mistral-7b-instruct": E5MistralSentenceEmbedder,
            "meta-llama/Meta-Llama-3.1-8B": LlamaSentenceEmbedder,
        }
        assert llm_id in factory, "cannot construct sentence embedder with id {}".format(llm_id)
        sentence_embedder_class = factory[llm_id]
        self.sentence_embedder = sentence_embedder_class(llm_id, query_prompt, passage_prompt, **kwargs)

    def forward_passage(self, passage):
        return self.sentence_embedder.forward_passage(passage)

    def forward_query(self, query):
        return self.sentence_embedder.forward_query(query)

class E5MistralSentenceEmbedder(AbstractSentenceEmbedder):

    def __init__(self, llm_id, passage_prompt="", query_prompt="", **kwargs):
        super().__init__(llm_id, passage_prompt, query_prompt)
        self.model = SentenceTransformer(llm_id)

    def add_eos(self, input_examples):
        input_examples = input_examples + self.model.tokenizer.eos_token
        return input_examples

    def forward_passage(self, passage):
        passage_embeddings = self.model.encode(passage, batch_size=1)
        return passage_embeddings


    def forward_query(self, query):
        query_embeddings = self.model.encode(query, batch_size=1, prompt=self.query_prompt,
                                        normalize_embeddings=True).tolist()
        return query_embeddings


class LlamaSentenceEmbedder(AbstractSentenceEmbedder):
    def __init__(self, llm_id, passage_prompt="", query_prompt="", **kwargs):
        super().__init__(llm_id, passage_prompt, query_prompt)
        self.model = LLMInterface(llm_id, **kwargs)
        assert "layer" in kwargs, "cannot construct sentence embedder without layer"
        self.interested_layer = kwargs['layer']


    def embedding_at_eos(self, text):
        tokens = self.model.tokenize(text).to('cuda')
        passage_embeddings = self.model.get_llm_at_layer(tokens, layer=self.interested_layer)
        passage_embeddings = passage_embeddings[:, -1, :].squeeze(dim=0)
        return passage_embeddings


    def forward_passage(self, passage):
        return self.embedding_at_eos(passage)

    def forward_query(self, query):
        return self.embedding_at_eos(query).detach().cpu().tolist()
