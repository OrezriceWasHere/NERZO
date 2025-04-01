from abc import ABC

import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

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

    def dim_size(self) -> int:
        raise NotImplementedError()


class SentenceEmbedder(AbstractSentenceEmbedder):

    def __init__(self, llm_id, passage_prompt="", query_prompt="", **kwargs):
        super().__init__(llm_id, passage_prompt, query_prompt)
        factory = {
            "intfloat/e5-mistral-7b-instruct": E5MistralSentenceEmbedder,
            "meta-llama/Meta-Llama-3.1-8B": LlamaSentenceEmbedder,
            "nvidia/NV-Embed-v2": NV_Embed_V2
        }
        assert llm_id in factory, "cannot construct sentence embedder with id {}".format(llm_id)
        sentence_embedder_class = factory[llm_id]
        self.sentence_embedder = sentence_embedder_class(llm_id, query_prompt, passage_prompt, **kwargs)

    def forward_passage(self, passage):
        return self.sentence_embedder.forward_passage(passage)

    def forward_query(self, query):
        return self.sentence_embedder.forward_query(query)

    def dim_size(self) -> int:
        return self.sentence_embedder.dim_size()

class E5MistralSentenceEmbedder(AbstractSentenceEmbedder):

    def __init__(self, llm_id, passage_prompt="", query_prompt="", **kwargs):
        super().__init__(llm_id, passage_prompt, query_prompt)
        self.model = SentenceTransformer(llm_id)

    def add_eos(self, input_examples):
        input_examples = input_examples + self.model.tokenizer.eos_token
        return input_examples

    def forward_passage(self, passages):
        passages = passages if isinstance(passages, list) else [passages]
        passage_embeddings = self.model.encode(passages, batch_size=len(passages), prompt=self.passage_prompt,
                                               normalize_embeddings=True)
        return passage_embeddings


    def forward_query(self, queries):
        queries = queries if isinstance(queries, list) else [queries]
        query_embeddings = self.model.encode(queries, batch_size=len(queries), prompt=self.query_prompt,
                                        normalize_embeddings=True)
        return query_embeddings

    def dim_size(self):
        return 4096

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

    def dim_size(self):
        return 1024

class NV_Embed_V2(AbstractSentenceEmbedder):

    def __init__(self, llm_id, passage_prompt="", query_prompt="", **kwargs):
        self.model = SentenceTransformer(llm_id, trust_remote_code=True)
        self.model.max_seq_length = 65356
        self.model.tokenizer.padding_side = "right"
        self.query_prefix = query_prompt
        self.model.half()
        torch.cuda.empty_cache()


    def add_eos(self, input_examples):
        input_examples = [input_example + self.model.tokenizer.eos_token for input_example in input_examples]
        return input_examples

    def forward_passage(self, passage: str | list[str]):
        passages = [passage] if isinstance(passage, str) else passage
        passages = [item[:9000] for item in passages]
        passage_embeddings = self.model.encode(self.add_eos(passages),
                                               convert_to_numpy=False,
                                               convert_to_tensor=True,
                                               batch_size=len(passage), normalize_embeddings=True).half()
        return passage_embeddings

    def forward_query(self, query):
        queries = [query] if isinstance(query, str) else query
        query_embeddings = self.model.encode(
            self.add_eos(queries), convert_to_numpy=False,
            convert_to_tensor=True,
            batch_size=len(query), prompt=self.query_prefix, normalize_embeddings=True
            ).half()
        return query_embeddings

    def dim_size(self) -> int:
        return 4096
