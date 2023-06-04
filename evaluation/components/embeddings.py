from typing import Any, Optional
import openai
from base import EmbeddingsFn
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import warnings


class OpenAI_Ada002(EmbeddingsFn):
    def __init__(
        self,
        api_key: str,
        api_base: str,
        deployment_name: Optional[str] = "text-embedding-ada-002",
        api_type: Optional[str] = "azure",
        api_version: Optional[str] = "2022-12-01",
        extra_options: Optional[dict] = {},
        **_kwargs: Any
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.deployment_name = deployment_name
        self.api_type = api_type
        self.api_version = api_version
        self.extra_options = extra_options

    def __call__(self, to_embs: str, **kwds: Any) -> list[float]:
        assert isinstance(to_embs, str), "The input to the embeddings function must be a string."
        to_embs = to_embs.replace("\n", " ")
        embedded = openai.Embedding.create(
            engine=self.deployment_name, 
            input=[to_embs],
            api_key=self.api_key,
            api_base=self.api_base,
            api_type=self.api_type,
            api_version=self.api_version,
            **{**kwds, **self.extra_options},
        )
        return embedded["data"][0]["embedding"]
    
    def __repr__(self):
        return f"OpenAI_Ada002(deployment_name={self.deployment_name}, extra_options={self.extra_options})"
    
class Intfloat_e5_large_v2(EmbeddingsFn):
    def __init__(
            self,
            **_kwargs: Any
    ):
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
        self.model = AutoModel.from_pretrained('intfloat/e5-large-v2')
    
    def _average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def __call__(self, to_embs: str, **kwds: Any) -> list[float]:
        assert isinstance(to_embs, str), "The input to the embeddings function must be a string."
        to_embs = to_embs.replace("\n", " ")
        if len(to_embs) > 512:
            warnings.warn(f"Input string is longer than 512 characters. Please check the input string.")
        batch_dict = self.tokenizer(to_embs, max_length=512, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**batch_dict)
        embedding = self._average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        return embedding.detach().numpy().tolist()
    
    def __repr__(self):
        return f"Intfloat_e5_large_v2()"