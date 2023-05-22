from typing import Any, Optional
import openai
from base import EmbeddingsFn


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
