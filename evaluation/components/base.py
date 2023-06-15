
from typing import Protocol, runtime_checkable, Optional, Union
from abc import ABC, abstractmethod
import pandas as pd



class CompletionResult(ABC):
    @abstractmethod
    def get_completions(self) -> list[str]:
        pass

@runtime_checkable
class EmbeddingsFn(Protocol):
    def __call__(
        self,
        to_embs: str,
        **kwargs,
    ) -> list[float]:
        """
        ARGS
        ====
        `to_embs`: A string to be embedded.
        `kwargs`: Other arguments passed to the API.

        RETURNS
        =======
        The embedding of the string as a list[float].
        """

@runtime_checkable
class RetrieverFn(Protocol):
    def __call__(
        self,
        query: str, 
        embedder: EmbeddingsFn, 
        k :int = 5, 
        **kwargs,
    ) -> pd.DataFrame:
        """
        ARGS
        ====
        `query`: A string for which the retriever will return k similar documents.
        `embedder`: An EmbeddingsFn that can embed the query.
        `k`: The number of documents to retrieve.
        `kwargs`: Other arguments passed to the API.

        RETURNS
        =======
        The retrieved documents as a pandas dataframe.
        """

@runtime_checkable
class CompletionFn(Protocol):
    def __call__(
        self,
        prompt_template: str,
        query: str,
        sources: Optional[Union[str, pd.DataFrame, list]] = None,
        embedder: Optional[EmbeddingsFn] = None,
        retriever: Optional[RetrieverFn] = None,
        k: Optional[int] = 5,
        **kwargs,
    ) -> CompletionResult:
        """
        ARGS
        ====
        `prompt_template`: A string template that will be formatted with the query and sources.

        `query`: A string (question) to be answered.

        `sources`: Optional: A string, list of strings/dicts, or pandas dataframe to be used as sources. Dicts and dataframe except `text`, `filename` or `key` columns.

        `embedder`: Optional: If no sources are provided, then the query will be embedded using this EmbeddingsFn.

        `retriever`: Optional: If no sources are provided, then the query will be used to retrieve documents using this RetrieverFn.

        `kwargs`: Other arguments passed to the Function and/or underlying API.

        RETURNS
        =======
        The result of the LLM.
        The prompt that was fed into the LLM as a str.
        """
