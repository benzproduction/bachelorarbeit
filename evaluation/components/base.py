
from typing import Protocol, runtime_checkable
from abc import ABC, abstractmethod



class CompletionResult(ABC):
    @abstractmethod
    def get_completions(self) -> list[str]:
        pass

@runtime_checkable
class CompletionFn(Protocol):
    def __call__(
        self,
        prompt: str,
        **kwargs,
    ) -> CompletionResult:
        """
        ARGS
        ====
        `prompt`: Either a `Prompt` object or a raw prompt that will get wrapped in
            the approriate `Prompt` class.
        `kwargs`: Other arguments passed to the API.

        RETURNS
        =======
        The result of the API call.
        The prompt that was fed into the API call as a str.
        """

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
    ) -> list[str]:
        """
        ARGS
        ====
        `query`: A string for which the retriever will return k similar documents.
        `embedder`: An EmbeddingsFn that can embed the query.
        `k`: The number of documents to retrieve.
        `kwargs`: Other arguments passed to the API.

        RETURNS
        =======
        The retrieved documents as a list[str].
        """