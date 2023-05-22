"""
This file defines the base specifications for models, evals, and runs.
"""
import base64
import datetime
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Protocol, Union, runtime_checkable
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

@dataclass
class PromptSpec:
    """
    Specification for a Prompt.
    """
    prompt: str
    key: Optional[str] = None
    group: Optional[str] = None

@dataclass
class EmbeddingsSpec:
    """
    Specification for a EmbeddingsFn.
    """
    cls: str
    args: Optional[Dict[str, Any]] = None
    key: Optional[str] = None
    group: Optional[str] = None

@dataclass
class RetrieverSpec:
    """
    Specification for a RetrieverFn.
    """
    cls: str
    args: Optional[Dict[str, Any]] = None
    key: Optional[str] = None
    group: Optional[str] = None

@dataclass
class CompletionSpec:
    """
    Specification for a CompletionFn.
    """
    cls: str
    args: Optional[Dict[str, Any]] = None
    key: Optional[str] = None
    group: Optional[str] = None

@dataclass
class RunSpec:
    eval_type: str
    eval_name: str
    run_config: Dict[str, Any]
    data: List[Tuple[Dict[str, Any], int]]
    created_by: Optional[str] = None
    run_id: str = None
    created_at: str = None

    def __post_init__(self):
        now = datetime.datetime.utcnow()
        rand_suffix = base64.b32encode(os.urandom(5)).decode("ascii")
        self.run_id = now.strftime("%y%m%d%H%M%S") + rand_suffix
        self.created_at = str(now)




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
        k :Optional[int] = 5, 
        **kwargs,
    ) -> list[str]:
        """
        ARGS
        ====
        `query`: A string to be retrieved.
        `embedder`: An EmbeddingsFn that can embed the query.
        `k`: The number of documents to retrieve.
        `kwargs`: Other arguments passed to the API.

        RETURNS
        =======
        The retrieved documents as a list[str].
        """

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
