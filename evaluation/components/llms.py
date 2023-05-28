from typing import Any, Optional, List, Union
from base import CompletionFn, CompletionResult, EmbeddingsFn, RetrieverFn
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import backoff
import openai
import pandas as pd

@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.error.ServiceUnavailableError,
        openai.error.APIError,
        openai.error.RateLimitError,
        openai.error.APIConnectionError,
        openai.error.Timeout,
    ),
    max_value=60,
    factor=1.5,
)
def openai_completion_create_retrying(*args, **kwargs):
    """
    Helper function for creating a completion.
    `args` and `kwargs` match what is accepted by `openai.Completion.create`.
    """
    result = openai.Completion.create(*args, **kwargs)
    if "error" in result:
        logging.warning(result)
        raise openai.error.APIError(result["error"])
    return result

def formatSourcesDF(sources) -> str:
    """
    Helper function for formatting a pandas dataframe into a string for the Prompt.
    """
    if isinstance(sources, pd.DataFrame):
        assert "text" in sources.columns, "If sources are provided as a pandas dataframe, it must have a column named 'text'"
        assert "filename" in sources.columns or "key" in sources.columns, "If sources are provided as a pandas dataframe, it must have a column named 'filename' or 'key'"
        for i, row in sources.iterrows():
            if "key" in sources.columns:
                sources.loc[i,'text'] = f"{row['key']}: {row['text']};"
            else:
                sources.loc[i,'text'] = f"{row['filename']}: {row['text']};"
        return sources['text'].str.cat(sep="\n\n")
    return sources

def formatSourcesList(sources) -> str:
    """
    Helper function for formatting a list of dicts into a string for the Prompt.
    """
    if isinstance(sources, list):
        if all(isinstance(source, str) for source in sources):
            return "\n\n".join(sources)
        assert all(
            isinstance(source, dict) and "text" in source and ("key" in source or "filename" in source)
            for source in sources
        ), "If sources are provided as a list of dicts, they must have keys 'text' and 'key' or 'filename'"
        for i, source in enumerate(sources):
            if "key" in source:
                sources[i]["text"] = f"{source['key']}: {source['text']};"
            else:
                sources[i]["text"] = f"{source['filename']}: {source['text']};"
        return "\n\n".join([source["text"] for source in sources])
    return sources

@dataclass
class Prompt(ABC):
    """
    A `Prompt` encapsulates everything required to present the `raw_prompt` in different formats.
    """

    @abstractmethod
    def to_formatted_prompt(self):
        """
        Return the actual data to be passed as the `prompt` field to your model.
        """

class CompletionPrompt(Prompt):
    def __init__(self, template: str, query: str, sources: Union[str, pd.DataFrame, list]):
        assert "{query}" in template, "Prompt template must contain {query}"
        assert "{sources}" in template, "Prompt template must contain {sources}"
        self.template = template
        self.query = query
        # if sources are provided as a pandas dataframe, format them
        if isinstance(sources, pd.DataFrame):
            sources = formatSourcesDF(sources)
        if isinstance(sources, list):
            sources = formatSourcesList(sources)
        if not isinstance(sources, str):
            raise ValueError(f"Sources must be a str, list, or pandas dataframe. Got {type(sources)}")
        self.sources = sources

    def to_formatted_prompt(self):
        return self.template.format(query=self.query, sources=self.sources)

class OpenAIBaseCompletionResult(CompletionResult):
    def __init__(self, raw_data: Any, prompt: Any):
        self.raw_data = raw_data
        self.prompt = prompt

    def get_completions(self) -> list[str]:
        raise NotImplementedError

class OpenAICompletionResult(OpenAIBaseCompletionResult):
    def get_completions(self) -> list[str]:
        completions = []
        if self.raw_data and "choices" in self.raw_data:
            for choice in self.raw_data["choices"]:
                if "text" in choice:
                    completions.append(choice["text"])
                elif "message" in choice:
                    completions.append(choice["message"]["content"])
        return completions


class OpenAICompletionFn(CompletionFn):
    def __init__(
        self,
        api_key: str,
        api_base: str,
        deployment_name: Optional[str] = "text-davinci-003",
        api_type: Optional[str] = "azure",
        api_version: Optional[str] = "2022-12-01",
        extra_options: Optional[dict] = {},
        **kwargs,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.deployment_name = deployment_name
        self.api_type = api_type
        self.api_version = api_version
        self.extra_options = extra_options

    def __call__(
        self,
        prompt_template: str,
        query: str,
        sources: Optional[Union[str, pd.DataFrame, list]] = None,
        embedder: Optional[EmbeddingsFn] = None,
        retriever: Optional[RetrieverFn] = None,
        k: Optional[int] = 5,
        **kwargs,
    ) -> OpenAICompletionResult:
        assert sources or (isinstance(embedder, EmbeddingsFn) and isinstance(retriever, RetrieverFn)), "Either sources must be provided or an embedder and retriever must be provided"
        if not sources:
            sources = retriever(query, embedder, k=k)
        prompt = CompletionPrompt(template=prompt_template, query=query, sources=sources)
        result = openai_completion_create_retrying(
            engine=self.deployment_name,
            prompt=prompt.to_formatted_prompt(),
            api_key=self.api_key,
            api_base=self.api_base,
            api_type=self.api_type,
            api_version=self.api_version,
            **self.extra_options,
        )
        result = OpenAICompletionResult(raw_data=result, prompt=prompt)
        return result
    
    def __repr__(self):
        return f"OpenAICompletionFn(deployment_name={self.deployment_name}, extra_options={self.extra_options})"