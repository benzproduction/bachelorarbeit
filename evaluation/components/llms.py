from typing import Any, Optional
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

def formatSourcesDF(sources):
    """
    Helper function for formatting a pandas dataframe into a string for the Prompt.
    """
    if isinstance(sources, pd.DataFrame):
        assert (
            "text" in sources.columns and "filename" in sources.columns
        ), "The df must have columns named 'text' and 'filename'"
        for i, row in sources.iterrows():
            sources.loc[i,'text'] = f"{row['filename']}: {row['text']};"
        return sources['text'].str.cat(sep="\n\n")
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
    def __init__(self, template: str, query: str, sources: list[str]):
        assert "{query}" in template, "Prompt template must contain {query}"
        assert "{sources}" in template, "Prompt template must contain {sources}"
        self.template = template
        self.query = query
        # if sources are provided as a pandas dataframe, format them
        if isinstance(sources, pd.DataFrame):
            sources = formatSourcesDF(sources)
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
        embedder: EmbeddingsFn,
        retriever: RetrieverFn,
        query: str,
        prompt_template: str,
        k: Optional[int] = 5,
        **kwargs,
    ) -> OpenAICompletionResult:
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