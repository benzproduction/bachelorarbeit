"""
Functions that handle registry of models, prompts and datasets.
"""
from pathlib import Path
from typing import Any, Optional, Sequence, Type, Union, Dict
from utils.misc import make_object2
import logging
import copy
import difflib
import functools
import os
import yaml
from base import EmbeddingsSpec, EmbeddingsFn, RetrieverSpec, RetrieverFn, CompletionSpec, CompletionFn, PromptSpec

logger = logging.getLogger(__name__)

DEFAULT_PATHS = [Path(__file__).parents[0].resolve() /
                 "registry", Path.home() / ".ba-evals"]


class Registry:
    def __init__(self, registry_paths: Sequence[Union[str, Path]] = DEFAULT_PATHS):
        self._registry_paths = [Path(p) if isinstance(
            p, str) else p for p in registry_paths]

    def add_registry_paths(self, paths: list[Union[str, Path]]):
        self._registry_paths.extend(
            [Path(p) if isinstance(p, str) else p for p in paths])

    def _dereference(self, name: str, d: dict, object: str, type: Type, **kwargs: dict) -> dict:
        if not name in d:
            logger.warning(
                (
                    f"{object} '{name}' not found. "
                    f"Closest matches: {difflib.get_close_matches(name, d.keys(), n=5)}"
                )
            )
            return None

        def get_alias():
            if isinstance(d[name], str):
                return d[name]
            if isinstance(d[name], dict) and "id" in d[name]:
                return d[name]["id"]
            return None

        logger.debug(f"Looking for {name}")
        while True:
            alias = get_alias()

            if alias is None:
                break
            name = alias

        spec = d[name]
        if kwargs:
            spec = copy.deepcopy(spec)
            spec.update(kwargs)

        try:
            return type(**spec)
        except TypeError as e:
            raise TypeError(f"Error while processing {object} '{name}': {e}")

    def _checkenv(self, args: dict):
        for k, v in args.items():
            if isinstance(v, str) and v.startswith("${"):
                args[k] = os.getenv(v[2:-1])
                assert args[k] is not None, f"Environment variable {v} is not set"
        return args

    def make_embedding(self, name: str) -> EmbeddingsFn:
        spec = self.get_embedding(name)
        if spec is None:
            raise ValueError(
                f"Could not find an Embedding Class in the registry with ID {name}")
        if spec.args is None:
            spec.args = {}

        spec.args = self._checkenv(spec.args)

        spec.args["registry"] = self
        instance = make_object2(spec.cls)(**spec.args or {})
        assert isinstance(
            instance, EmbeddingsFn), f"{name} must be EmbeddingsFn"
        return instance

    def make_retriever(self, name: str, custom_args: Optional[Dict[str, Any]] = None) -> RetrieverFn:
        spec = self.get_retriever(name)
        if spec is None:
            raise ValueError(
                f"Could not find a Retriever Class in the registry with ID {name}")
        if spec.args is None:
            spec.args = {}
        if custom_args:
            spec.args.update(custom_args)

        spec.args["registry"] = self
        instance = make_object2(spec.cls)(**spec.args or {})
        assert isinstance(instance, RetrieverFn), f"{name} must be RetrieverFn"
        return instance

    def make_llm(self, name: str) -> CompletionFn:
        spec = self.get_llm(name)
        if spec is None:
            raise ValueError(
                f"Could not find a LLM Class in the registry with ID {name}")
        if spec.args is None:
            spec.args = {}

        spec.args = self._checkenv(spec.args)

        spec.args["registry"] = self
        instance = make_object2(spec.cls)(**spec.args or {})
        assert isinstance(instance, CompletionFn), f"{name} must be CompletionFn"
        return instance
    
    def get_prompt(self, name: str) -> str:
        spec = self._dereference(name, self._prompts, "prompt", PromptSpec)
        if spec is None:
            raise ValueError(
                f"Could not find a Prompt in the registry with ID {name}")
        return spec.prompt

    def get_embedding(self, name: str) -> EmbeddingsSpec:
        return self._dereference(name, self._embeddings, "embeddings", EmbeddingsSpec)

    def get_retriever(self, name: str) -> RetrieverSpec:
        return self._dereference(name, self._retrievers, "retriever", RetrieverSpec)

    def get_llm(self, name: str) -> CompletionSpec:
        return self._dereference(name, self._llms, "llm", CompletionSpec)

    def _process_file(self, registry, path):
        with open(path, "r") as f:
            d = yaml.safe_load(f)

        if d is None:
            # no entries in the file
            return

        for name, spec in d.items():
            assert name not in registry, f"duplicate entry: {name} from {path}"
            if isinstance(spec, dict):
                if "key" in spec:
                    raise ValueError(
                        f"key is a reserved keyword, but was used in {name} from {path}"
                    )
                if "group" in spec:
                    raise ValueError(
                        f"group is a reserved keyword, but was used in {name} from {path}"
                    )
                if "cls" in spec:
                    raise ValueError(
                        f"cls is a reserved keyword, but was used in {name} from {path}"
                    )

                spec["key"] = name
                spec["group"] = str(os.path.basename(path).split(".")[0])
                if "class" in spec:
                    spec["cls"] = spec["class"]
                    del spec["class"]
            registry[name] = spec

    def _process_directory(self, registry, path):
        files = Path(path).glob("*.yaml")
        for file in files:
            self._process_file(registry, file)

    def _load_registry(self, paths):
        """Load registry from a list of paths.

        Each path or yaml specifies a dictionary of name -> spec.
        """
        registry = {}
        for path in paths:
            logging.info(f"Loading registry from {path}")
            if os.path.exists(path):
                if os.path.isdir(path):
                    self._process_directory(registry, path)
                else:
                    self._process_file(registry, path)
        return registry

    @functools.cached_property
    def _embeddings(self):
        return self._load_registry([p / "embeddings" for p in self._registry_paths])

    @functools.cached_property
    def _retrievers(self):
        return self._load_registry([p / "retriever" for p in self._registry_paths])

    @functools.cached_property
    def _llms(self):
        return self._load_registry([p / "llm" for p in self._registry_paths])
    
    @functools.cached_property
    def _prompts(self):
        return self._load_registry([p / "prompt" for p in self._registry_paths])


if __name__ == "__main__":
    reg = Registry()
    embedder = reg.make_embedding("text_embedding_ada_002")
    retriever = reg.make_retriever("knn")
    print(retriever("hello", embedder))
    # llm = reg.make_llm("text-davinci-003")
    # print(llm)
    # prompt = reg.get_prompt("baseline")
