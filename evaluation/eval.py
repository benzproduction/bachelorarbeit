"""
This file defines the `EvalRun` classes which log eval results to a local JSON file.
"""
import atexit
from base import RunSpec
from typing import Any, Dict, List, Optional, Sequence, Union, Callable, Iterable, Tuple
from dataclasses import dataclass, asdict
import time
import threading
import blobfile as bf
from utils.files import jsondumps
from utils.misc import t
import logging
from contextvars import ContextVar
from datetime import datetime, timezone
import pandas as pd
import re
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.metrics import edit_distance
from nltk.translate import meteor, bleu
import tiktoken
import yaml
import os
import openai
import string

logger = logging.getLogger(__name__)
MIN_FLUSH_EVENTS = 100
MIN_FLUSH_SECONDS = 10

def _green(str):
    return f"\033[1;32m{str}\033[0m"
def _red(str):
    return f"\033[1;31m{str}\033[0m"
def _yellow(str):
    return f"\033[1;33m{str}\033[0m"

# class that gets the run specs from the user and actually runs the evaluation
@dataclass
class Event:
    run_id: str
    event_id: int
    sample_id: Optional[str]
    type: str
    data: dict
    created_by: Optional[str]
    created_at: str


class EvalRun():

    def __init__(
        self,
        run_spec: RunSpec,
        log_path: str,
    ) -> None:
        self._sample_id: ContextVar[Optional[int]] = ContextVar("_sample_id", default=None)
        self.run_spec = run_spec
        self.eval_type = run_spec.eval_type
        self._events: List[Event] = []
        self._last_flush_time = time.time()
        self._flushes_done = 0
        self._written_events = 0
        self._flushes_started = 0
        self._event_lock = threading.Lock()
        self._paused_ids: List[str] = []
        self._default_tokenizer = tiktoken.get_encoding("cl100k_base")
        atexit.register(self.flush_events)
        self.event_file_path = log_path
        if log_path is not None:
            spec = asdict(self.run_spec)
            ignore = {"data"}
            new_spec = {key: value for key,
                        value in spec.items() if key not in ignore}
            with bf.BlobFile(log_path, "wb") as f:
                f.write((jsondumps({"spec": new_spec}) + "\n").encode("utf-8"))

    def _flush_events_internal(self, events_to_write: Sequence[Event]):
        start = time.time()
        try:
            lines = [jsondumps(event) + "\n" for event in events_to_write]
        except TypeError as e:
            logger.error(f"Failed to serialize events: {events_to_write}")
            raise e

        with bf.BlobFile(self.event_file_path, "ab") as f:
            f.write(b"".join([l.encode("utf-8") for l in lines]))

        logger.info(
            f"Logged {len(lines)} rows of events to {self.event_file_path}: insert_time={t(time.time()-start)}"
        )

        self._last_flush_time = time.time()
        self._flushes_done += 1
        pass

    def flush_events(self):
        with self._event_lock:
            if len(self._events) == self._written_events:
                return
            events_to_write = self._events[self._written_events:]
            self._written_events = len(self._events)
            self._flushes_started += 1
        self._flush_events_internal(events_to_write)

    def current_sample_id(self) -> Optional[str]:
        return self._sample_id.get()
    
    def get_events(self, type: str) -> Sequence[Event]:
        with self._event_lock:
            return [event for event in self._events if event.type == type]
    
    def record_event(self, type, data=None, sample_id=None):
        if sample_id is None:
            sample_id = self.current_sample_id()
        if sample_id is None:
            raise ValueError("No sample_id set! Either pass it in or use as_default_recorder!")

        # if self.is_paused(sample_id):
        #     return
        with self._event_lock:
            event = Event(
                run_id=self.run_spec.run_id,
                event_id=len(self._events),
                type=type,
                sample_id=sample_id,
                data=data,
                created_by=self.run_spec.created_by,
                created_at=str(datetime.now(timezone.utc)),
            )
            self._events.append(event)
            if (
                self._flushes_done < self._flushes_started
                or len(self._events) < self._written_events + MIN_FLUSH_EVENTS
                or time.time() < self._last_flush_time + MIN_FLUSH_SECONDS
            ):
                return
            events_to_write = self._events[self._written_events :]
            self._written_events = len(self._events)
            self._flushes_started += 1
            self._flush_events_internal(events_to_write)

    def record_final_report(self, final_report: Any):
        with bf.BlobFile(self.event_file_path, "ab") as f:
            f.write(
                (jsondumps({"final_report": final_report}) + "\n").encode("utf-8"))

        logging.info(
            f"Final report: {final_report}. Logged to {self.event_file_path}")
    
    def get_metric_avg(self, metric_name: str) -> float:
        metric_values = []
        for event in self.get_events("metric"):
            if event.data is not None and metric_name in event.data.keys():
                metric_info = event.data[metric_name]
                if isinstance(metric_info, str):
                    metric_values.append(event.data["info"]["score"])
                elif isinstance(metric_info, (int, float)):
                    metric_values.append(metric_info)
        return sum(metric_values) / len(metric_values) if metric_values else None
    
    def generate_report(self) -> pd.DataFrame:
        report = {}
        metric_names = set()
        for event in self.get_events("metric"):
            if event.data is not None:
                metric_names.update([key for key in event.data.keys() if key != 'info'])
        
        for metric_name in metric_names:
            report[metric_name] = self.get_metric_avg(metric_name)
        
        # Adding final evaluation (you may implement your own method for this)
        avg_score = sum(report.values()) / len(report.values()) if report.values() else None
        if avg_score is not None:
            if avg_score > 0.8:
                report['final_evaluation'] = 'Good'
            elif avg_score > 0.5:
                report['final_evaluation'] = 'Average'
            else:
                report['final_evaluation'] = 'Poor'
        
        return pd.DataFrame(report, index=[0])

    def _normalize_completion(self, completion: str) -> str:
        completion = completion.replace("\n", " ").replace("\t", " ")
        completion = re.sub(r"\s+", " ", completion)
        completion = re.sub(r'\[[^\]]+\]', '', completion)
        completion = re.sub(r'(\d+),(\d+)', r'\1.\2', completion) # make sure percent numbers are displayed with a dot
        return completion.strip()
    
    def _normalize_ground_truth(self, ground_truth: str) -> str:
        ground_truth = ground_truth.replace("\n", " ").replace("\t", " ")
        ground_truth = re.sub(r'(\d+),(\d+)', r'\1.\2', ground_truth)
        return ground_truth.strip()
        

    def _f1_score(self, completion: str, ground_truth: List[str]) -> float:
        ground_truth_combined = " ".join([gt.lower() for gt in ground_truth])
        f1 = f1_score([ground_truth_combined], [completion.lower()], average='micro')
        print(_green(f"  F1 score: {f1}")) if f1 > 0.5 else print(_red(f"  F1 score: {f1}"))
        return f1
    
    def _vector_similarity(self, completion: str, ground_truth: List[str]) -> float:
        vectorizer = TfidfVectorizer()
        ground_truth_combined = " ".join([gt.lower() for gt in ground_truth])
        completion = completion.lower()
        vectors = vectorizer.fit_transform([ground_truth_combined, completion])
        cosine_sim = cosine_similarity(vectors[0], vectors[1])[0][0]
        print(_green(f"  Cosine similarity: {cosine_sim}")) if cosine_sim > 0.5 else print(_red(f"  Cosine similarity: {cosine_sim}"))
        return cosine_sim
        
    def _jaccard_similarity(self, completion: str, ground_truth: List[str]) -> float:
        ground_truth_tokens = [set(self._default_tokenizer.encode(gt.lower())) for gt in ground_truth]
        completion_tokens = set(self._default_tokenizer.encode(completion.lower()))
        jaccard_scores = [len(gt_tokens.intersection(completion_tokens)) / len(gt_tokens.union(completion_tokens)) for gt_tokens in ground_truth_tokens]
        avg_jaccard = sum(jaccard_scores) / len(jaccard_scores)
        print(_green(f"  Jaccard similarity: {avg_jaccard}")) if avg_jaccard > 0.5 else print(_red(f"  Jaccard similarity: {avg_jaccard}"))
        return avg_jaccard
    
    def _wer_score(self, completion: str, ground_truth: List[str]) -> float:
        ground_truth_tokens = [self._default_tokenizer.encode(gt.lower()) for gt in ground_truth]
        predicted_tokens = self._default_tokenizer.encode(completion.lower())
        wer_scores = [edit_distance(gt_tokens, predicted_tokens) / len(gt_tokens) for gt_tokens in ground_truth_tokens]
        average_wer = sum(wer_scores) / len(wer_scores)
        print(_green(f"  WER score: {average_wer}")) if average_wer < 0.5 else print(_red(f"  WER score: {average_wer}"))
        return average_wer
    
    def _meteor_score(self, completion: str, ground_truth: List[str]) -> float:
        #### TODO: Fix this
        meteor_score = meteor(ground_truth, completion)
        print(_green(f"  METEOR score: {meteor_score}")) if meteor_score > 0.5 else print(_red(f"  METEOR score: {meteor_score}"))
        return meteor_score
    
    def _bleu_score(self, completion: str, ground_truth: List[str]) -> float:
        bleu_score = bleu(ground_truth, completion)
        print(_green(f"  BLEU score: {bleu_score}")) if bleu_score > 0.5 else print(_red(f"  BLEU score: {bleu_score}"))
        return bleu_score

    def _model_graded_gpt_fact_gpt(self, query: str, completion: str, ground_truth: str) -> Tuple[str, Dict[str, Any]]:
        """
            Use Openai GPT to grade the model's completion of the query.
            Reads in the registry/model_graded/fact.yaml file as config.
        """
        INVALID_STR = "__invalid__"
        MATCH_FNS = {
            "include": lambda x, y: float(x in y),
            "exact": lambda x, y: float(x == y),
            "endswith": lambda x, y: x.endswith(y),
            "starts_or_endswith": lambda x, y: x.startswith(y) or x.endswith(y),
        }

        def get_choice(
            text: str, eval_type: str, match_fn: Union[str, Callable], choice_strings: Iterable[str]
        ) -> str:
            """Clean the answer string to a choice string to one of choice_strings. Return '__invalid__.' if no match."""
            if isinstance(match_fn, str):
                match_fn = MATCH_FNS[match_fn]
            lines = text.strip().split("\n")
            if eval_type.startswith("cot_classify"):
                lines = lines[::-1]  # reverse lines
            for line in lines:
                line = line.strip()
                line = "".join(c for c in line if c not in string.punctuation)
                if not line:
                    continue
                for choice in choice_strings:
                    if match_fn(line, choice):
                        return choice
            logging.warn(f"Choices {choice_strings} not parsable for {eval_type}: {text}")
            return INVALID_STR
        
        def get_choice_score(
            choice: str,
            choice_strings: Iterable[str],
            choice_scores: Optional[Union[dict[str, float], str]] = None,
        ) -> Optional[float]:
            if choice_scores is None:
                return None
            if choice_scores == "from_strings":
                choice_scores = {c: float(c) for c in choice_strings}
            # assumption: each INVALID_STR contributes the lowest score
            if choice == INVALID_STR:
                return min(choice_scores.values())
            return choice_scores[choice]
        
        def print_result(result: str, score: float) -> None:
            result.replace("\n", "")
            if score > 0.5:
                print(_green(f"  {result}"))
            else:
                print(_red(f"  {result}"))

        if isinstance(ground_truth, list):
            ground_truth = " ".join(ground_truth)
        with open(os.path.join(os.path.dirname(__file__), "registry/model_graded/fact.yaml")) as f:
            config = yaml.safe_load(f)
        prompt = config["fact"]["prompt"]
        prompt = prompt.replace("{input}", query).replace("{completion}", completion).replace("{ideal}", ground_truth)
        result = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            max_tokens=100,
            n=1,
            api_key=os.environ["OPENAI_API_KEY"],
            api_base=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_type="azure",
            api_version="2022-12-01"
        )['choices'][0]['text']
        eval_type = config["fact"]["eval_type"] if "eval_type" in config else "classify"
        match_fn = config["fact"]["match_fn"] if "match_fn" in config else "starts_or_endswith"
        choice_strings = config["fact"]["choice_strings"]
        choice = get_choice(result, eval_type, match_fn, choice_strings)
        score = get_choice_score(choice, choice_strings, config["fact"].get("choice_scores"))
        print_result(result, score)
        return choice, dict(
            score=score,
            sampled=[result],
            prompt=prompt,
            invalid_choice=choice == INVALID_STR,
        )

    def run(self, data: Optional[List[Tuple[Dict[str, Any], int]]] = None) -> None:
        if data is None:
            data = self.run_spec.data
            assert data is not None, "No data provided to run"

        if self.eval_type == "answer_generator":
            for sample, sample_id in data:
                query = sample["input"][0]["content"]
                print(_yellow(f"Evaluating sample ({sample_id}): {query}"))
                ground_truth = [self._normalize_ground_truth(gt) for gt in sample["ideal"]]
                result = self.run_spec.run_config["llm"](query=query, **self.run_spec.run_config)
                completion = self._normalize_completion(result.get_completions()[0])
                self.record_event("completion", {"query": query, "ground_truth": ground_truth, "completion": completion}, sample_id=sample_id)
                f1_score = self._f1_score(completion, ground_truth)
                self.record_event("metric", {"f1_score": f1_score}, sample_id=sample_id)
                vector_similarity = self._vector_similarity(completion, ground_truth)
                self.record_event("metric", {"vector_similarity": vector_similarity}, sample_id=sample_id)
                jaccard_similarity = self._jaccard_similarity(completion, ground_truth)
                self.record_event("metric", {"jaccard_similarity": jaccard_similarity}, sample_id=sample_id)
                model_graded_fact_gpt, info = self._model_graded_gpt_fact_gpt(query, completion, ground_truth)
                self.record_event("metric", {"model_graded_fact_gpt": model_graded_fact_gpt, "info": info}, sample_id=sample_id)
                bleu_score = self._bleu_score(completion, ground_truth)
                self.record_event("metric", {"bleu_score": bleu_score}, sample_id=sample_id)
                # meteor_score = self._meteor_score(completion, ground_truth)
                # self.record_event("metric", {"meteor_score": meteor_score}, sample_id=sample_id)
        elif self.eval_type == "information_retriever":
            for sample, sample_id in data:
                query = sample["input"]
                print(_yellow(f"Evaluating sample ({sample_id}): {query}"))
                result = self.run_spec.run_config["retriever"](query=query, **self.run_spec.run_config)
                print(result)

