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
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.metrics import edit_distance
from nltk.translate import meteor, bleu
import tiktoken
import yaml
import os
import openai
import string
import copy
import numpy as np
import statistics

logger = logging.getLogger(__name__)
MIN_FLUSH_EVENTS = 100
MIN_FLUSH_SECONDS = 10
INVALID_STR = "__invalid__"
MATCH_FNS = {
    "include": lambda x, y: float(x in y),
    "exact": lambda x, y: float(x == y),
    "endswith": lambda x, y: x.endswith(y),
    "starts_or_endswith": lambda x, y: x.startswith(y) or x.endswith(y),
}

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
    category: Optional[str] = None


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
            spec = asdict(copy.deepcopy(run_spec))
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
    
    def record_event(self, type, data=None, sample_id=None, category=None):
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
                category=category,
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
    
    def get_metric_avg(self, metric_name: str, category: Optional[str] = None) -> Optional[float]:
        metric_values = []
        for event in self.get_events("metric"):
            if event.data is not None and metric_name in event.data.keys() and event.category == category:
                metric_info = event.data[metric_name]
                if isinstance(metric_info, str):
                    metric_values.append(event.data["info"]["score"])
                elif isinstance(metric_info, (int, float)):
                    metric_values.append(metric_info)
        return statistics.mean(metric_values) if len(metric_values) > 0 else None
    
    def generate_report(self) -> pd.DataFrame:
        """ Generate a report for the current run.
        Will generate a matrix of metrics x categories.
        @Example:

        |            | Metric 1 | Metric 2 | Metric 3 | 
        |------------|----------|----------|----------| 
        | Category 1 |    0.5   |   0.6    |    0.7   |
        | Category 2 |    0.5   |   0.6    |    0.7   |
        | Overall    |    0.5   |   0.6    |    0.7   |
        """
        metric_avgs = {}

        metric_names = set()
        category_names = set()
        for event in self.get_events("metric"):
            if event.data is not None:
                metric_names.update([key for key in event.data.keys() if key != 'info'])
            if event.category is not None:
                category_names.add(event.category)
        
        for metric_name in metric_names:
            for category_name in category_names:
                avg = self.get_metric_avg(metric_name, category_name)
                if avg is not None:
                    if metric_name not in metric_avgs:
                        metric_avgs[metric_name] = {}
                    metric_avgs[metric_name][category_name] = avg
        
        for metric_name in metric_names:
            metric_values = [metric_avgs[metric_name].get(category_name, 0) for category_name in category_names]
            metric_avgs[metric_name]['Overall'] = statistics.mean(metric_values)

        return pd.DataFrame(metric_avgs)
    
    def _check_retriever_result(self, result: pd.DataFrame, ideal: Dict[str, str]) -> bool:
        """
        Checks if the result of the retriever is correct, but very simple. (Checks how many of the ideal sentences are in the result)
        """
        paragraphs = ideal.get("paragraph", [])
        filenames = ideal.get("filename", [])
        if isinstance(paragraphs, List):
            paragraphs = " ".join(paragraphs)
        if isinstance(filenames, str):
            filenames = [filenames]
        combined_text = ""
        # inter the rows of the result and check if the filename is in the ideal filenames
        for row in result.itertuples():
            if row.filename in filenames:
                combined_text += f" {row.text}"
        text_tokens = self._default_tokenizer.encode(combined_text)
        paragraphs_tokens = self._default_tokenizer.encode(paragraphs)
        text_token_set = set(text_tokens)
        paragraph_token_set = set(paragraphs_tokens)
        intersection_ratio = len(paragraph_token_set.intersection(text_token_set)) / len(paragraph_token_set)
        print(_green(f"  Intersection ratio: {intersection_ratio}")) if intersection_ratio >= 0.9 else print(_red(f"  Intersection ratio: {intersection_ratio}"))
        return intersection_ratio >= 0.9
    
    def _fscore_retriever(self, result_ids: List[str], ideal_ids: List[str]) -> float:
        def flatten_nested_list(nested_list: List) -> List:
            if any(isinstance(item, list) for item in nested_list):
                flattened_list = []
                for sublist in nested_list:
                    if isinstance(sublist, list):
                        flattened_list.extend(sublist)
                    else:
                        flattened_list.append(sublist)
                return flattened_list
            else:
                return nested_list
        
        returned_set = set(flatten_nested_list(result_ids))
        ideal_set = set(flatten_nested_list(ideal_ids))
        true_positives = len(ideal_set.intersection(returned_set))
        false_positives = len(returned_set.difference(ideal_set))
        false_negatives = len(ideal_set.difference(returned_set))
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        try:
            fscore = 2 * (precision * recall) / (precision + recall)
            print(_green(f"  F1 score: {fscore}")) if fscore > 0.5 else print(_red(f"  F1 score: {fscore}"))
            return 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            print(_red("  F1 score: 0.0"))
            return 0.0
    
    def _precision_at_k(self, result_ids: List[str], ideal_ids: List[str], k: int = 5) -> float:
        def flatten_nested_list(nested_list: List) -> List:
            if any(isinstance(item, list) for item in nested_list):
                flattened_list = []
                for sublist in nested_list:
                    if isinstance(sublist, list):
                        flattened_list.extend(sublist)
                    else:
                        flattened_list.append(sublist)
                return flattened_list
            else:
                return nested_list
        
        result_ids = flatten_nested_list(result_ids)
        if len(result_ids) < k:
            print(_red(f"  Precision@{k}: 0.0"))
            return 0.0
        else:
            pAtk = len(set(result_ids[:k]).intersection(set(ideal_ids))) / k
            print(_green(f"  Precision@{k}: {pAtk}")) if pAtk > 0.5 else print(_red(f"  Precision@{k}: {pAtk}"))
            return pAtk

    def _compute_ndcg(self, result_ids: List[str], ideal_ids: List[str]) -> float:
        """
        Metric used to evaluate the retrieval of documents.

        While traditionally used for ranking, it can be adapted 
        by considering all correct documents equally important (i.e., giving them equal 'gain') 
        and simply calculating the 'discount' based on the position in the list.
        """
        relevant_set = set(ideal_ids)
        dcg = sum([1.0 / (idx+1) if doc_id in relevant_set else 0.0 for idx, doc_id in enumerate(result_ids)])
        idcg = sum([1.0 / (idx+1) for idx in range(len(ideal_ids))])
        ndcg = dcg / idcg
        print(_green(f"  NDCG: {ndcg}")) if ndcg > 0.5 else print(_red(f"  NDCG: {ndcg}"))
        return ndcg

    def _compute_modified_precision(self, result_ids: List[str], ideal_ids: List[str]) -> float:
        """
        Modified precision metric where the number of correctly retrieved documents gets divided by the total number of documents that should have been retrieved, 
        instead of the total number of retrieved documents. This will give a 1.0 score when the retriever gets all correct documents.

        --> The total number of retrieved documents is not the denominator, instead the total number of relevant documents is the denominator.
        """
        retrieved_set = set(result_ids)
        relevant_set = set(ideal_ids)
        true_positive = len(relevant_set.intersection(retrieved_set))
        precision = true_positive / len(relevant_set)
        print(_green(f"  Modified precision: {precision}")) if precision > 0.5 else print(_red(f"  Modified precision: {precision}"))
        return precision

    def _compute_matthew_corr(self, confusion_matrix: np.ndarray) -> float:
        assert confusion_matrix.shape == (2, 3), f"Got shape: {confusion_matrix.shape}"
        r = confusion_matrix[:, :2]
        r[:, 0] += confusion_matrix[:, 2]
        return (r[1, 1] * r[0, 0] - r[1, 0] * r[0, 1]) / np.sqrt(
            r[1, :].sum() * r[0, :].sum() * r[:, 0].sum() * r[:, 1].sum()
        )
    def _compute_precision(self, confusion_matrix: np.ndarray, idx: int = 0) -> float:
        return confusion_matrix[idx, idx] / confusion_matrix[:, idx].sum()
    
    def _compute_recall(self, confusion_matrix: np.ndarray, idx: int = 0) -> float:
        return confusion_matrix[idx, idx] / confusion_matrix[idx, :].sum()
    
    def _compute_f_score(self, confusion_matrix: np.ndarray, idx: int = 0, beta: float = 1.0) -> float:
        precision = self._compute_precision(confusion_matrix, idx=idx)
        recall = self._compute_recall(confusion_matrix, idx=idx)
        return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
    
    def _compute_averaged_f_score(self, confusion_matrix: np.ndarray, average: str = "macro", beta: float = 1.0) -> float:
        assert average in ["macro"]
        f_scores = []
        for i in range(confusion_matrix.shape[0]):
            f_scores.append(self._compute_f_score(confusion_matrix, idx=i, beta=beta))
        return np.array(f_scores).mean()

    def _normalize_completion(self, completion: str) -> str:
        completion = re.sub(r"\s+", " ", completion)
        completion = re.sub(r'\[[^\]]+\]', '', completion)
        completion = re.sub(r'(\d+),(\d+)', r'\1.\2', completion) # make sure percent numbers are displayed with a dot
        return completion.strip()
    
    def _normalize_ground_truth(self, ground_truth: str) -> str:
        ground_truth = ground_truth.replace("\n", " ").replace("\t", " ")
        ground_truth = re.sub(r'(\d+),(\d+)', r'\1.\2', ground_truth)
        return ground_truth.strip()
    
    def _normalize_general(self, s: str) -> str:
        """Lower text and remove punctuation, articles and extra whitespace."""
        s = s.lower()
        exclude = set(string.punctuation)
        s = "".join(char for char in s if char not in exclude)
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = " ".join(s.split())
        return s
    
    def _fuzzy_match(self, s1: str, s2: str) -> bool:
        s1 = self._normalize_general(s1)
        s2 = self._normalize_general(s2)
        if s1 == "" or s2 == "":
            return s1 == s2
        match = s1 in s2 or s2 in s1
        print(_green(f"  Fuzzy match: {match}")) if match else print(_red(f"  Fuzzy match: {match}"))
        return match
        

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
    
    def _mg_get_choice(self, text: str, eval_type: str, match_fn: Union[str, Callable], choice_strings: Iterable[str]
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
    
    def _mg_get_choice_score(
            self, 
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

    def _model_graded_gpt_fact_gpt(self, query: str, completion: str, ground_truth: str) -> Tuple[str, Dict[str, Any]]:
        """
            Use Openai GPT to grade the model's completion of the query.
            Reads in the registry/model_graded/fact.yaml file as config.
        """
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
        choice = self._mg_get_choice(result, eval_type, match_fn, choice_strings)
        score = self._mg_get_choice_score(choice, choice_strings, config["fact"].get("choice_scores"))
        print_result(result, score)
        return choice, dict(
            score=score,
            sampled=[result],
            prompt=prompt,
            invalid_choice=choice == INVALID_STR,
        )

    def _model_graded_gpt_closedqa(self, question: str, ideal_answer: str, context: str, completion: str) -> Tuple[str, Dict[str, Any]]:
        """
            Use Openai GPT to grade the model's completion of the query.
            Reads in the registry/model_graded/closedqa.yaml file as config.
        """
        if isinstance(context, list):
            if all(isinstance(c, dict) for c in context):
                context = " ".join(c["text"] for c in context)
            else:
                context = " ".join(context)
        assert isinstance(context, str), f"Context must be a string, not {type(context)}"

        with open(os.path.join(os.path.dirname(__file__), "registry/model_graded/closedqa.yaml")) as f:
            config = yaml.safe_load(f)
        prompt = config["closedqa"]["prompt"]
        prompt = prompt.replace("{input}", question).replace("{completion}", completion).replace("{ideal}", ideal_answer).replace("{context}", context)
        result = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            max_tokens=1200,
            n=1,
            api_key=os.environ["OPENAI_API_KEY"],
            api_base=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_type="azure",
            api_version="2022-12-01"
        )['choices'][0]['text']
        eval_type = config["closedqa"]["eval_type"] if "eval_type" in config["closedqa"] else "classify"
        match_fn = config["closedqa"]["match_fn"] if "match_fn" in config["closedqa"] else "starts_or_endswith"
        choice_strings = config["closedqa"]["choice_strings"]
        choice = self._mg_get_choice(result, eval_type, match_fn, choice_strings)
        score = self._mg_get_choice_score(choice, choice_strings, config["closedqa"].get("choice_scores"))
        # print score, that is either 1.0 or 0.0
        print(_green(f"  Model graded GPT: {result}") if score > 0.5 else _red(f"  Model graded GPT: {result}"))
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

        if self.eval_type == "end_to_end":
            for sample, sample_id in data:
                query = sample["input"]
                if isinstance(query, dict) and "question" in query:
                    query = query["question"]
                assert isinstance(query, str), f"Query must be a string, not {type(query)}"
                category = sample["category"] if "category" in sample else "NOT_SPECIFIED"
                print(_yellow(f"Evaluating sample ({sample_id}): {query}"))
                ground_truth = [self._normalize_ground_truth(gt) for gt in sample["ideal"]]
                result = self.run_spec.run_config["llm"](query=query, **self.run_spec.run_config)
                completion = self._normalize_completion(result.get_completions()[0])
                self.record_event("result", {"query": query, "ground_truth": ground_truth, "completion": completion}, sample_id=sample_id, category=category)
                # compute metrics
        elif self.eval_type == "answer_generator":
            for sample, sample_id in data:
                assert "input" in sample and "ideal" in sample and "sources" in sample["input"] and "answer" in sample["ideal"], \
                    "Incomplete or missing input data"
                category = sample["category"] if "category" in sample else "NOT_SPECIFIED"
                query = sample["input"]
                if isinstance(query, dict) and "question" in query:
                    query = query["question"]
                sources = sample["input"]["sources"]
                print(_yellow(f"Evaluating sample ({sample_id}): {query}"))
                result = self.run_spec.run_config["llm"](query=query, sources=sources, **self.run_spec.run_config)
                result: str = result.get_completions()[0]
                ideal = sample["ideal"]
                if isinstance(ideal, dict) and "answer" in ideal:
                    ideal = ideal["answer"]
                self.record_event("result", {"query": query, "sources": sources, "completion": result, "ideal": ideal}, sample_id=sample_id, category=category)
                # compute metrics
                fuzzy_match = self._fuzzy_match(result, ideal)
                self.record_event("metric", {"fuzzy_match": fuzzy_match}, sample_id=sample_id, category=category)
                model_graded_qa_gpt, info = self._model_graded_gpt_closedqa(query, ideal, sources, result)
                self.record_event("metric", {"model_graded_qa_gpt": model_graded_qa_gpt, "info": info}, sample_id=sample_id, category=category)

        elif self.eval_type == "information_retriever":
            for sample, sample_id in data:
                assert "input" in sample and "ideal" in sample and "paragraph" in sample["ideal"] and "filename" in sample["ideal"], \
                    "Incomplete or missing input data"
                category = sample["category"] if "category" in sample else "NOT_SPECIFIED"
                query = sample["input"]
                if isinstance(query, dict) and "question" in query:
                    query = query["question"]
                ideal = sample["ideal"] 
                print(_yellow(f"Evaluating sample ({sample_id}): {query}"))
                result: pd.DataFrame = self.run_spec.run_config["retriever"](query=query, **self.run_spec.run_config)
                assert "text" in result.columns and "filename" in result.columns, "Incomplete or missing result data"
                self.record_event("result", result[["text", "filename"]].to_dict(orient="records"), sample_id=sample_id, category=category)
                # compute metrics
                if "ids" in ideal and "key" in result.columns:
                    ideal_ids = ideal["ids"]
                    result_ids = result[['key']].values.tolist()
                    if isinstance(result_ids[0], list):
                        result_ids = [item for sublist in result_ids for item in sublist]
                    ndcg = self._compute_ndcg(result_ids, ideal_ids)
                    self.record_event("metric", {"ndcg": ndcg}, sample_id=sample_id, category=category)
                    precision = self._compute_modified_precision(result_ids, ideal_ids)
                    self.record_event("metric", {"precision": precision}, sample_id=sample_id, category=category)
                else: 
                    correct = self._check_retriever_result(result[["text", "filename"]], ideal)
                    self.record_event("metric", {"correct": correct}, sample_id=sample_id, category=category)
