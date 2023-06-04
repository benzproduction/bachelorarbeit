from typing import Any
import numpy as np
import pandas as pd
from base import EmbeddingsFn, RetrieverFn
from ast import literal_eval
from redis import Redis
from redis.commands.search.query import Query
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification


def load_embeddings(embeddings_and_text_path: str):
    df = pd.read_csv(embeddings_and_text_path, converters={"embedding": literal_eval})
    assert (
        "text" in df.columns and "embedding" in df.columns
    ), "The embeddings file must have columns named 'text' and 'embedding'"
    return df

def find_top_k_closest_embeddings(embedded_prompt: list[float], embs: list[list[float]], k: int):
    # Normalize the embeddings
    norm_embedded_prompt = embedded_prompt / np.linalg.norm(embedded_prompt)
    norm_embs = embs / np.linalg.norm(embs, axis=1)[:, np.newaxis]

    # Calculate cosine similarity
    cosine_similarities = np.dot(norm_embs, norm_embedded_prompt)

    # Get the indices of the top k closest embeddings
    top_k_indices = np.argsort(cosine_similarities)[-k:]

    return top_k_indices[::-1]




class CSVRetriever(RetrieverFn):
    def __init__(self, embeddings_and_text_path: str, **_kwargs: Any) -> None:
        """
        Args:
            embeddings_and_text_path: The path to a CSV containing "text" and "embedding" columns.
            _kwargs: Additional arguments.
        """
        self.embeddings_df = load_embeddings(embeddings_and_text_path)

    def __call__(self, query: str, embedder: EmbeddingsFn, k: int = 5, **_kwargs: Any) -> pd.DataFrame:
        embedded_query = embedder(query)
        embs = self.embeddings_df["embedding"].to_list()
        return self.embeddings_df.iloc[
                find_top_k_closest_embeddings(embedded_query, embs, k)
            ]#.text.values
    
    def __repr__(self):
        return f"KNNRetriever"
    
class RedisRetriever(RetrieverFn):
    def __init__(
        self,
        index: str,
        password: str = None,
        host: str = 'localhost',
        port: str = '6379',
        db: int = 0,
        vector_field_name: str = 'content_vector',
        **_kwargs: Any
    ) -> None:
        self.index = index
        self.password = password
        self.host = host
        self.port = port
        self.db = db
        self.vector_field_name = vector_field_name
        self.redisClient = None

    def _connect(self) -> None:
        if self.redisClient is None:
            self.redisClient = Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=False,
                password=self.password
            )
            assert self.redisClient.ping(), "Redis is not connected"
            assert int(self.redisClient.ft(self.index).info()['num_docs']) > 0, "Index is empty"

    def __call__(self, query: str, embedder: EmbeddingsFn, k: int = 5, **_kwargs: Any) -> pd.DataFrame:
        self._connect()

        embedded_query = np.array(embedder(query), dtype=np.float32).tobytes()
        q = Query(f'*=>[KNN {k} @{self.vector_field_name } $vec_param AS vector_score]').sort_by('vector_score').paging(0,k).return_fields('vector_score','filename','text_chunk').dialect(2) 
        params_dict = {"vec_param": embedded_query}
        results = self.redisClient.ft(self.index).search(q, query_params=params_dict)

        if results.total == 0:
            return pd.DataFrame()

        query_result_list = []
        for i, result in enumerate(results.docs):
            result_order = i
            text = result.text_chunk
            score = result.vector_score
            filename = result.filename
            key = result.id
            query_result_list.append((result_order, key, text, score, filename))
        
        result_df = pd.DataFrame(query_result_list)
        result_df.columns = ['id', 'key', 'text', 'certainty', 'filename']
        return result_df

    def __repr__(self):
        return f"RedisRetriever@{self.index}"
    
class RankedRedisRetriever(RetrieverFn):
    def __init__(
        self,
        index: str,
        reranking_k: int = 100,
        password: str = None,
        host: str = 'localhost',
        port: str = '6379',
        db: int = 0,
        vector_field_name: str = 'content_vector',
        **_kwargs: Any
    ) -> None:
        self.index = index
        self.password = password
        self.host = host
        self.port = port
        self.db = db
        self.vector_field_name = vector_field_name
        self.reranking_k = reranking_k
        self.redisClient = None
        self.tokenizer = AutoTokenizer.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")
        self.model = AutoModelForSequenceClassification.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")

    def _connect(self) -> None:
        if self.redisClient is None:
            self.redisClient = Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=False,
                password=self.password
            )
            assert self.redisClient.ping(), "Redis is not connected"
            assert int(self.redisClient.ft(self.index).info()['num_docs']) > 0, "Index is empty"
    
    def __call__(self, query: str, embedder: EmbeddingsFn, k: int = 5, **_kwargs: Any) -> pd.DataFrame:
        self._connect()

        reranking_k = _kwargs.get('reranking_k', self.reranking_k)
        embedded_query = np.array(embedder(query), dtype=np.float32).tobytes()
        q = Query(f'*=>[KNN {reranking_k} @{self.vector_field_name } $vec_param AS vector_score]').sort_by('vector_score').paging(0,reranking_k).return_fields('vector_score','filename','text_chunk').dialect(2) 
        params_dict = {"vec_param": embedded_query}
        results = self.redisClient.ft(self.index).search(q, query_params=params_dict)

        if results.total == 0:
            return pd.DataFrame()
        
        pairs = [(query, document.text_chunk) for document in results.docs]
        inputs = self.tokenizer(pairs, max_length=512, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        scores = logits[:, 0].tolist()
        ranked_docs = sorted(zip(results.docs, scores), key=lambda x: x[1], reverse=True)
        top_k_documents = [doc for doc, score in ranked_docs[:k]]

        query_result_list = []
        for i, result in enumerate(top_k_documents):
            result_order = i
            text = result.text_chunk
            vector_score = result.vector_score
            reranking_score = ranked_docs[i][1]
            filename = result.filename
            key = result.id
            query_result_list.append((result_order, key, text, vector_score, reranking_score, filename))
        
        result_df = pd.DataFrame(query_result_list)
        result_df.columns = ['id', 'key', 'text', 'certainty', 'reranking_score', 'filename']
        return result_df




        
if __name__ == "__main__":
    import os
    from embeddings import OpenAI_Ada002
    embedder = OpenAI_Ada002(api_key=os.environ["OPENAI_API_KEY"], api_base=os.environ["AZURE_OPENAI_ENDPOINT"])
    retriever = RankedRedisRetriever(index="real_estate_index", password="weak")
    result = retriever('Has the affordability of housing dropped in recent years, affecting both young and mid-career households?', embedder, k=5, reranking_k=75)
    print(result["key"].to_list())
    print(result["reranking_score"].to_list())
    ids = ["real_estate_index:Emerging-Trends_USCanada-2023.pdf-!71", "real_estate_index:bouwinvest_international-market-outlook_2023-2025-1.pdf-!3"]