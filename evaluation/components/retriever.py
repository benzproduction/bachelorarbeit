from typing import Any
import numpy as np
import pandas as pd
from base import EmbeddingsFn, RetrieverFn
from ast import literal_eval
from redis import Redis
from redis.commands.search.query import Query


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
        self.redisClient = Redis(host=host, port=port, db=db, decode_responses=False, password=password)
        assert self.redisClient.ping(), "Redis is not connected"
        self.index = index
        assert self.redisClient.ft(index).info()['num_docs'] > 0, "Index is empty"
        self.vector_field_name = vector_field_name

    def __call__(self, query: str, embedder: EmbeddingsFn, k: int = 5, **_kwargs: Any) -> pd.DataFrame:
        embedded_query = np.array(embedder(query), dtype=np.float32).tobytes()
        q = Query(f'*=>[KNN {k} @{self.vector_field_name } $vec_param AS vector_score]').sort_by('vector_score').paging(0,k).return_fields('vector_score','filename','text_chunk').dialect(2) 
        params_dict = {"vec_param": embedded_query}
        results = self.redisClient.ft(self.index).search(q, query_params = params_dict)

        if results.total == 0:
            return pd.DataFrame()

        query_result_list = []
        for i, result in enumerate(results.docs):
            result_order = i
            text = result.text_chunk
            score = result.vector_score
            filename = result.id
            query_result_list.append((result_order, text, score, filename))
        
        result_df = pd.DataFrame(query_result_list)
        result_df.columns = ['id', 'text', 'certainty', 'filename']
        return result_df
    
    def __repr__(self):
        return f"RedisRetriever@{self.index}"

        

    
