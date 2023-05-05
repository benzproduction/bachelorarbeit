from redis import Redis
from typing import List
from .models import DocumentChunk

# Get a Redis connection
def get_redis_connection(host='localhost',port='6379',db=0):
    
    r = Redis(host=host, port=port, db=db,decode_responses=False)
    return r

def save_vectors(vectors: List[DocumentChunk]) -> None:
    """
    Saves the vectors to Redis
    """
    r = get_redis_connection()
    for vector in vectors:
        r.set(vector.id, vector.vector)