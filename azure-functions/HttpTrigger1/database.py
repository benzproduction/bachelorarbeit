from redis import Redis
from typing import List
from .models import DocumentChunk
from redis.commands.search.field import (
    TextField,
    VectorField,
    NumericField
    )
from redis.commands.search.indexDefinition import (
        IndexDefinition,
        IndexType
    )
import numpy as np
import logging

VECTOR_FIELD_NAME='content_vector'
VECTOR_DIM = 1536 
DISTANCE_METRIC = "COSINE"

# Get a Redis connection
def get_redis_connection(host='localhost',port='6379',db=0):
    
    r = Redis(host=host, port=port, db=db,decode_responses=False)
    return r

def save_vectors(vectors: List[DocumentChunk], index: str, delete_index = False) -> None:
    """
    Saves the vectors to Redis
    """
    r = get_redis_connection()
    assert r.ping(), "Redis is not connected"
    logging.info("Successfully connected to Redis")
    try:
        r.ft(index).info()
        logging.info(f"Index {index} already exists")
        logging.info(f"Docsin index: {r.ft(index).info()['num_docs']}")
        if delete_index:
            logging.info(f"Deleting index {index}")
            r.ft(index).dropindex(delete_documents=True)
            logging.info(f"Creating index {index}")
            create_redis_index(r,index)
    except Exception as e:
        logging.info(f"Index {index} does not exist. Creating it now")
        create_redis_index(r,index)
    p = r.pipeline(transaction=False)
    # load vectors
    for vector in vectors:
        #hash key
        key=f"{index}:{vector.id}"
        item_metadata = {}
        item_metadata["filename"] = vector.metadata.document_metadata.source_filename
        item_metadata["text_chunk"] = vector.text
        item_metadata["page"] = vector.metadata.page
        item_keywords_vector = np.array(vector.embedding,dtype= 'float32').tobytes()
        item_metadata[VECTOR_FIELD_NAME]=item_keywords_vector

        # HSET
        r.hset(key,mapping=item_metadata)
    p.execute()





def create_redis_index(redis_conn:Redis, index_name:str):
    filename = TextField("filename")
    text_chunk = TextField("text_chunk")
    page = NumericField("page")
    text_embedding = VectorField(VECTOR_FIELD_NAME,
        "HNSW", {
            "TYPE": "FLOAT32",
            "DIM": VECTOR_DIM,
            "DISTANCE_METRIC": DISTANCE_METRIC
        }
    )
    fields = [filename,text_chunk,page,text_embedding]
    redis_conn.ft(index_name).create_index(
        fields = fields,
        definition = IndexDefinition(prefix=[index_name], index_type=IndexType.HASH)
    )


# # Create a Redis pipeline to load all the vectors and their metadata
# def load_vectors(client:Redis, input_list, vector_field_name):
#     p = client.pipeline(transaction=False)
#     for text in input_list:    
#         #hash key
#         key=f"{PREFIX}:{text['id']}"
        
#         #hash values
#         item_metadata = text['metadata']
#         #
#         item_keywords_vector = np.array(text['vector'],dtype= 'float32').tobytes()
#         item_metadata[vector_field_name]=item_keywords_vector
        
#         # HSET
#         p.hset(key,mapping=item_metadata)
            
#     p.execute()