import openai
from openai.embeddings_utils import get_embedding
import os
import tiktoken
import textract
from database import load_vectors, get_redis_connection
from config import TEXT_EMBEDDING_CHUNK_SIZE, INDEX_NAME, VECTOR_FIELD_NAME
from clean_text import clean_text
from numpy import array, average
from helpers import get_env

API_KEY, RESOURCE_ENDPOINT = get_env("azure-openai")

openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = RESOURCE_ENDPOINT
openai.api_version = "2022-12-01"


data_dir = 'data/raw/txts'
txt_files = sorted([x for x in os.listdir(data_dir) if 'DS_Store' not in x])

tokenizer = tiktoken.get_encoding("cl100k_base")

redis_conn = get_redis_connection()
redis_conn.ft(INDEX_NAME)

# Split a text into smaller chunks of size n, preferably ending at the end of a sentence
def chunks(text, n, tokenizer):
    tokens = tokenizer.encode(text)
    """Yield successive n-sized chunks from text."""
    i = 0
    while i < len(tokens):
        # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j

def get_col_average_from_list_of_lists(list_of_lists):
    """Return the average of each column in a list of lists."""
    if len(list_of_lists) == 1:
        return list_of_lists[0]
    else:
        list_of_lists_array = array(list_of_lists)
        average_embedding = average(list_of_lists_array, axis=0)
        return average_embedding.tolist()

def get_unique_id_for_file_chunk(filename, chunk_index):
    return str(filename+"-!"+str(chunk_index))

start_doc = 1
end_doc = 4
txt_files = txt_files[start_doc:end_doc]

for txt_file in txt_files:
    
    txt_path = os.path.join(data_dir,txt_file)
    print(txt_path)
    text = open(txt_path, 'r').read()
    text = clean_text(text)


    filename = txt_file.split('.')[0]
    text_to_embed = "Filename is: {}; {}".format(
        filename, text)
    

    token_chunks = list(chunks(text_to_embed, TEXT_EMBEDDING_CHUNK_SIZE, tokenizer))
    text_chunks = [tokenizer.decode(chunk) for chunk in token_chunks]

    embeddings = [get_embedding(text_chunk, engine = 'text-embedding-ada-002') for text_chunk in text_chunks]

    text_embeddings = list(zip(text_chunks, embeddings))

    average_embedding = get_col_average_from_list_of_lists(embeddings)

    # Get the vectors array of triples: file_chunk_id, embedding, metadata for each embedding
    # Metadata is a dict with keys: filename, file_chunk_index
    vectors = []
    for i, (text_chunk, embedding) in enumerate(text_embeddings):
        id = get_unique_id_for_file_chunk(filename, i)
        vectors.append(({'id': id
                         , "vector": embedding, 'metadata': {"filename": filename
                                                              , "text_chunk": text_chunk
                                                              , "file_chunk_index": i}}))

    try:
        load_vectors(redis_conn, vectors,VECTOR_FIELD_NAME)

    except Exception as e:
        print(f'Ran into a problem uploading to Redis: {e}')
