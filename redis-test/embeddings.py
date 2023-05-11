import openai
from openai.embeddings_utils import get_embedding
import os
import tiktoken
from database import load_vectors, get_redis_connection, save_chunks
from config import TEXT_EMBEDDING_CHUNK_SIZE, INDEX_NAME, VECTOR_FIELD_NAME, PREFIX
from redis.commands.search.field import (
    TextField,
    VectorField,
    NumericField
    )
from redis.commands.search.indexDefinition import (
        IndexDefinition,
        IndexType
    )
from redis import Redis
from tqdm import tqdm
from clean_text import clean_text
from numpy import array, average
from helpers import get_env
import PyPDF2
import textract
from typing import Dict
from models import Document, DocumentChunk, DocumentChunkMetadata

API_KEY, RESOURCE_ENDPOINT = get_env("azure-openai")

openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = RESOURCE_ENDPOINT
openai.api_version = "2022-12-01"


pdf_dir = '/Users/shuepers001/dev/bachelorarbeit/data/raw/rest_pdfs'
pdf_files = sorted([x for x in os.listdir(pdf_dir) if 'DS_Store' not in x])

tokenizer = tiktoken.get_encoding("cl100k_base")

def create_redis_index(redis_conn:Redis):
    VECTOR_DIM = 1536 #len(data['title_vector'][0]) # length of the vectors
    #VECTOR_NUMBER = len(data)                 # initial number of vectors
    DISTANCE_METRIC = "COSINE"                # distance metric for the vectors (ex. COSINE, IP, L2)
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
    redis_conn.ft(INDEX_NAME).create_index(
        fields = fields,
        definition = IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH)
    )
    return print(f"Index ({INDEX_NAME}) was created.")

redis_conn = get_redis_connection(password="weak")
assert redis_conn.ping() == True, "Redis connection not working"
try:
    redis_conn.ft(INDEX_NAME).info()
    print("Index already exists")
    print(f"Docs in index: {redis_conn.ft(INDEX_NAME).info()['num_docs']}")
    # exit()
    # optional extra step: deleting & recreating
    # print("deleting index...")
    # redis_conn.ft(INDEX_NAME).dropindex(delete_documents=True)
    # create_redis_index(redis_conn)
except Exception as e:
    create_redis_index(redis_conn)
    assert redis_conn.ft(INDEX_NAME).info() != None, "Index not created"

def pdf_to_text_map(pdf_path):
    # Read the PDF
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)

        # Iterate through pages
        page_texts = {}
        for i in tqdm(range(len(reader.pages))):
            # Extract the page
            page = reader.pages[i]

            # Save the page as a temporary PDF
            with open("temp.pdf", "wb") as output:
                writer = PyPDF2.PdfWriter()
                writer.add_page(page)
                writer.write(output)

            # Use textract to extract text from the temporary PDF
            text = textract.process("temp.pdf", method='pdfminer', encoding='utf-8').decode()

            # Store the extracted text
            page_texts[i] = text

            # Remove the temporary PDF
            os.remove("temp.pdf")

    return page_texts

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

def find_p_num(chunk:str, m: Dict[int, str], t: tiktoken.Encoding) -> int:
    """
    Find the most likely page number of a given text chunk in a page map.

    Args:
        - chunk (str): The text chunk to find in the page map.
        - m (Dict[int, str]): The page map with page numbers as keys and their text as values.
        - t (Tokenizer): The tokenizer used to tokenize the text. Only tested with the tiktoken library.
    
    Returns:
        - int: The most likely page number of the given text chunk.
    """
    c = set(t.encode(chunk))
    s = {n: len(c.intersection(set(t.encode(txt)))) for n, txt in m.items()}
    return max(s, key=s.get)

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



for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_dir, pdf_file)
    print("Creating text map for: ", pdf_file)
    page_texts = pdf_to_text_map(pdf_path)
    text = "\n".join(page_texts.values())
    text = clean_text(text)
    token_chunks = list(chunks(text, TEXT_EMBEDDING_CHUNK_SIZE, tokenizer))
    text_chunks = [tokenizer.decode(chunk) for chunk in token_chunks]
    print("Splitted text into {} chunks".format(len(text_chunks)))
    embeddings = []
    for text_chunk in tqdm(text_chunks):
        embeddings.append(get_embedding(text_chunk, engine='text-embedding-ada-002'))

    text_embeddings = list(zip(text_chunks, embeddings))

    vectors = []
    for i, (text_chunk, embedding) in enumerate(text_embeddings):
        id = get_unique_id_for_file_chunk(pdf_file, i)
        metadata = DocumentChunkMetadata(
            source_filename=pdf_file,
            page=find_p_num(text_chunk, page_texts, tokenizer),
        )
        chunk = DocumentChunk(
            id=id,
            text=text_chunk,
            embedding=embedding,
            metadata=metadata,
        )
        vectors.append(chunk)
    save_chunks(r=redis_conn, vectors=vectors, index=INDEX_NAME)
    print(f"Saved {len(vectors)} chunks to index {INDEX_NAME}")
