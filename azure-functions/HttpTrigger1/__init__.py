import logging

import azure.functions as func
from .parser import pdf_to_text_mapping, get_metadata, pdf_to_text
from .chunker import split_text__chunk_default, split_text__chunk_page
from .clean_text import clean_text
import io
import mimetypes
from .models import Document, DocumentChunk, DocumentChunkMetadata
import tiktoken
import json

def unique_id_from_filename(index:str, filename: str) -> str:
    return f"{index}-{filename}"


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processing a request.')

    index = req.params.get('index')
    if not index:
        return func.HttpResponse(
          "Please pass an index on the query string or in the request body",
            status_code=400
        )
    
    parse_method = req.params.get('parse_method', 'chunk_default') # options are 'chunk_default' and 'chunk_page'

    TEXT_EMBEDDING_CHUNK_SIZE = req.params.get('chunk_size', 300)
    try:
        TEXT_EMBEDDING_CHUNK_SIZE = int(TEXT_EMBEDDING_CHUNK_SIZE)
        if TEXT_EMBEDDING_CHUNK_SIZE < 100 or TEXT_EMBEDDING_CHUNK_SIZE > 1000:
            raise ValueError
    except:
        return func.HttpResponse(
            "Please pass a valid chunk_size on the query string (100 <= chunk_size <= 1000)",
            status_code=400
        )

    for input_file in req.files.values():
        filename = input_file.filename
        logging.info(f"File {filename}")
        mime_type, encoding = mimetypes.guess_type(filename)
        contents = input_file.stream.read()
        try:
            file_stream = io.BytesIO(contents)
            if mime_type == 'application/pdf':
                metadata = get_metadata(file_stream, mime_type, filename)
                document = Document(
                    id=unique_id_from_filename(index, filename),
                    metadata=metadata
                )
                if parse_method == 'chunk_default':
                    tokenizer = tiktoken.get_encoding("cl100k_base")
                    text = pdf_to_text(file_stream)
                    text = clean_text(text)
                    token_chunks = list(split_text__chunk_default(text, TEXT_EMBEDDING_CHUNK_SIZE, tokenizer))
                    text_chunks = [tokenizer.decode(chunk) for chunk in token_chunks]

                    return func.HttpResponse(
                        json.dumps(text_chunks),
                        status_code=200,
                        mimetype="application/json"
                    )
                    embeddings = []
                    # for text_chunk in text_chunks:
                    #     embeddings.append(get_embedding(text_chunk, engine='text-embedding-ada-002'))

                    text_embeddings = list(zip(text_chunks, embeddings))

                    # vectors = []
                    # for i, (text_chunk, embedding) in enumerate(text_embeddings):
                    #     id = get_unique_id_for_file_chunk(filename, i)
                    #     vectors.append(({'id': id, "vector": embedding,
                    #                     'metadata': {"filename": filename, "text_chunk": text_chunk, "file_chunk_index": i}}))

                    # load_vectors(redis_conn, vectors, VECTOR_FIELD_NAME)
                elif parse_method == 'chunk_page':
                    page_map = pdf_to_text_mapping(file_stream)
                    chunks = []
                    for (section, pagenum) in split_text__chunk_page(page_map):
                        text_chunk = clean_text(section)
                        chunks.append(text_chunk)
                    return func.HttpResponse(
                        json.dumps(chunks),
                        status_code=200,
                        mimetype="application/json"
                    )
                else:
                    return func.HttpResponse(
                        "Invalid parse method",
                        status_code=400
                    )
            else:
                return func.HttpResponse(
                    "Invalid file type",
                    status_code=400
                )
        except Exception as e:
            logging.error(f"Error reading PDF: {e}")
            return func.HttpResponse(
                "Error reading PDF",
                status_code=500
            )

    return func.HttpResponse(
        "This function triggered successfully. Index: " + index,
        status_code=200
    )
