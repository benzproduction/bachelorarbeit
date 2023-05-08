import logging
import io
import mimetypes
import tiktoken
import json
import os
import openai
from openai.embeddings_utils import get_embedding
import azure.functions as func
from .parser import pdf_to_text_mapping, get_metadata, pdf_to_txt_w_pages, find_page_from_text
from .chunker import split_text__chunk_default, split_text__chunk_page
from .clean_text import clean_text
from .models import Document, DocumentChunk, DocumentChunkMetadata
from .database import save_vectors, get_redis_connection


API_KEY = os.environ["OPENAI_API_KEY"]
RESOURCE_ENDPOINT = os.environ["OPENAI_RESOURCE_ENDPOINT"]

openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = RESOURCE_ENDPOINT
openai.api_version = "2022-12-01"

def unique_id_from_filename(index:str, filename: str) -> str:
    return f"{index}-{filename}"
def get_unique_id_for_file_chunk(filename, chunk_index):
    return str(filename+"-!"+str(chunk_index))

PARSE_OPTIONS = ['chunk_default', 'chunk_page', 'logic_segments']

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processing a request.')

    # if method is POST do the following
    if req.method == "POST":
        if not req.files:
            return func.HttpResponse(
                "Please pass at least one file in the request body",
                status_code=400
            )

        index = req.params.get('index')
        if not index:
            return func.HttpResponse(
            "Please pass an index on the query string or in the request body",
                status_code=400
            )
        
        parse_method = req.params.get('parse_method', 'chunk_default')
        if parse_method not in PARSE_OPTIONS:
            return func.HttpResponse(
                f"Please pass a valid parse_method on the query string (one of {PARSE_OPTIONS})",
                status_code=400
            )

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
        
        reset_index = req.params.get('reset_index', False)
        try:
            reset_index = bool(reset_index)
        except:
            return func.HttpResponse(
                "Please pass a valid reset_index on the query string (True or False)",
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
                        text, pages = pdf_to_txt_w_pages(file_stream)
                        text = clean_text(text)
                        token_chunks = list(split_text__chunk_default(text, TEXT_EMBEDDING_CHUNK_SIZE, tokenizer))
                        text_chunks = [tokenizer.decode(chunk) for chunk in token_chunks]

                        logging.info(f"Splitted into {len(text_chunks)} chunks")

                        embeddings = []
                        for text_chunk in text_chunks:
                            embeddings.append(get_embedding(text_chunk, engine='text-embedding-ada-002'))

                        assert len(text_chunks) == len(embeddings), "Number of text chunks and embeddings do not match"
                        logging.info(f"Successfully created the embeddings for the chunks")

                        text_embeddings = list(zip(text_chunks, embeddings))

                        vectors = []
                        for i, (text_chunk, embedding) in enumerate(text_embeddings):
                            id = get_unique_id_for_file_chunk(filename, i)
                            metadata = DocumentChunkMetadata(
                                document_id=document.id,
                                document_metadata=document.metadata,
                                page=find_page_from_text(text_chunk, pages)
                            )
                            chunk = DocumentChunk(
                                id=id,
                                text=text_chunk,
                                metadata=metadata,
                                embedding=embedding
                            )
                            vectors.append(chunk)
                        logging.info(f"Saving {len(vectors)} vectors")
                        save_vectors(vectors, index, delete_index=reset_index)
                        return func.HttpResponse(
                            "Successfully created vectors",
                            status_code=200
                        )

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
                    elif parse_method == 'logic_segments':
                        return func.HttpResponse(
                            "Logic segments parsing method not implemented",
                            status_code=400
                        )
                else:
                    return func.HttpResponse(
                        "Invalid file type",
                        status_code=400
                    )
            except Exception as e:
                logging.error(f"Error reading file: {e}")
                return func.HttpResponse(
                    "Error reading file",
                    status_code=500
                )
    elif req.method == "GET":
        index = req.params.get('index')
        if not index:
            return func.HttpResponse(
                "Please pass an index on the query string or in the request body",
                status_code=400
            )
        r = get_redis_connection()
        # get all key value pairs
        keys = r.keys(f"{index}:*")
        keys = [key.decode('utf-8') for key in keys]
        docs = []
        for key in keys:
            element_data = {'id': key}
            element_metadata = r.hgetall(key)
            for key, value in element_metadata.items():
                try:
                    element_data[key.decode('utf-8')] = value.decode('utf-8')
                except UnicodeDecodeError:
                    element_data[key.decode('utf-8')] = value.decode('ISO-8859-1')
            docs.append(element_data)
        
        return func.HttpResponse(
            json.dumps(docs),
            status_code=200,
            mimetype="application/json"
        )



    else:
        return func.HttpResponse(
            "Please pass a valid method (GET or POST)",
            status_code=400
        )
