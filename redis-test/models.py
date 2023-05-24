from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class Source(str, Enum):
    pdf = "pdf"
    docx = "docx"
    txt = "txt"
    url = "url"
    pptx = "pptx"


class DocumentMetadata(BaseModel):
    source: Optional[Source] = None
    source_filename: Optional[str] = None
    url: Optional[str] = None
    created_at: Optional[str] = None
    author: Optional[str] = None
    title: Optional[str] = None


class DocumentChunkMetadata(DocumentMetadata):
    document_id: Optional[str] = None
    document_metadata: Optional[DocumentMetadata] = None
    page: Optional[int] = None


class DocumentChunk(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: DocumentChunkMetadata
    embedding: Optional[List[float]] = None


class Document(BaseModel):
    id: Optional[str] = None
    metadata: Optional[DocumentMetadata] = None