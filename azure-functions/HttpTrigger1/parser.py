from PyPDF2 import PdfReader
from typing import List, Tuple
from .models import DocumentMetadata, Source

def get_metadata(file_stream, mimetype, filename):
    reader = PdfReader(file_stream)
    if "/" in filename:
        filename = filename.split("/")[-1]
    
    # TODO: add url
    return DocumentMetadata(
        source=Source.pdf,
        source_filename=filename,
        title=reader.metadata.title if reader.metadata.title else "",
        author=reader.metadata.author if reader.metadata.author else "",
        created_at=reader.metadata.creation_date_raw if reader.metadata.creation_date_raw else ""
    )
def pdf_to_text(file_stream : str) -> str:
    """
    Extracts the text from all pages of a PDF document and returns it as a single string.
    
    Args:
    ----------
    - file_stream: the file stream of the PDF document
    
    Returns:
    ----------
    - A single string containing the text extracted from all pages of the PDF document
    
    Raises:
    ----------
    - FileNotFoundError if the given filename does not exist or cannot be opened
    - ValueError if the given filename does not correspond to a PDF document
    
    Example usage:
    >>> pdf_to_text('example.pdf')
    'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt...'
    """
    reader = PdfReader(file_stream)
    pages = reader.pages
    return "\n".join([p.extract_text() for p in pages])

def pdf_to_text_mapping(file_stream: str) -> List[Tuple[int, int, str]]:
    """
    Extracts the text from all pages of a PDF document and returns it as a list of tuples.
    
    Args:
    ----------
    - file_stream: the file stream of the PDF document
    
    Returns:
    ----------
    - A list of tuples, where each tuple has the following elements:
      * the page number (starting from 0)
      * the offset in characters of the start of the page text relative to the beginning of the document
      * the text extracted from the page
      
    Raises:
    ----------
    - FileNotFoundError if the given filename does not exist or cannot be opened
    - ValueError if the given filename does not correspond to a PDF document
    
    Example usage:
    >>> pdf_to_text_mapping('example.pdf')
    [(0, 0, 'Lorem ipsum dolor sit amet...'), 
     (1, 1192, 'Consectetur adipiscing elit...'), 
     (2, 2249, 'Sed do eiusmod tempor incididunt...'), 
     ...]
    """
    offset = 0
    page_map = []

    reader = PdfReader(file_stream)
    pages = reader.pages
    for page_num, p in enumerate(pages):
        page_text = p.extract_text()
        page_map.append((page_num, offset, page_text))
        offset += len(page_text)
    return page_map