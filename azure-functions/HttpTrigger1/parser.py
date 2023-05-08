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
    
    Example usage:
    >>> pdf_to_text('example.pdf')
    'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt...'
    """
    reader = PdfReader(file_stream)
    pages = reader.pages
    return "\n".join([p.extract_text() for p in pages])

def pdf_to_txt_w_pages(file_stream: str) -> Tuple[str, List[str]]:
    """
    Extracts the text from all pages of a PDF document and returns it as a single string.
    
    Args:
    ----------
    - file_stream: the file stream of the PDF document
    
    Returns:
    ----------
    - A tuple containing:
      * A single string containing the text extracted from all pages of the PDF document
      * A list of strings, where each string is the text extracted from a single page of the PDF document
    
    Example usage:
    >>> pdf_to_text('example.pdf')
    ('Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt...', 
     ['Lorem ipsum dolor sit amet...', 
      'Consectetur adipiscing elit...', 
      'Sed do eiusmod tempor incididunt...', 
      ...])
    """
    reader = PdfReader(file_stream)
    pages = reader.pages
    page_texts = [p.extract_text() for p in pages]
    text = "\n".join(page_texts)
    return text, page_texts

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

def find_page_from_text(text:str, pages: List[str]) -> int:
    """
    Finds the page number of a given text in a list of pages.
    
    Args:
    ----------
    - text: the text to search for
    - pages: a list of strings, where each string is the text extracted from a single page of the PDF document
    
    Returns:
    ----------
    - The page number (starting from 0) of the given text in the list of pages
    
    Example usage:
    >>> find_page_from_text('Sed do eiusmod tempor incididunt...', 
                            ['Lorem ipsum dolor sit amet...', 
                             'Consectetur adipiscing elit...', 
                             'Sed do eiusmod tempor incididunt...', 
                             ...])
    2
    """
    offset = 0
    for page_num, page_text in enumerate(pages):
        if text in page_text:
            return page_num
        offset += len(page_text)
    return -1