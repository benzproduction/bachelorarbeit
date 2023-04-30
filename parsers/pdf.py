from PyPDF2 import PdfReader
from typing import List, Tuple

def pdf_to_text(filename: str) -> str:
	"""Converts a PDF file to a string of text.

	Parameters
	----------
	filename : str
		The name of the PDF file to convert.

	Returns
	-------
	str
		A string containing the text from the PDF file.

	"""
	pdf_file = PdfReader(filename)
	text=''
	
	for page in pdf_file.pages:
		text = text + page.extract_text()
	
	return text

def pdf_to_text_mapping(filename: str) -> List[Tuple[int, int, str]]:
    """
    Extracts the text from all pages of a PDF document and returns it as a list of tuples.
    
    Args:
    ----------
    - filename: str, the name or path of the PDF file to read
    
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
    >>> get_document_text('example.pdf')
    [(0, 0, 'Lorem ipsum dolor sit amet...'), 
     (1, 1192, 'Consectetur adipiscing elit...'), 
     (2, 2249, 'Sed do eiusmod tempor incididunt...'), 
     ...]
    """
    offset = 0
    page_map = []

    reader = PdfReader(filename)
    pages = reader.pages
    for page_num, p in enumerate(pages):
        page_text = p.extract_text()
        page_map.append((page_num, offset, page_text))
        offset += len(page_text)
    return page_map