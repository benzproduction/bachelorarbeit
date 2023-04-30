import mammoth
from typing import List, Tuple
from docx import Document

def docx_to_html(input_filename: str) -> str:
    """Convert the given docx file to html.

    Requires the mammoth package.

    """
    with open(input_filename, "rb") as docx_file:
        result = mammoth.convert_to_html(docx_file)
        return result.value

    
def docx_to_text_mapping(filename: str) -> List[Tuple[int, int, str]]:
    """
    Extracts the text from all paragraphs of a Word (.docx) file and returns it as a list of tuples.
    
    Args:
    - filename: str, the name or path of the Word file to read
    
    Returns:
    - A list of tuples, where each tuple has the following elements:
      * the paragraph number (starting from 0)
      * the offset in characters of the start of the paragraph text relative to the beginning of the document
      * the text extracted from the paragraph
      
    Raises:
    - FileNotFoundError if the given filename does not exist or cannot be opened
    - ValueError if the given filename does not correspond to a Word file
    
    Example usage:
    >>> extract_docx_text('example.docx')
    [(0, 0, 'This is some text in the first paragraph.'), 
     (1, 34, 'This is some text in the second paragraph.'), 
     (2, 70, 'This is some text in the third paragraph.\n\nThis is some more text in the third paragraph.'), 
     ...]
    """
    offset = 0
    para_map = []
    
    doc = Document(filename)
    paragraphs = doc.paragraphs
    for para_num, para in enumerate(paragraphs):
        para_text = para.text
        para_map.append((para_num, offset, para_text))
        offset += len(para_text)
        
    return para_map