"""
This script is used to split a document into its logical segments by using gpt as a text segmentation model.
WARNING: Using this script will burn through a lot of tokens very quickly.
"""

import os
from pydantic import BaseModel
from typing import List, Optional
from enum import Enum
import openai
from helpers import get_env, get_segments_from_output
from chunk import split_text
from pypdf import PdfReader

API_KEY, RESOURCE_ENDPOINT = get_env("azure-openai")

openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = RESOURCE_ENDPOINT
openai.api_version = "2022-12-01"


class Source(str, Enum):
    pdf = "pdf"


class DocumentMetadata(BaseModel):
    source: Optional[Source] = None
    source_filename: Optional[str] = None
    url: Optional[str] = None
    created_at: Optional[str] = None
    author: Optional[str] = None
    title: Optional[str] = None


class DocumentChunkMetadata(DocumentMetadata):
    document_id: Optional[str] = None
    page: Optional[int] = None


class DocumentChunk(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: DocumentChunkMetadata
    embedding: Optional[List[float]] = None


class Document(BaseModel):
    id: Optional[str] = None
    metadata: Optional[DocumentMetadata] = None


text_path = os.path.join('data', 'raw', 'real_estate_txts', 'pv12.pdf.txt')
pdf_path = os.path.join('data', 'raw', 'real_estate_pdfs', 'pv12.pdf')
stage_path = './topic_segmentation/stage_text.txt'
result_path = './topic_segmentation/segmented_text.txt'

MAX_TOKENS = 2014
MAX_SEGMENT_LENGTH = 1000

prompt = '''
Task: Segment a large chunk of text into segments of 300 to 500 words. Each segment should be marked with a start word <SEG> and an end word </SEG>. The text originates from a PDF document and contains page numbers as metadata.
To create a segment, you should concatenate the text from the beginning of a page or the end of the last segment until you reach a logical end or end of a sentence after the maximum word count. The page number of the first word in the segment should be used as metadata for the segment.
Input:
- A large chunk of text from a PDF document.
- Pages are marked with the format ###### - X - ######, where X is the page number.
Output:
- Segments, each marked with <SEG> and </SEG>.
- Each segment should be 300 to 500 words in length.
- The page number of the first word in each segment should be included as metadata.
Split the following text: {text}
'''


def get_metadata(filename):
    reader = PdfReader(filename)
    if "/" in filename:
        filename = filename.split("/")[-1]
    # TODO: add url
    return DocumentMetadata(
        source=Source.pdf,
        source_filename=filename,
        title=reader.metadata.title,
        author=reader.metadata.author,
        created_at=reader.metadata.creation_date_raw,
    )


def main():
    print("Starting...")

    metadata = get_metadata(pdf_path)
    document = Document(id="1", metadata=metadata)

    chunks = []
    # Split the document into its pages
    for i, (section, pagenum) in enumerate(split_text(pdf_path)):
        prepped_prompt = prompt.format(text=section)
        completion = openai.Completion.create(
            engine="davinci",
            prompt=prepped_prompt,
            temperature=0.5,
            max_tokens=MAX_TOKENS,
            n=1
        )
        segments = get_segments_from_output(completion.choices[0].text)
        for y, segment in enumerate(segments):
            chunk = DocumentChunk(
                id=f"{document.id}-{pagenum}-{y}",
                text=segment,
                metadata=DocumentChunkMetadata(
                    document_id=document.id,
                    page=pagenum,
                )
            )
            chunks.append(chunk)
        
        print(f"Finished page {pagenum} of {pdf_path}")
        if pagenum == 4:
            break
    
    with open(result_path, "w") as f:
        f.write("\n#################\n".join([chunk.text for chunk in chunks]))

    print("Finished!")

def split_progressive():
    
    for i, (section, pagenum) in enumerate(split_text(pdf_path)):
        # append the section to the stage txt file and write a ####### with the pagenum
        with open(stage_path, "a") as f:
            f.write(f"\n###### - {pagenum} - ######\n{section}")






if __name__ == "__main__":
    split_progressive()