import PyPDF2
import textract
import os
import tiktoken
import pandas as pd
import time
from typing import Dict, Any

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

def find_first_page_number(chunk, page_map):
    min_page_num = None
    
    for page_num, text in page_map.items():
        start_index = text.find(chunk[:20])  # Search for the first 20 characters of the chunk
        if start_index >= 0:
            if min_page_num is None or page_num < min_page_num:
                min_page_num = page_num
                
    return min_page_num

def find_most_likely_page_number(chunk, page_map, tokenizer):
    tokenized_chunk = set(tokenizer.encode(chunk))

    # Calculate scores for each page
    scores = {}
    for page_num, text in page_map.items():
        tokenized_page = set(tokenizer.encode(text))
        intersection = tokenized_chunk.intersection(tokenized_page)
        score = len(intersection)
        scores[page_num] = score

    # Find the page with the highest score
    most_likely_page_num = max(scores, key=scores.get)
    return most_likely_page_num

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


def find_page_number(chunk, page_map, tokenizer, min_subsequence_length=5):
    tokenized_chunk = tokenizer.encode(chunk)

    def find_max_subsequence_length(a, b):
        longest_subseq = 0
        for i in range(len(a)):
            for j in range(len(b)):
                length = 0
                while (i + length < len(a)) and (j + length < len(b)) and (a[i + length] == b[j + length]):
                    length += 1
                longest_subseq = max(longest_subseq, length)
        return longest_subseq

    scores = {page_num: find_max_subsequence_length(tokenized_chunk, tokenizer.encode(text)) for page_num, text in page_map.items()}
    most_likely_page_num = max(scores, key=scores.get)
    
    return most_likely_page_num if scores[most_likely_page_num] >= min_subsequence_length else -1


def pdf_to_text_map(pdf_path):
    # Read the PDF
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)

        # Iterate through pages
        page_texts = {}
        for i in range(len(reader.pages)):
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

# Use the function to extract text from a PDF
pdf_path = "/Users/shuepers001/dev/bachelorarbeit/data/raw/pdfs2/2022-12-12_ABG_Statements_Hoeller_2022-2023_en.pdf"
page_texts = pdf_to_text_map(pdf_path)
text = "\n".join(page_texts.values())

# # save the dict as a excel file
# df = pd.DataFrame.from_dict(page_texts, orient='index')
# df.to_excel('test.xlsx')

# chunk the text and print the chunks n=300
tokenizer = tiktoken.get_encoding("cl100k_base")

token_chunks = list(chunks(text, 300, tokenizer))
text_chunks = [tokenizer.decode(chunk) for chunk in token_chunks]


# for every chunk find the first page number
chunks_page_map = {}
for chunk in text_chunks:
    # stop how long it takes to get the page number with the first method
    start1 = time.time()
    page_num1 = find_page_number(chunk, page_texts, tokenizer)
    end1 = time.time()
    print(f"Time for first method: {end1 - start1}")
    start2 = time.time()
    page_num2 = find_most_likely_page_number(chunk, page_texts, tokenizer)
    end2 = time.time()
    print(f"Time for second method: {end2 - start2}")
    # print the results
    print(f"Method 1: {page_num1} --- Method 2: {page_num2}")



# print the chunks and the page number
for chunk, page_num in chunks_page_map.items():
    print(f"Page {page_num}: {chunk}")
