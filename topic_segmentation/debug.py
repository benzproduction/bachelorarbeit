from chunk import split_text
import os
from pypdf import PdfReader

pdf_path = os.path.join("data/raw/real_estate_pdfs/3.-Due-DIligence-Reports_Durres-region.pdf")
text_path = os.path.join('data', 'raw', 'txts', '_10-K-2019-(As-Filed).pdf.txt')
result_path = './topic_segmentation/segmented_text.txt'

def get_document_text(filename):
    offset = 0
    page_map = []

    reader = PdfReader(filename)
    pages = reader.pages
    for page_num, p in enumerate(pages):
        page_text = p.extract_text()
        page_map.append((page_num, offset, page_text))
        offset += len(page_text)
    return page_map

with open("result.txt", 'wt') as f:
    for page in get_document_text(pdf_path):
        f.write(page[2])
        f.write(f"\n##########   - {page[0]} - #########\n")


# with open(result_path, 'wt') as f:
#     for i, (section, pagenum) in enumerate(split_text(pdf_path)):
#         f.write(section)
#         f.write(f"\n##########   - {pagenum} - #########\n")

