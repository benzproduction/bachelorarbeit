#################################################################################
# Driver Script to preprocess the raw pdf documents into maschine readable text #
#################################################################################

import os
import textract
from tqdm import tqdm


local_path = os.path.join(os.path.dirname(__file__), "raw", "txts")
os.makedirs(local_path, exist_ok=True)

pdf_files = os.listdir(os.path.join(os.path.dirname(__file__), "raw", "pdfs"))

for pdf_file in tqdm(pdf_files):
    if os.path.exists(os.path.join(local_path, pdf_file + ".txt")):
        tqdm.write(f"Skipping {pdf_file}")
        continue
    pdf_path = os.path.join(os.path.dirname(__file__), "raw", "pdfs", pdf_file)
    tqdm.write(f"Processing {pdf_file}")

    try:
        # Extract the raw text from each PDF using textract
        text = textract.process(pdf_path, method='pdfminer')
        text = text.decode("utf-8")

        # save the text to a file
        with open(os.path.join(local_path, pdf_file + ".txt"), "w") as f:
            f.write(text)
    except Exception as e:
        tqdm.write(f"Error processing {pdf_file}: {e}")
        continue





