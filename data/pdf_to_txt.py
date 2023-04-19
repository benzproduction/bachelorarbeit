#################################################################################
# Driver Script to preprocess the raw pdf documents into maschine readable text #
#################################################################################

import os
import textract
from tqdm import tqdm


output_dir = os.path.join(os.path.dirname(__file__), "raw", "real_estate_txts")
os.makedirs(output_dir, exist_ok=True)

input_dir = os.path.join(os.path.dirname(__file__), "raw", "real_estate_pdfs")

pdf_files = os.listdir(input_dir)

for pdf_file in tqdm(pdf_files):
    if os.path.exists(os.path.join(output_dir, pdf_file + ".txt")):
        tqdm.write(f"Skipping {pdf_file}")
        continue
    pdf_path = os.path.join(input_dir, pdf_file)
    tqdm.write(f"Processing {pdf_file}")

    try:
        # Extract the raw text from each PDF using textract
        text = textract.process(pdf_path, method='pdfminer')
        text = text.decode("utf-8")

        # save the text to a file
        with open(os.path.join(output_dir, pdf_file + ".txt"), "w") as f:
            f.write(text)
    except Exception as e:
        tqdm.write(f"Error processing {pdf_file}: {e}")
        continue





