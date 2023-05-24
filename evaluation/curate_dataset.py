# Expected output in the evaluation/data/15042023.jsonl file:
# {"input":[{"role":"system","content":"By what percentage have residential rents increased/decreased?"}],"ideal":["increased by 3,5%"]}
# this should be generated from the data/eval/20230419_Overview_RE Reports 2023 - Gold Standard.xlsx file
# The data is on sheet 2, where the questions are in column A and the list of ideal informations are in column J
# The cells in col J might look like this:
# - has risen sharply 
# - 10 year bond up to nearly 1,8% in June 22
# - stabilizing at 1,5 % in September 22
# each - is a new ideal information for the same question

import pandas as pd
import os
import jsonlines
import re

cwd = os.getcwd()

output_path = cwd + "/evaluation/data/first10q.jsonl"
fm_df = pd.read_excel(cwd + "/data/eval/20230419_Overview_RE Reports 2023 - Gold Standard.xlsx", sheet_name="Sheet1", usecols="C,E", header=None, names=['filename','source'])
fm_df = fm_df.drop([0,1])
fm_df = fm_df.dropna(subset=['filename'])
mapping = fm_df.set_index('source')['filename'].to_dict()
def replace_urls_with_filenames(urls):
    filenames = []
    for url in urls:
        filename = mapping.get(url, url)
        filenames.append(filename)
    return filenames
# read the excel file and only use the cols A and J
df = pd.read_excel(cwd + "/data/eval/20230419_Overview_RE Reports 2023 - Gold Standard.xlsx", sheet_name="Sheet2", usecols="A,L,M", header=None, names=['Question','paragraph','source'])
# # drop the first two rows as they do not contain any data
df = df.drop([0,1])
# # drop every row where answers is NaN
df = df.dropna(subset=['source'])
df['filename'] = df['source'].apply(lambda x: replace_urls_with_filenames(re.findall(r'(?:https?://[^\s]+)', x)))
data = df.apply(lambda row: {"input": row['Question'], "ideal": {"paragraph": row['paragraph'], "filename": [row['filename']] if isinstance(row['filename'], str) else row['filename']}}, axis=1).tolist()



# # create a list of dicts with the following structure:
# # {"input":[{"role":"system","content":question}],"ideal":[answer1, answer2, ...]}
# # where question is the question from the excel file and answer1, answer2, ... are the answers from the excel file
# data = []
# for index, row in df.iterrows():
#     question = row['Question']
#     answers = row['Answers'].split('\n')
#     answers = [answer.replace('-','').strip() for answer in answers]
#     answers = list(filter(None, answers))
#     data.append({"input":[{"role":"system","content":question}],"ideal":answers})

# write the data to the jsonl file
with jsonlines.open(output_path, mode='w') as writer:
    writer.write_all(data)

