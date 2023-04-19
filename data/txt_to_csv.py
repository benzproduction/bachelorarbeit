###########################################################################################
# Script to store txt documents in one csv file (tab seperated) with 2 cols (title, text) #
###########################################################################################

import os
import pandas as pd
from tqdm import tqdm
import re

output_path = os.path.join(os.path.dirname(__file__), "clean", "csv", "real_estate_dataset.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

input_dir = os.path.join(os.path.dirname(__file__), "raw", "real_estate_txts")

txt_files = os.listdir(input_dir)

data = []
for txt_file in tqdm(txt_files):
    txt_path = os.path.join(input_dir, txt_file)
    tqdm.write(f"Processing {txt_file}")

    try:
        with open(txt_path, "r") as f:
            text = f.read()
        data.append({"title": txt_file, "text": text})
    except Exception as e:
        tqdm.write(f"Error processing {txt_file}: {e}")
        continue

df = pd.DataFrame(data)

# clean the text
def replace_semicolon(text, threshold=10):
    '''
    Get rid of semicolons.

    First split text into fragments between the semicolons. If the fragment 
    is longer than the threshold, turn the semicolon into a period. O.w treat
    it as a comma.

    Returns new text
    '''
    new_text = ""
    for subset in re.split(';', text):
        subset = subset.strip() # Clear off spaces
        # Check word count
        if len(subset.split()) > threshold:
            # Turn first char into uppercase
            new_text += ". " + subset[0].upper() + subset[1:]
        else:
            # Just append with a comma 
            new_text += ", " + subset

    return new_text

USC_re = re.compile('[Uu]\.*[Ss]\.*[Cc]\.]+') 
PAREN_re = re.compile('\([^(]+\ [^\(]+\)')
BAD_PUNCT_RE = re.compile(r'([%s])' % re.escape('"#%&\*\+/<=>@[\]^{|}~_'), re.UNICODE)
BULLET_RE = re.compile('\n[\ \t]*`*\([a-zA-Z0-9]*\)')
DASH_RE = re.compile('--+')
WHITESPACE_RE = re.compile('\s+')
EMPTY_SENT_RE = re.compile('[,\.]\ *[\.,]')
FIX_START_RE = re.compile('^[^A-Za-z]*')
FIX_PERIOD = re.compile('\.([A-Za-z])')


def clean_text(text):
    """
    Borrowed from the FNDS text processing with additional logic added in.
    Note: we do not take care of token breaking - assume a tokenizer
    will handle this for us.
    """

    # Remove parantheticals 
    text = PAREN_re.sub('', text)

    # Get rid of enums as bullets or ` as bullets
    text = BULLET_RE.sub(' ',text)
    
    # Clean html 
    text = text.replace('&lt;all&gt;', '')

    # Remove annoying punctuation, that's not relevant
    text = BAD_PUNCT_RE.sub('', text)

    # Get rid of long sequences of dashes - these are formating
    text = DASH_RE.sub( ' ', text)

    # removing newlines, tabs, and extra spaces.
    text = WHITESPACE_RE.sub(' ', text)
    
    # If we ended up with "empty" sentences - get rid of them.
    text = EMPTY_SENT_RE.sub('.', text)
    
    # Attempt to create sentences from bullets 
    text = replace_semicolon(text)
    
    # Fix weird period issues + start of text weirdness
    #text = re.sub('\.(?=[A-Z])', '  . ', text)
    # Get rid of anything thats not a word from the start of the text
    text = FIX_START_RE.sub( '', text)
    # Sometimes periods get formatted weird, make sure there is a space between periods and start of sent   
    text = FIX_PERIOD.sub(". \g<1>", text)

    # Fix quotes
    text = text.replace('``', '"')
    text = text.replace('\'\'', '"')

    # remove ambigous unicode characters
    text = text.encode('ascii', 'ignore').decode('ascii')

    return text

df['text'] = df['text'].apply(clean_text)
# remove the .txt from the title col
df['title'] = df['title'].apply(lambda x: x.replace(".txt", ""))

df.to_csv(output_path, sep="\t", index=False)