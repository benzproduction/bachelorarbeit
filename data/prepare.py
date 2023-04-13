"""
Prepare the dataset for language modeling.
Encoding with GPT-2 BPE tokens. 
Saves train.bin and val.bin files.
"""
import os
import tiktoken
import numpy as np

output_dir = os.path.join(os.path.dirname(__file__), 'model_input')
os.makedirs(output_dir, exist_ok=True)


input_file_path = os.path.join(os.path.dirname(__file__), 'clean')

with open(input_file_path, 'r') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(output_dir, 'train.bin')
val_ids.tofile(output_dir, 'val.bin')
