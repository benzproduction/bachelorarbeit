from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Generate questions from a text corpus
text = "The quick brown fox jumps over the lazy dog. The dog barks loudly."
sentences = text.split(". ")

# Generate a question for each sentence
questions = []
for sentence in sentences:
    prompt = "generate a question for this sentence: "
    input_text = prompt + sentence[:-1]  # remove the period from the end of the sentence
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_ids = model.generate(input_ids=input_ids, max_length=64, num_beams=4, early_stopping=True)
    question = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    questions.append(question)

# Print the generated questions
print(questions)
