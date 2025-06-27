from sentence_transformers import SentenceTransformer
import numpy as np


model = SentenceTransformer('all-MiniLM-L6-v2')


file_path = './chunks_output.txt'


with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()


embedding = model.encode(text)

print("Embedding vector (first 10 values):", embedding[:10])
print("Embedding shape:", embedding.shape)
