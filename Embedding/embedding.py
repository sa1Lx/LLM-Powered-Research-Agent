from sentence_transformers import SentenceTransformer
import numpy as np

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the path to your .txt file
file_path = './chunks_output.txt'

# Step 1: Read the file content
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Step 2: Generate the embedding
embedding = model.encode(text)

# Step 3: Print the embedding vector
print("Embedding vector (first 10 values):", embedding[:10])
print("Embedding shape:", embedding.shape)
