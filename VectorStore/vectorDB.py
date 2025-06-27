import faiss
import numpy as np
from embedding import embedding_vector  

# Assume you already have embeddings as a 2D NumPy array
# Example shape: (nb, d), where nb = number of vectors, d = dimension
# embedding_vectors = np.array([[...], [...], ...], dtype='float32')

# ✅ Replace with your actual embeddings
embedding_vector = embedding_vector.astype('float32')  # Example shape

# Step 1: Get dimensionality
d = embedding_vector.shape[1]  # Example: 384

# Step 2: Create FAISS index (L2 or cosine)
index = faiss.IndexFlatL2(d)  # L2 = Euclidean distance
# Use IndexFlatIP for cosine similarity (after normalizing vectors)

# Step 3: Add vectors to the index
index.add(embedding_vector)  # Now the index holds all vectors

# Step 4: (Optional) Save index to disk
faiss.write_index(index, "vector_index.faiss")

# Step 5: (Optional) Load index later
# index = faiss.read_index("vector_index.faiss")

# Step 6: (Optional) Perform a search
# Query vector: assume one of your existing ones or new
query_vector = embedding_vector[0:1]  # Shape must be (1, d)
k = 5  # Top 5 nearest neighbors

distances, indices = index.search(query_vector, k)

print("Nearest indices:", indices)
print("Distances:", distances)
