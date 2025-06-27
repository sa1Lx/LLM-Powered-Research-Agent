import faiss
import numpy as np
from embedding import embedding_vector  



embedding_vector = embedding_vector.astype('float32')  # Example shape

d = embedding_vector.shape[1]  # Example: 384


index = faiss.IndexFlatL2(d)  # L2 = Euclidean distance


index.add(embedding_vector)  # Now the index holds all vectors


faiss.write_index(index, "vector_index.faiss")


query_vector = embedding_vector[0:1]  # Shape must be (1, d)
k = 5  # Top 5 nearest neighbors

distances, indices = index.search(query_vector, k)

print("Nearest indices:", indices)
print("Distances:", distances)
