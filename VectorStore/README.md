VectorStore objects contain methods for adding text and Document objects to the store, and querying them using various similarity metrics. They are often initialized with embedding models, which determine how text data is translated to numeric vectors.

Note that most vector store implementations will allow you to connect to an existing vector store-- e.g., by providing a client, index name, or other information.

Once we've instantiated a VectorStore that contains documents, we can query it. VectorStore includes methods for querying:

1. Synchronously and asynchronously;
2. By string query and by vector;
3. With and without returning similarity scores;
4. By similarity and maximum marginal relevance (to balance similarity with query to diversity in retrieved results).