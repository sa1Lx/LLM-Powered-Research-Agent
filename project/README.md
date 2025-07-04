# Introduction

Implementing a RAG pipeline in Jupyter Notebook using Langchain to answer questions from the first three chapters of Zero to One by Peter Thiel. 
Reference used to build: [Langchain Documentation](https://python.langchain.com/docs/tutorials/rag/).

## Document Loading

Document loaders in Langchain: [Document Loaders](https://python.langchain.com/docs/integrations/document_loaders/).

I will be using [`PyPDFLoader`](https://python.langchain.com/docs/integrations/document_loaders/pypdfloader/) from `langchain_community.document_loaders` to load the PDF documents. This loader is simple and fast, preserving page metadata but does not parse layout or tables, not required for the current task.
There are 3 types of loading methods available:

| Method         | Type         | Description                                                                                |
| -------------- | ------------ | ------------------------------------------------------------------------------------------ |
| `load()`       | Eager        | Loads **all documents at once** into memory.                                               |
| `lazy_load()`  | Lazy (sync)  | Returns a **generator**, yielding documents one at a time. Saves memory.                   |
| `alazy_load()` | Lazy (async) | Returns an **async generator** for non-blocking document loading (used with `async` code). |

To install: `pip install -qU pypdf`<br>
To import: `from langchain_community.document_loaders import PyPDFLoader`

For Eager loading, use:
```python
Loader = PyPDFLoader("path/to/your/document.pdf")
documents = Loader.load()
```
For Lazy loading, use:
```python
Loader = PyPDFLoader("path/to/your/document.pdf")
documents = Loader.lazy_load()
for doc in documents:
    print(doc.page_content)  # Process each document as it loads
```
I used lazy loading to parse the first three chapters of "Zero to One" by Peter Thiel. Since the first page has no text layer embedding, I manually added the book intro with minimal metadata.

## Text Splitting

See the approaches for text splitting in Langchain [here](https://python.langchain.com/docs/concepts/text_splitters/#approaches).

I will use [`RecursiveCharacterTextSplitter`](https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html) from `langchain.text_splitter` to split the text into manageable chunks. This splitter recursively divides text based on character count, ensuring each chunk is coherent and contextually relevant.

To import: `from langchain.text_splitter import RecursiveCharacterTextSplitter`

Used method [`split_documents`](https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html#langchain_text_splitters.character.RecursiveCharacterTextSplitter.split_documents) to split the loaded documents into chunks of 100 characters with no overlap.
```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_documents(pages)
```

## Embedding the Chunks

I will be using [`SentenceTransformerEmbeddings`](https://sbert.net/) for text embedding. Sentence Transformers (a.k.a. SBERT) is the go-to Python module for accessing, using, and training state-of-the-art embedding and reranker models.

To install: `pip install -qU langchain-huggingface`<br>
To import: `from langchain_huggingface import HuggingFaceEmbeddings`

The sentence-transformer model I am using is [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2): It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.
```python
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

## Storing in FAISS

To install: `pip install -qU langchain-huggingface` `pip install -qU langchain-community`<br>
To import: `from langchain_huggingface import HuggingFaceEmbeddings` `import faiss` `from langchain_community.docstore.in_memory import InMemoryDocstore`
`from langchain_community.vectorstores import FAISS`

```python
db = FAISS.from_documents(texts, embeddings_model)
```

* the vectors are stored in a FAISS index, which is a library for efficient similarity search and clustering of dense vectors.

```python
db.index.ntotal # returns the number of vectors in the index

import numpy as np # inspects vectors directly
v = np.zeros((1, 384), dtype="float32")
db.index.search(v, k=3)

db.index.add(np.array([...], dtype='float32')) # add more vectors
db.index.search(query_vector, k) # search raw FAISS
```

* in-memory document store: A simple key-value store that holds the original LangChain Document objects. It maps doc_id → Document

```python
doc = db.docstore.search("5") # searching for a document by its ID
print(doc.page_content)

for k, v in db.docstore._dict.items(): # iterate over all documents in the store
    print(k, v.page_content)
```

* A dictionary that maps FAISS vector index positions to docstore IDs. Since FAISS uses numeric indices (0, 1, 2, ...), LangChain stores a mapping.

```python
db.index_to_docstore_id[0] # get the docstore ID for the first vector
```

# Retriever

The LangChain retriever interface is straightforward:

1. Input: A query (string)
2. Output: A list of documents (standardized LangChain Document objects)

A LangChain retriever is a runnable, which is a standard interface for LangChain components. This means that it has a few common methods, including invoke, that are used to interact with it. A retriever can be invoked with a query:
```python
docs = retriever.invoke(query)
```
Retrievers return a list of Document objects, which have two attributes:

1. page_content: The content of this document. Currently is a string.
2. metadata: Arbitrary metadata associated with this document (e.g., document id, file name, source, etc).

Vector stores are a powerful and efficient way to index and retrieve unstructured data. A vectorstore can be used as a retriever by calling the as_retriever() method:
```python
vectorstore = MyVectorStore()
retriever = vectorstore.as_retriever()
```

```python
from langchain import hub

# N.B. for non-US LangSmith endpoints, you may need to specify
# api_url="https://api.smith.langchain.com" in hub.pull.
prompt = hub.pull("rlm/rag-prompt")

example_messages = prompt.invoke(
    {"context": "(context goes here)", "question": "(question goes here)"}
).to_messages()

assert len(example_messages) == 1
print(example_messages[0].content)
```

# LangGraph

[LangGraph API Concepts](https://langchain-ai.github.io/langgraph/concepts/low_level/?_gl=1*waztpb*_ga*OTc1ODQ5Nzg0LjE3NTE1NDczMjI.*_ga_47WX3HKKY2*czE3NTE1NjgyNjMkbzYkZzEkdDE3NTE1NjkyMTgkajU4JGwwJGgw)

To use LangGraph, we need to define three things:

1. The state of our application;
2. The nodes of our application (i.e., application steps);
3. The "control flow" of our application (e.g., the ordering of the steps).

## State
he state of our application controls what data is input to the application, transferred between steps, and output by the application. It is typically a TypedDict, but can also be a Pydantic BaseModel.

For a simple RAG application, we can just keep track of the input question, retrieved context, and generated answer:
```python
from langchain_core.documents import Document
from typing_extensions import List, TypedDict


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
```

