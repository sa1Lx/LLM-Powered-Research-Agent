# Langchain Basics

LangChain is a Python framework that helps you:

- Connect to LLMs like GPT
- Load and split documents
- Retrieve useful text chunks
- Build chains for question-answering

## LangSmith 
### What is LangSmith? (Simple Explanation)

LangSmith is like a "quality control dashboard" for AI language models. It's a tool that helps developers:

1. **Build better AI apps** - Like having a mechanic's toolkit for your AI projects
2. **Fix problems** - Shows you exactly where your AI might be making mistakes
3. **Test improvements** - Lets you try different versions to see what works best

#### Think of it like this:
- If AI apps were cars, LangChain would be the factory that builds them
- LangSmith would be the **diagnostic computer** that mechanics use to check the car's performance and find issues

### Key Features:
- **Debugging** - Finds where your AI is getting confused
- **Monitoring** - Watches your AI's performance over time
- **Testing** - Helps compare different versions of your AI

It's made by the same team that created LangChain, but focuses more on making AI apps ready for real-world use rather than just experimenting.

## ChatGroq

### Installation
```bash
pip install -U langchain-groq
```

### Setup 

```python
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv("api.env")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
)
```

### Invocation

```python
messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)
```

## Retriever

Many different types of retrieval systems exist, including vectorstores, graph databases, and relational databases. With the rise on popularity of large language models, retrieval systems have become an important component in AI application (e.g., RAG). Because of their importance and variability, LangChain provides a uniform interface for interacting with different types of retrieval systems. The LangChain retriever interface is straightforward:

1. Input: A query (string)
2. Output: A list of documents (standardized LangChain Document objects)

The only requirement for a retriever is the ability to accepts a query and return documents. In particular, LangChain's retriever class only requires that the _get_relevant_documents method is implemented, which takes a query: str and returns a list of Document objects that are most relevant to the query. The underlying logic used to get relevant documents is specified by the retriever and can be whatever is most useful for the application.

A LangChain retriever is a runnable, which is a standard interface for LangChain components. This means that it has a few common methods, including invoke, that are used to interact with it. A retriever can be invoked with a query:

```python
docs = retriever.invoke(query)
```
Retrievers return a list of Document objects, which have two attributes:

1. page_content: The content of this document. Currently is a string.
2. metadata: Arbitrary metadata associated with this document (e.g., document id, file name, source, etc).

Vector stores are a powerful and efficient way to index and retrieve unstructured data. A vectorstore can be used as a retriever by calling the as_retriever() method.

```python
vectorstore = MyVectorStore()
retriever = vectorstore.as_retriever()
```

