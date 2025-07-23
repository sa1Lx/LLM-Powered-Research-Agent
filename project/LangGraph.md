# LangGraph

LangGraph is a framework for building stateful, multi-step applications with language models. In this case, it's orchestrating a simple RAG pipeline with two sequential steps.

## 1. State Management

```python
from langchain_core.documents import Document
from typing_extensions import List, TypedDict

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
```

LangGraph uses a state object that gets passed between different steps. This state acts like a shared memory that each function can read from and update. Here, it tracks:

* The user's question
* Retrieved context documents
* The final answer

## 2. The Two-Step Pipeline

```python
def retrieve(state: State):
    retrieved_docs = db.similarity_search(state["question"])
    return {"context": retrieved_docs}
```

* Takes the question from state
* Searches the FAISS vector database for similar documents
* Updates state with the retrieved context

```python
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}
```

* Takes question and context from state
* Combines context documents into a single string
* Uses the prompt template and LLM to generate an answer
* Updates state with the final answer

## 3. The Graph Construction

```python
from langgraph.graph import START, StateGraph

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
```

LangGraph creates a directed graph where:

* Each node is a function that processes the state
* `add_sequence([retrieve, generate])` creates: retrieve → generate
* `add_edge(START, "retrieve")` sets the entry point
* The graph automatically handles state passing between nodes

## 4. Displaying the Graph

```python
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
```

# Why Use LangGraph Here?

While this example is simple, LangGraph provides several benefits:

* State Management: Automatic state passing between steps
* Modularity: Each step is a separate, testable function
* Visualization: Can generate visual graphs of the pipeline
* Extensibility: Easy to add more steps (like re-ranking, validation, etc.)
* Error Handling: Built-in error propagation and recovery
* Debugging: Can inspect state at each step