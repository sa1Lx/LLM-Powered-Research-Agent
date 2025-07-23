# Core Features

## Basic RAG Pipeline

* PDF Upload & Processing: Users can upload research papers and get them automatically processed
* Vector Store: Uses FAISS for efficient similarity search
* Chat Interface: Conversational RAG to chat with uploaded papers
* LLM: Uses Groq's Llama-3.1-8B model (fast and free)

## LangGraph Agents Implemented

1. Smart Summarizer Agent

* Extracts different sections (abstract, methods, results) step-by-step
* Creates comprehensive summaries with section breakdowns
* Uses multi-step LangGraph pipeline

Note: Assumes the abstract is within the first 3000 characters of the paper (for saving time costs)

2. Literature Comparator Agent

* Compares multiple uploaded papers across key dimensions
* Analyzes research objectives, methods, findings, and limitations
* Structured comparison output

# Technical Architecture

* State Management: Each agent has its own TypedDict state
* LangGraph Pipelines: Sequential and parallel processing nodes
* Streamlit UI: Tabbed interface for different agent functions
* Session State: Maintains chat history and paper database
* Error Handling: Comprehensive error handling throughout