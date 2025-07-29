from openai import api_key
import streamlit as st
import os
import tempfile
from typing import List, Dict, Any, TypedDict
from datetime import datetime
import json
import requests

# Core libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub

# LangGraph
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated

# Set page config
st.set_page_config(
    page_title="Moist Research Assistant",
    page_icon="page_icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables inside load_models()
from dotenv import load_dotenv


# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_paper' not in st.session_state:
    st.session_state.current_paper = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'papers_db' not in st.session_state:
    st.session_state.papers_db = {}

# State definitions for different agents
class ChatState(TypedDict):
    question: str
    context: List[Document]
    answer: str
    chat_history: Annotated[list, add_messages]

class SummarizerState(TypedDict):
    documents: List[Document]
    sections: Dict[str, str]
    final_summary: str

class ComparatorState(TypedDict):
    papers: List[Dict[str, Any]]
    comparison_matrix: Dict[str, Any]
    final_comparison: str

# Initialize models
@st.cache_resource
def load_models():
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Check project.env or path.")

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Initialize LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        api_key=api_key
    )
    
    return embeddings, llm

embeddings_model, llm = load_models()

# Document processing functions
def process_paper(uploaded_file):
    """Process uploaded PDF and create vector store"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # Load PDF
        loader = PyPDFLoader(tmp_file_path)
        documents = list(loader.lazy_load())
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        texts = text_splitter.split_documents(documents)
        
        # Create vector store
        vector_store = FAISS.from_documents(texts, embeddings_model)
        
        return documents, texts, vector_store
    
    finally:
        os.unlink(tmp_file_path)

# RAG Chat Agent
def create_chat_agent():
    def retrieve(state: ChatState):
        if st.session_state.vector_store is None:
            return {"context": []}
        
        retrieved_docs = st.session_state.vector_store.similarity_search(
            state["question"], k=4
        )
        return {"context": retrieved_docs}
    
    def generate(state: ChatState):
        if not state["context"]:
            return {"answer": "Please upload a paper first to chat with it."}
        
        context_text = "\n\n".join([doc.page_content for doc in state["context"]])
        
        prompt = ChatPromptTemplate.from_template("""
        You are a research assistant helping users understand academic papers. 
        Use the following context from the research paper to answer the question.
        If you cannot find the answer in the context, say so clearly.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """)
        
        messages = prompt.invoke({
            "context": context_text,
            "question": state["question"]
        })
        
        response = llm.invoke(messages)
        return {"answer": response.content}
    
    # Build graph
    graph_builder = StateGraph(ChatState)
    graph_builder.add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    
    return graph_builder.compile()

# Smart Summarizer Agent
def create_summarizer_agent():
    def extract_sections(state: SummarizerState):
        documents = state["documents"]
        full_text = "\n".join([doc.page_content for doc in documents])
        
        # Extract different sections
        sections = {}
        
        # Abstract extraction
        abstract_prompt = ChatPromptTemplate.from_template("""
        Extract and summarize the abstract from this research paper. If no clear abstract is found, 
        summarize the introduction/overview section.
        
        Text: {text}
        
        Abstract Summary:
        """)
        
        abstract_response = llm.invoke(abstract_prompt.invoke({"text": full_text[:3000]}))
        sections["abstract"] = abstract_response.content
        
        # Methods extraction
        methods_prompt = ChatPromptTemplate.from_template("""
        Extract and summarize the methodology, approach, or methods section from this research paper.
        
        Text: {text}
        
        Methods Summary:
        """)
        
        methods_response = llm.invoke(methods_prompt.invoke({"text": full_text}))
        sections["methods"] = methods_response.content
        
        # Results extraction
        results_prompt = ChatPromptTemplate.from_template("""
        Extract and summarize the results, findings, or conclusions from this research paper.
        
        Text: {text}
        
        Results Summary:
        """)
        
        results_response = llm.invoke(results_prompt.invoke({"text": full_text}))
        sections["results"] = results_response.content
        
        return {"sections": sections}
    
    def create_final_summary(state: SummarizerState):
        sections = state["sections"]
        
        final_prompt = ChatPromptTemplate.from_template("""
        Create a comprehensive summary of this research paper based on the following sections:
        
        Abstract: {abstract}
        
        Methods: {methods}
        
        Results: {results}
        
        Provide a well-structured, coherent summary that captures the key contributions and findings.
        
        Final Summary:
        """)
        
        final_response = llm.invoke(final_prompt.invoke(sections))
        return {"final_summary": final_response.content}
    
    # Build graph
    graph_builder = StateGraph(SummarizerState)
    graph_builder.add_sequence([extract_sections, create_final_summary])
    graph_builder.add_edge(START, "extract_sections")
    
    return graph_builder.compile()

# Literature Comparator Agent
def create_comparator_agent():
    def analyze_papers(state: ComparatorState):
        papers = state["papers"]
        
        comparison_prompt = ChatPromptTemplate.from_template("""
        Compare the following research papers across these dimensions:
        1. Research objectives and questions
        2. Methodological approaches
        3. Key findings and contributions
        4. Limitations and future work
        
        Papers to compare:
        {papers_info}
        
        Provide a structured comparison:
        """)
        
        papers_info = ""
        for i, paper in enumerate(papers, 1):
            papers_info += f"\nPaper {i}: {paper['title']}\nSummary: {paper['summary']}\n"
        
        response = llm.invoke(comparison_prompt.invoke({"papers_info": papers_info}))
        
        return {
            "comparison_matrix": {"analysis": response.content},
            "final_comparison": response.content
        }
    
    # Build graph
    graph_builder = StateGraph(ComparatorState)
    graph_builder.add_node("analyze_papers", analyze_papers)
    graph_builder.add_edge(START, "analyze_papers")
    
    return graph_builder.compile()

# Initialize agents
@st.cache_resource
def load_agents():
    return {
        "chat": create_chat_agent(),
        "summarizer": create_summarizer_agent(),
        "comparator": create_comparator_agent(),
    }

agents = load_agents()

# Main UI
def main():
    st.title("Moist Research Assistant")
    st.markdown("Upload research papers and interact with Moist to get insights, summaries, and more!")
    
    # Sidebar
    with st.sidebar:
        st.header("📄 Paper Management")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Research Paper (PDF)",
            type="pdf",
            help="Upload a research paper to analyze"
        )
        
        if uploaded_file is not None:
            if st.button("Process Paper"):
                with st.spinner("Moist is processing your paper..."):
                    try:
                        documents, texts, vector_store = process_paper(uploaded_file)
                        
                        # Store in session state
                        st.session_state.current_paper = {
                            "name": uploaded_file.name,
                            "documents": documents,
                            "texts": texts,
                            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        st.session_state.vector_store = vector_store
                        
                        # Add to papers database
                        paper_id = f"paper_{len(st.session_state.papers_db)}"
                        st.session_state.papers_db[paper_id] = st.session_state.current_paper
                        
                        st.success("Moist has processed your paper successfully!")
                        
                    except Exception as e:
                        st.error(f"Error processing paper: {str(e)}")
        
        # Display current paper info
        if st.session_state.current_paper:
            paper_ids = list(st.session_state.papers_db.keys())  # Get list of all paper IDs

            # Create a list of paper names, corresponding one-to-one to the IDs
            paper_names = [st.session_state.papers_db[pid]["name"] for pid in paper_ids]

            # Display a radio button selector with the paper names
            selected_name = st.radio(
                "Select a paper to interact with:",
                paper_names
            )

            # Find the paper ID that corresponds to the chosen paper's name
            selected_paper_id = paper_ids[paper_names.index(selected_name)]

            # Update session state to track the chosen paper
            st.session_state.current_paper_id = selected_paper_id
            st.session_state.current_paper = st.session_state.papers_db[selected_paper_id]

            st.session_state.vector_store = FAISS.from_documents(
            st.session_state.current_paper["texts"], embeddings_model
            )

            st.write(f"**Current Paper:** {st.session_state.current_paper['name']}")

        # Display all papers
        if st.session_state.papers_db:
            st.subheader("All Papers")
            for paper_id, paper_info in st.session_state.papers_db.items():
                st.caption(f"• {paper_info['name']}")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs([
        "Chai and Chat With Moist", "📋 Summarizer", "🔍 Comparator"
    ])
    
    with tab1:
        cols = st.columns([7, 1])    # Left (header) wide, right (button) narrow
        with cols[0]:
            st.header("Ask Moist Anything!")
        with cols[1]:
            if st.session_state.chat_history:  # Only show when there is history
                if st.button("🗑️ Clear", key="clear_chat", help="Clear chat history"):
                    st.session_state.chat_history = []
                    st.rerun()  # Refresh UI
        
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if question := st.chat_input("Ask a question about the paper..."):
            if st.session_state.current_paper is None:
                st.warning("Please upload a paper first!")
            else:
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": question})
                
                with st.chat_message("user"):
                    st.write(question)
                
                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("Let Moist Cook..."):
                        result = agents["chat"].invoke({
                            "question": question,
                            "context": [],
                            "answer": "",
                            "chat_history": []
                        })
                        
                        answer = result["answer"]
                        st.write(answer)
                        
                        # Add AI response to chat history
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()
    
    with tab2:
        cols = st.columns([7, 1])
        with cols[0]:
            st.header("Smart Paper Summarizer")
        with cols[1]:
            if SummarizerState in st.session_state:
                if st.button("🗑️ Clear", key="clear_summary", help="Clear summary result"):
                    st.session_state.pop(SummarizerState, None)
                    st.rerun()
        
        if st.session_state.current_paper is None:
            st.warning("Please upload a paper first!")
        else:
            if st.button("Generate Smart Summary", key="summarize"):
                with st.spinner("Moist is analyzing paper sections..."):
                    result = agents["summarizer"].invoke({
                        "documents": st.session_state.current_paper["documents"],
                        "sections": {},
                        "final_summary": ""
                    })
                    
                    st.subheader("📄 Comprehensive Summary")
                    st.write(result["final_summary"])
                    
                    # Show section summaries
                    if "sections" in result:
                        with st.expander("View Section Summaries"):
                            for section, content in result["sections"].items():
                                st.subheader(section.title())
                                st.write(content)
    
    with tab3:
        cols = st.columns([7, 1])
        with cols[0]:
            st.header("Literature Comparator")
        with cols[1]:
            if ComparatorState in st.session_state:  # Replace with comparison state if any
                if st.button("🗑️ Clear", key="clear_comparison", help="Clear comparison result"):
                    st.session_state.pop(ComparatorState, None)
                    st.rerun()
        
        if len(st.session_state.papers_db) < 2:
            st.warning("Upload at least 2 papers to compare!")
        else:
            selected_papers = st.multiselect(
                "Select papers to compare:",
                options=list(st.session_state.papers_db.keys()),
                format_func=lambda x: st.session_state.papers_db[x]["name"]
            )
            
            if len(selected_papers) >= 2 and st.button("Compare Papers", key="compare"):
                with st.spinner("Moist is comparing papers..."):
                    # Prepare paper data for comparison
                    papers_data = []
                    for paper_id in selected_papers:
                        paper = st.session_state.papers_db[paper_id]
                        # Generate quick summary for comparison
                        summary_text = " ".join([doc.page_content[:500] for doc in paper["documents"][:2]])
                        papers_data.append({
                            "title": paper["name"],
                            "summary": summary_text
                        })
                    
                    result = agents["comparator"].invoke({
                        "papers": papers_data,
                        "comparison_matrix": {},
                        "final_comparison": ""
                    })
                    
                    st.subheader("📊 Paper Comparison")
                    st.write(result["final_comparison"])            

if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv("GROQ_API_KEY"):
        st.error("Please set your GROQ_API_KEY in the project.env file!")
        st.stop()
    
main()