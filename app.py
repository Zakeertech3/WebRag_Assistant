"""
app.py - Streamlit-based Web Interface for WebRAG Assistant

This module serves as the main entry point for the WebRAG Assistant application, providing a 
responsive web interface built with Streamlit. It orchestrates the user interaction flow from
API key setup to website indexing and conversational question answering with source citation.

Architecture Design:
- Implements a stateful web application using Streamlit's session state
- Provides a secure interface for API key management
- Establishes a responsive, chat-based UI with source attribution
- Handles error states gracefully with informative user feedback
- Implements resource caching for improved performance

"""

import streamlit as st
import logging
import os
import sys
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline

# Configure comprehensive logging to streamline debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Load environment variables for configuration flexibility
load_dotenv()


# Define custom CSS styling for enhanced UI components and source citation display
st.markdown("""
<style>
.source-item {
    border-left: 3px solid #4361ee;
    padding-left: 10px;
    margin-bottom: 15px;
}
.source-title {
    font-weight: bold;
    margin-bottom: 5px;
}
.source-meta {
    color: #555;
    font-size: 0.9em;
    margin-bottom: 5px;
}
.relevance-high {
    color: #10b981;
    font-weight: bold;
}
.relevance-medium {
    color: #f59e0b;
    font-weight: bold;
}
.relevance-low {
    color: #ef4444;
    font-weight: bold;
}
.api-key-input {
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


# Application header with branding elements
st.title("WebRAG Assistant 📊")
st.subheader("Retrieval-Augmented Generation for websites")

# Initialize session state variables for persistence across interactions
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""
if "firecrawl_api_key" not in st.session_state:
    st.session_state.firecrawl_api_key = ""

def update_api_keys():
    """
    Update session state with user-provided API keys.
    Centralizes API key management to maintain secure storage patterns.
    """
    st.session_state.groq_api_key = groq_key
    st.session_state.firecrawl_api_key = firecrawl_key

# API Key configuration interface with expandable section for improved UX
with st.expander("API Keys Setup", expanded=not (st.session_state.groq_api_key and st.session_state.firecrawl_api_key)):
    st.markdown("Enter your API keys to use the application:")
    
    # Two-column layout for API key inputs
    col1, col2 = st.columns(2)
    with col1:
        groq_key = st.text_input(
            "Groq API Key:", 
            value="",
            type="password",  # Securely mask input
            help="Get your API key from https://console.groq.com",
            key="groq_key_input"
        )
    with col2:
        firecrawl_key = st.text_input(
            "FireCrawl API Key:", 
            value="",
            type="password",  # Securely mask input
            help="Get your API key from https://firecrawl.dev",
            key="firecrawl_key_input"
        )
    
    # API key submission button with success feedback
    if st.button("Save API Keys"):
        update_api_keys()
        st.success("API keys saved!")

@st.cache_resource
def get_rag_pipeline(_groq_key, _firecrawl_key):
    """
    Create and cache a RAG pipeline instance to optimize resource utilization.
    
    This decorator ensures we don't recreate the pipeline on every UI interaction,
    significantly improving application performance and reducing API calls.
    
    Args:
        _groq_key: Groq API key for LLM access
        _firecrawl_key: FireCrawl API key for web crawling
        
    Returns:
        RAGPipeline: Initialized pipeline instance
    """
    logging.info("Creating new RAG pipeline instance")
    return RAGPipeline(groq_api_key=_groq_key, firecrawl_api_key=_firecrawl_key)

# Website URL input form with initialization button
with st.form(key="website_form"):
    website_url = st.text_input("Enter website URL:", placeholder="https://example.com")
    col1, col2 = st.columns([1, 3])
    with col1:
        submit_button = st.form_submit_button("Initialize")
    with col2:
        # Visual feedback for initialization status
        if "initialized" in st.session_state and st.session_state["initialized"]:
            st.success("Ready to answer questions")

# Handle website initialization with comprehensive error handling
if submit_button and website_url:
    if not st.session_state.groq_api_key or not st.session_state.firecrawl_api_key:
        st.error("Please provide both Groq and FireCrawl API keys.")
    else:
        with st.spinner("Crawling and indexing website... This may take a few minutes."):
            try:
                # Initialize pipeline with user-provided website
                pipeline = get_rag_pipeline(st.session_state.groq_api_key, st.session_state.firecrawl_api_key)
                pipeline.initialize(website_url)
                st.session_state["initialized"] = True
                st.success(f"Successfully indexed {website_url}")
            except Exception as e:
                # Detailed error handling with user-friendly messages
                st.error(f"Error initializing: {str(e)}")
                logging.error(f"Error initializing pipeline: {str(e)}", exc_info=True)
                st.session_state["initialized"] = False


# Initialize chat history in session state if not present
if "messages" not in st.session_state:
    st.session_state.messages = []


# Render existing chat messages with source attribution
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display source citations for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("View Sources"):
                for i, source in enumerate(message["sources"]):
                    # Visual indication of source relevance with color coding
                    relevance_class = "relevance-low"
                    if source['similarity'] > 0.8:
                        relevance_class = "relevance-high"
                    elif source['similarity'] > 0.6:
                        relevance_class = "relevance-medium"
                    
                    # Structured source display with metadata and content preview
                    st.markdown(f"""
                    <div class="source-item">
                      <div class="source-title">Source {i+1}: {source['title']}</div>
                      <div class="source-meta">URL: <a href="{source['url']}" target="_blank">{source['url']}</a></div>
                      <div class="source-meta">Relevance: <span class="{relevance_class}">{source['similarity']:.2f}</span></div>
                      <div>Content: {source['content'][:300]}...</div>
                    </div>
                    """, unsafe_allow_html=True)


# Chat interface for asking questions about the website
if "initialized" in st.session_state and st.session_state["initialized"]:
    if prompt := st.chat_input("Ask a question about this website"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response with visual loading indicator
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get pipeline instance and generate response
                    pipeline = get_rag_pipeline(st.session_state.groq_api_key, st.session_state.firecrawl_api_key)
                    response = pipeline.answer_question(prompt)
                    
                    if response["success"]:
                        # Display successful response
                        st.markdown(response["answer"])
                        
                        # Process and store source information for citation
                        if response["context"]:
                            sources = []
                            for doc in response["context"]:
                                sources.append({
                                    "title": doc["metadata"]["title"],
                                    "url": doc["metadata"]["source"],
                                    "similarity": doc["similarity"],
                                    "content": doc["page_content"]
                                })
                            
                            # Add response with sources to chat history
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response["answer"],
                                "sources": sources
                            })
                        else:
                            # Add basic response without sources to chat history
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response["answer"]
                            })
                    else:
                        # Display error message from response
                        st.error(response["answer"])
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response["answer"]
                        })
                except Exception as e:
                    # Handle unexpected errors with detailed logging and user feedback
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    logging.error(error_msg, exc_info=True)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"I encountered an error while processing your question: {str(e)}"
                    })
else:
    # Provide appropriate guidance based on application state
    if st.session_state.groq_api_key and st.session_state.firecrawl_api_key:
        st.info("Please enter a website URL and click 'Initialize' to start chatting.")
    else:
        st.warning("Please enter your API keys in the setup section above.")


# Sidebar with application instructions and usage notes
with st.sidebar:
    st.subheader("How to use")
    st.markdown("""
    1. Enter your API keys in the setup section
    2. Enter a website URL and click 'Initialize'
    3. Wait for the website to be crawled and indexed
    4. Ask questions about the website content
    5. View sources used for each answer
    """)
    
    st.subheader("Notes")
    st.markdown("""
    - Initialization may take a few minutes depending on website size
    - Only the first 20 pages will be crawled by default
    - Results are cached for the duration of the session
    """)