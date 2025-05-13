import streamlit as st
import logging
import os
import sys
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Load environment variables from .env (as fallback)
load_dotenv()

# Add minimal custom CSS for better readability
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

# App title and description
st.title("WebRAG Assistant 📊")
st.subheader("Retrieval-Augmented Generation for websites")

# Initialize session state for API keys
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""
if "firecrawl_api_key" not in st.session_state:
    st.session_state.firecrawl_api_key = ""

# Function to update session state with API keys
def update_api_keys():
    st.session_state.groq_api_key = groq_key
    st.session_state.firecrawl_api_key = firecrawl_key

# API Key input form
with st.expander("API Keys Setup", expanded=not (st.session_state.groq_api_key and st.session_state.firecrawl_api_key)):
    st.markdown("Enter your API keys to use the application:")
    
    col1, col2 = st.columns(2)
    with col1:
        groq_key = st.text_input(
            "Groq API Key:", 
            value="",
            type="password",
            help="Get your API key from https://console.groq.com",
            key="groq_key_input"
        )
    with col2:
        firecrawl_key = st.text_input(
            "FireCrawl API Key:", 
            value="",
            type="password",
            help="Get your API key from https://firecrawl.dev",
            key="firecrawl_key_input"
        )
    
    if st.button("Save API Keys"):
        update_api_keys()
        st.success("API keys saved!")

# Initialize the RAG pipeline
@st.cache_resource
def get_rag_pipeline(_groq_key, _firecrawl_key):
    logging.info("Creating new RAG pipeline instance")
    # We'll pass the API keys directly to the pipeline
    return RAGPipeline(groq_api_key=_groq_key, firecrawl_api_key=_firecrawl_key)

# Website URL input
with st.form(key="website_form"):
    website_url = st.text_input("Enter website URL:", placeholder="https://example.com")
    col1, col2 = st.columns([1, 3])
    with col1:
        submit_button = st.form_submit_button("Initialize")
    with col2:
        if "initialized" in st.session_state and st.session_state["initialized"]:
            st.success("Ready to answer questions")

# Handle initialization
if submit_button and website_url:
    # Check if API keys are provided
    if not st.session_state.groq_api_key or not st.session_state.firecrawl_api_key:
        st.error("Please provide both Groq and FireCrawl API keys.")
    else:
        with st.spinner("Crawling and indexing website... This may take a few minutes."):
            try:
                pipeline = get_rag_pipeline(st.session_state.groq_api_key, st.session_state.firecrawl_api_key)
                pipeline.initialize(website_url)
                st.session_state["initialized"] = True
                st.success(f"Successfully indexed {website_url}")
            except Exception as e:
                st.error(f"Error initializing: {str(e)}")
                logging.error(f"Error initializing pipeline: {str(e)}", exc_info=True)
                st.session_state["initialized"] = False

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("View Sources"):
                for i, source in enumerate(message["sources"]):
                    # Determine relevance class
                    relevance_class = "relevance-low"
                    if source['similarity'] > 0.8:
                        relevance_class = "relevance-high"
                    elif source['similarity'] > 0.6:
                        relevance_class = "relevance-medium"
                    
                    st.markdown(f"""
                    <div class="source-item">
                      <div class="source-title">Source {i+1}: {source['title']}</div>
                      <div class="source-meta">URL: <a href="{source['url']}" target="_blank">{source['url']}</a></div>
                      <div class="source-meta">Relevance: <span class="{relevance_class}">{source['similarity']:.2f}</span></div>
                      <div>Content: {source['content'][:300]}...</div>
                    </div>
                    """, unsafe_allow_html=True)

# User input
if "initialized" in st.session_state and st.session_state["initialized"]:
    if prompt := st.chat_input("Ask a question about this website"):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate a response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    pipeline = get_rag_pipeline(st.session_state.groq_api_key, st.session_state.firecrawl_api_key)
                    response = pipeline.answer_question(prompt)
                    
                    if response["success"]:
                        st.markdown(response["answer"])
                        
                        # Prepare sources for display
                        if response["context"]:
                            sources = []
                            for doc in response["context"]:
                                sources.append({
                                    "title": doc["metadata"]["title"],
                                    "url": doc["metadata"]["source"],
                                    "similarity": doc["similarity"],
                                    "content": doc["page_content"]
                                })
                            
                            # Save the message with sources
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response["answer"],
                                "sources": sources
                            })
                        else:
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response["answer"]
                            })
                    else:
                        st.error(response["answer"])
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response["answer"]
                        })
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    logging.error(error_msg, exc_info=True)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"I encountered an error while processing your question: {str(e)}"
                    })
else:
    if st.session_state.groq_api_key and st.session_state.firecrawl_api_key:
        st.info("Please enter a website URL and click 'Initialize' to start chatting.")
    else:
        st.warning("Please enter your API keys in the setup section above.")

# Sidebar with instructions
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