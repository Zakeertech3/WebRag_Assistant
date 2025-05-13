"""
config.py - Configuration Management for WebRAG Assistant

This module centralizes all configuration parameters for the WebRAG Assistant application,
providing a single source of truth for settings across the entire system. It handles
environment variable loading, API credentials, and tunable parameters for each component
of the RAG pipeline.

Design Principles:
- Separation of configuration from implementation logic
- Environment-based configuration with sensible defaults
- Centralized parameter management for consistent system behavior
- Clear documentation of configuration options and their effects

"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file for flexible deployment
load_dotenv()

# API keys and model configuration settings
# These parameters define the core AI services used by the application
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dimensional embeddings with good performance/efficiency balance
LLM_MODEL_NAME = "llama3-8b-8192"  # Selected for strong reasoning with reasonable latency

# FireCrawl website crawling settings
# These parameters control the scope and behavior of the web crawler
MAX_PAGES_TO_CRAWL = 20  # Reduced from 50 to 20 for faster testing and lower API usage
CRAWL_DEPTH = 2  # Limit crawl depth to prevent excessive page exploration
EXCLUDE_URLS = [  # URLs patterns to skip during crawling to avoid non-content pages
    "*/privacy-policy*",
    "*/terms-of-service*",
    "*/login*",
    "*/signup*",
]

# ChromaDB vector database settings
# Directory where the vector database will be persisted between sessions
CHROMA_PERSIST_DIRECTORY = "./chroma_db"

# Text processing and chunking configuration
# These parameters affect how documents are split for embedding and retrieval
CHUNK_SIZE = 500  # Size of text chunks in characters (balance between context and granularity)
CHUNK_OVERLAP = 50  # Overlap between chunks to maintain context across chunk boundaries

# RAG retrieval settings
# These parameters control search behavior and response generation
TOP_K_RESULTS = 8  # Number of documents to retrieve (increased from 5 to provide more context)
SIMILARITY_THRESHOLD = 0.5  # Minimum similarity score for including documents (lowered from 0.7 to catch more relevant documents)