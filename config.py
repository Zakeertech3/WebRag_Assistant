# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API keys and model settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3-8b-8192"

# FireCrawl settings
MAX_PAGES_TO_CRAWL = 20  # Reduced from 50 to 20 for faster testing
CRAWL_DEPTH = 2
EXCLUDE_URLS = [
    "*/privacy-policy*",
    "*/terms-of-service*",
    "*/login*",
    "*/signup*",
]

# ChromaDB settings
CHROMA_PERSIST_DIRECTORY = "./chroma_db"

# Text processing settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# RAG settings
TOP_K_RESULTS = 8  # Increased from 5 to provide more context
SIMILARITY_THRESHOLD = 0.5  # Lowered from 0.7 to catch more relevant documents