"""
rag_pipeline.py - Core RAG Pipeline Orchestration for WebRAG Assistant

This module serves as the central orchestrator for the entire Retrieval-Augmented
Generation pipeline, connecting the crawler, vector store, and LLM components into
a cohesive system. It manages the end-to-end process from website initialization
to question answering.

Key Features:
- End-to-end RAG pipeline orchestration
- Component lifecycle management and coordination
- Error handling and graceful degradation
- Website initialization and indexing workflow
- Question answering with context retrieval

Architecture Design:
- Clean orchestration layer with minimal business logic
- Dependency injection for component flexibility
- Comprehensive error handling and state management
- URL parsing and website identification utilities

"""

import logging
from typing import Dict, Any
from urllib.parse import urlparse
from crawler import WebsiteCrawler
from vector_store import VectorStore
from llm_integration import LLMIntegration

# Configure module-level logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RAGPipeline:
    """
    Orchestrates the end-to-end RAG pipeline for website knowledge extraction and QA.
    
    This class serves as the central coordinator for the entire system, managing
    the lifecycle of crawling, embedding, retrieval, and answer generation components.
    It provides a clean interface for initializing websites and answering questions
    while handling the complexities of component interaction.
    """
    
    def __init__(self, groq_api_key=None, firecrawl_api_key=None):
        """
        Initialize the RAG pipeline with component instances.
        
        Args:
            groq_api_key: Optional API key for Groq LLM
            firecrawl_api_key: Optional API key for FireCrawl service
        """
        # Initialize component instances with dependency injection
        self.crawler = WebsiteCrawler(firecrawl_api_key=firecrawl_api_key)
        self.vector_store = VectorStore()
        self.llm = LLMIntegration(groq_api_key=groq_api_key)
        
        # Track initialization state for validation
        self.initialized = False
        self.website_url = None
    
    def extract_website_name(self, url: str) -> str:
        """
        Extract a normalized identifier for the website from its URL.
        
        Creates a consistent, filesystem-safe identifier for the website
        to use in collection naming and data persistence.
        
        Args:
            url: Website URL to process
            
        Returns:
            Normalized website identifier
        """
        parsed_url = urlparse(url)
        return parsed_url.netloc.replace(".", "_")
    
    def initialize(self, website_url: str) -> None:
        """
        Initialize the RAG pipeline for a specific website.
        
        This method orchestrates the full initialization process:
        1. Creating/selecting the vector collection
        2. Crawling the website content
        3. Processing and embedding documents
        4. Setting up the system for question answering
        
        Args:
            website_url: URL of the website to initialize
        
        Raises:
            Various exceptions from component operations with context
        """
        logging.info(f"Initializing RAG pipeline for {website_url}")
        
        # Store website URL for reference
        self.website_url = website_url
        
        # Generate consistent website identifier
        website_name = self.extract_website_name(website_url)
        
        # Initialize or select vector collection for this website
        self.vector_store.create_collection(website_name)
        
        # Crawl website and extract content
        documents = self.crawler.crawl_website(website_url)
        
        # Process and store document embeddings
        self.vector_store.embed_documents(documents)
        
        # Mark system as initialized and ready for queries
        self.initialized = True
        logging.info(f"RAG pipeline initialized for {website_url}")
    
    def answer_question(self, query: str) -> Dict[str, Any]:
        """
        Answer a question using the RAG pipeline.
        
        This method implements the full RAG workflow:
        1. Validates system initialization state
        2. Performs semantic search for relevant context
        3. Formats retrieved context for the LLM
        4. Generates a contextually-aware answer
        5. Returns comprehensive response data
        
        Args:
            query: User's question about the website
            
        Returns:
            Dictionary containing the answer, context documents, and success status
        """
        # Validate initialization state
        if not self.initialized:
            return {
                "answer": "Please initialize the RAG pipeline with a website URL first.",
                "context": None,
                "success": False
            }
        
        try:
            # Retrieve relevant documents via semantic search
            search_results = self.vector_store.search(query)
            
            # Format context for the LLM
            context = self.llm.format_context(search_results)
            
            # Generate answer based on retrieved context
            answer = self.llm.get_answer(query, context)
            
            # Return comprehensive response data
            return {
                "answer": answer,
                "context": search_results,
                "success": True
            }
        except Exception as e:
            # Handle errors with detailed logging
            logging.error(f"Error answering question: {str(e)}")
            return {
                "answer": f"An error occurred: {str(e)}",
                "context": None,
                "success": False
            }