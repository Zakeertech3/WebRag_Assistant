# rag_pipeline.py
import logging
from typing import Dict, Any
from urllib.parse import urlparse
from crawler import WebsiteCrawler
from vector_store import VectorStore
from llm_integration import LLMIntegration

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RAGPipeline:
    def __init__(self, groq_api_key=None, firecrawl_api_key=None):
        # Use provided API keys or fall back to environment variables
        self.crawler = WebsiteCrawler(firecrawl_api_key=firecrawl_api_key)
        self.vector_store = VectorStore()
        self.llm = LLMIntegration(groq_api_key=groq_api_key)
        self.initialized = False
        self.website_url = None
    
    def extract_website_name(self, url: str) -> str:
        """Extract a name for the website from the URL"""
        parsed_url = urlparse(url)
        return parsed_url.netloc.replace(".", "_")
    
    def initialize(self, website_url: str) -> None:
        """
        Initialize the RAG pipeline for a website
        
        Args:
            website_url: URL of the website to initialize
        """
        logging.info(f"Initializing RAG pipeline for {website_url}")
        
        self.website_url = website_url
        website_name = self.extract_website_name(website_url)
        
        # Create collection
        self.vector_store.create_collection(website_name)
        
        # Crawl website
        documents = self.crawler.crawl_website(website_url)
        
        # Embed and store documents
        self.vector_store.embed_documents(documents)
        
        self.initialized = True
        logging.info(f"RAG pipeline initialized for {website_url}")
    
    def answer_question(self, query: str) -> Dict[str, Any]:
        """
        Answer a question using the RAG pipeline
        
        Args:
            query: User's question
            
        Returns:
            Dictionary containing the answer and context
        """
        if not self.initialized:
            return {
                "answer": "Please initialize the RAG pipeline with a website URL first.",
                "context": None,
                "success": False
            }
        
        try:
            # Search for relevant documents
            search_results = self.vector_store.search(query)
            
            # Format context
            context = self.llm.format_context(search_results)
            
            # Generate answer
            answer = self.llm.get_answer(query, context)
            
            return {
                "answer": answer,
                "context": search_results,
                "success": True
            }
        except Exception as e:
            logging.error(f"Error answering question: {str(e)}")
            return {
                "answer": f"An error occurred: {str(e)}",
                "context": None,
                "success": False
            }