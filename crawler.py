"""
crawler.py - Intelligent Web Crawling Module for WebRAG Assistant

This module implements robust website crawling and content extraction functionality
using the FireCrawl API. It handles the complexities of modern web page rendering,
content extraction, and document processing to prepare website content for the RAG
pipeline.

Key Features:
- JavaScript-rendered website crawling with proper rendering
- Clean content extraction with noise filtering
- Efficient text chunking for embedding optimization
- Metadata extraction and preservation for source attribution
- Rate-limiting compliance and error resilience

Architecture Design:
- Composition-based class structure for maintainability
- Asynchronous crawling with error handling for robustness
- Document processing pipeline with configurable parameters
- Comprehensive metadata preservation for traceability

"""

import logging
import time
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import MAX_PAGES_TO_CRAWL, CHUNK_SIZE, CHUNK_OVERLAP, FIRECRAWL_API_KEY

# Import FireCrawl API components
from firecrawl import FirecrawlApp, ScrapeOptions

# Configure module-level logging with standardized format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WebsiteCrawler:
    """
    Implements website crawling and content extraction functionality.
    
    This class encapsulates the logic for crawling websites, extracting clean content,
    and processing it into structured document chunks suitable for vector embedding.
    It handles the complexities of modern websites including JavaScript rendering,
    pagination, and content structure.
    """
    
    def __init__(self, firecrawl_api_key=None):
        """
        Initialize the WebsiteCrawler with text processing capabilities.
        
        Args:
            firecrawl_api_key: Optional API key for FireCrawl service, falls back to environment variable
        """
        # Initialize text splitter with configured parameters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # Set up API client with provided key or environment variable
        api_key = firecrawl_api_key or FIRECRAWL_API_KEY
        self.app = FirecrawlApp(api_key=api_key)
    
    def crawl_website(self, base_url: str) -> List[Dict[str, Any]]:
        """
        Crawl a website and extract clean, structured content.
        
        This method orchestrates the full crawling process:
        1. First scrapes the main URL to validate access
        2. Maps additional URLs within the site
        3. Processes each page into chunked documents with metadata
        4. Implements rate limiting and error handling
        
        Args:
            base_url: The website URL to crawl
            
        Returns:
            List of document chunks with comprehensive metadata
        """
        logging.info(f"Starting to crawl: {base_url}")
        documents = []
        
        try:
            # Initial validation scrape of the main URL
            scrape_result = self.app.scrape_url(
                url=base_url,
                formats=["markdown"]  # Request markdown format for LLM-optimized text
            )
            
            logging.info(f"Successfully scraped the main URL")
            
            # Process the main URL result
            if hasattr(scrape_result, 'markdown') and scrape_result.markdown:
                markdown_content = scrape_result.markdown
                
                # Extract metadata for document attribution
                metadata = {}
                title = "Untitled"
                
                if hasattr(scrape_result, 'metadata'):
                    metadata = scrape_result.metadata
                    if hasattr(metadata, 'title'):
                        title = metadata.title
                
                # Split content into optimally sized chunks for embedding
                chunks = self.text_splitter.split_text(markdown_content)
                
                # Create document objects with rich metadata
                for i, chunk in enumerate(chunks):
                    documents.append({
                        "page_content": chunk,
                        "metadata": {
                            "source": base_url,
                            "title": title,
                            "chunk_index": i,
                            "page_index": 0  # Main page is index 0
                        }
                    })
                
                logging.info(f"Created {len(chunks)} document chunks from main URL")
            
            # Discover and process additional pages within the site
            try:
                # Map the site to find linked pages
                mapped_urls = self.app.map_url(url=base_url, limit=MAX_PAGES_TO_CRAWL)
                
                # Extract additional URLs from mapping result
                additional_urls = []
                if hasattr(mapped_urls, 'links'):
                    additional_urls = mapped_urls.links
                    logging.info(f"Found {len(additional_urls)} additional URLs to crawl")
                
                # Process each additional URL with rate limiting
                for page_index, url in enumerate(additional_urls[:MAX_PAGES_TO_CRAWL-1]):
                    try:
                        # Implement rate limiting to respect service limits
                        time.sleep(1)
                        
                        # Scrape the current URL
                        page_result = self.app.scrape_url(url=url, formats=["markdown"])
                        
                        # Process page content if available
                        if hasattr(page_result, 'markdown') and page_result.markdown:
                            page_content = page_result.markdown
                            
                            # Extract page metadata
                            page_title = "Untitled"
                            if hasattr(page_result, 'metadata') and hasattr(page_result.metadata, 'title'):
                                page_title = page_result.metadata.title
                            
                            # Split page content into optimized chunks
                            page_chunks = self.text_splitter.split_text(page_content)
                            
                            # Create document objects with consistent metadata structure
                            for i, chunk in enumerate(page_chunks):
                                documents.append({
                                    "page_content": chunk,
                                    "metadata": {
                                        "source": url,
                                        "title": page_title,
                                        "chunk_index": i,
                                        "page_index": page_index + 1  # 1-based indexing for additional pages
                                    }
                                })
                            
                            logging.info(f"Added {len(page_chunks)} chunks from {url}")
                    except Exception as e:
                        # Handle per-page errors gracefully to continue processing other pages
                        logging.warning(f"Error processing URL {url}: {str(e)}")
                        continue
            except Exception as e:
                # Handle URL mapping errors while preserving main page results
                logging.warning(f"Error mapping URLs: {str(e)}")
            
            logging.info(f"Created a total of {len(documents)} document chunks")
            return documents
            
        except Exception as e:
            # Handle critical errors with detailed logging
            logging.error(f"Error during crawling: {str(e)}")
            # Return empty document list if crawling fails completely
            return []