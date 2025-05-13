# crawler.py
import logging
import time
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import MAX_PAGES_TO_CRAWL, CHUNK_SIZE, CHUNK_OVERLAP, FIRECRAWL_API_KEY

# Import the correct FireCrawl classes
from firecrawl import FirecrawlApp, ScrapeOptions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WebsiteCrawler:
    def __init__(self, firecrawl_api_key=None):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # Use provided API key or fall back to environment variable
        api_key = firecrawl_api_key or FIRECRAWL_API_KEY
        # Initialize FirecrawlApp with API key
        self.app = FirecrawlApp(api_key=api_key)
    
    def crawl_website(self, base_url: str) -> List[Dict[str, Any]]:
        """
        Crawl a website and extract clean, structured content
        
        Args:
            base_url: The website URL to crawl
            
        Returns:
            List of document chunks with metadata
        """
        logging.info(f"Starting to crawl: {base_url}")
        documents = []
        
        try:
            # First, scrape the single URL to ensure it works
            scrape_result = self.app.scrape_url(
                url=base_url,
                formats=["markdown"]
            )
            
            logging.info(f"Successfully scraped the main URL")
            
            # The response is an object, not a dictionary, so we need to access attributes directly
            # First, check if the response has a markdown attribute
            if hasattr(scrape_result, 'markdown') and scrape_result.markdown:
                markdown_content = scrape_result.markdown
                
                # Get metadata if available
                metadata = {}
                title = "Untitled"
                
                if hasattr(scrape_result, 'metadata'):
                    metadata = scrape_result.metadata
                    if hasattr(metadata, 'title'):
                        title = metadata.title
                
                # Split text into chunks
                chunks = self.text_splitter.split_text(markdown_content)
                
                # Create document objects with metadata
                for i, chunk in enumerate(chunks):
                    documents.append({
                        "page_content": chunk,
                        "metadata": {
                            "source": base_url,
                            "title": title,
                            "chunk_index": i,
                            "page_index": 0
                        }
                    })
                
                logging.info(f"Created {len(chunks)} document chunks from main URL")
            
            # Now try to get additional URLs if available
            try:
                mapped_urls = self.app.map_url(url=base_url, limit=MAX_PAGES_TO_CRAWL)
                
                # Check if mapped_urls has a links attribute
                additional_urls = []
                if hasattr(mapped_urls, 'links'):
                    additional_urls = mapped_urls.links
                    logging.info(f"Found {len(additional_urls)} additional URLs to crawl")
                
                # Process a limited number of additional URLs
                for page_index, url in enumerate(additional_urls[:MAX_PAGES_TO_CRAWL-1]):
                    try:
                        # Avoid rate limiting
                        time.sleep(1)
                        
                        # Scrape this URL
                        page_result = self.app.scrape_url(url=url, formats=["markdown"])
                        
                        # Check if page_result has a markdown attribute
                        if hasattr(page_result, 'markdown') and page_result.markdown:
                            page_content = page_result.markdown
                            
                            # Get metadata
                            page_title = "Untitled"
                            if hasattr(page_result, 'metadata') and hasattr(page_result.metadata, 'title'):
                                page_title = page_result.metadata.title
                            
                            # Split text into chunks
                            page_chunks = self.text_splitter.split_text(page_content)
                            
                            # Create document objects with metadata
                            for i, chunk in enumerate(page_chunks):
                                documents.append({
                                    "page_content": chunk,
                                    "metadata": {
                                        "source": url,
                                        "title": page_title,
                                        "chunk_index": i,
                                        "page_index": page_index + 1
                                    }
                                })
                            
                            logging.info(f"Added {len(page_chunks)} chunks from {url}")
                    except Exception as e:
                        logging.warning(f"Error processing URL {url}: {str(e)}")
                        continue
            except Exception as e:
                logging.warning(f"Error mapping URLs: {str(e)}")
            
            logging.info(f"Created a total of {len(documents)} document chunks")
            return documents
            
        except Exception as e:
            logging.error(f"Error during crawling: {str(e)}")
            # Return empty document list if crawling fails
            return []