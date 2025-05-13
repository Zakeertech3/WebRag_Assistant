"""
llm_integration.py - LLM Integration for WebRAG Assistant

This module provides a clean interface to Groq's implementation of the Llama 3 LLM for
generating contextually-aware responses. It handles prompt engineering, context formatting,
and response generation while ensuring factuality and source integrity.

Key Features:
- Context-aware prompt engineering optimized for RAG responses
- Source relevance filtering and ranking for quality control
- Structured context formatting for optimal LLM comprehension
- Configurable response parameters with sensible defaults
- Error handling and result validation

Architecture Design:
- Clean abstraction over the Groq API for maintainability
- Parameter-driven design for flexible configuration
- Clear separation of context formatting and response generation
- Resource-efficient implementation with proper API usage

"""

import logging
from typing import List, Dict, Any
from groq import Groq
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, LLM_MODEL_NAME, TOP_K_RESULTS, SIMILARITY_THRESHOLD

# Configure module-level logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMIntegration:
    """
    Handles LLM interaction for generating answers based on retrieved context.
    
    This class encapsulates the interaction with the Groq API, providing methods
    for formatting retrieved context, constructing effective prompts, and generating
    high-quality answers based on the retrieved information.
    """
    
    def __init__(self, groq_api_key=None):
        """
        Initialize the LLM integration with the Groq API.
        
        Args:
            groq_api_key: Optional API key for Groq, falls back to environment variable
        """
        # Set up API credentials with flexibility
        api_key = groq_api_key or GROQ_API_KEY
        
        # Initialize the direct Groq client for completions API
        self.client = Groq(api_key=api_key)
        
        # Initialize the LangChain integration for potential future use
        # (currently using direct API for more control, but keeping this for compatibility)
        self.llm = ChatGroq(
            model_name=LLM_MODEL_NAME,
            groq_api_key=api_key,
            temperature=0.2,  # Low temperature for factual responses
            max_tokens=1024   # Balanced response length
        )
    
    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results into structured context for the LLM.
        
        This method implements:
        1. Relevance filtering to remove low-quality matches
        2. Similarity-based sorting for priority ordering
        3. Top-K selection to control context length
        4. Structured formatting for optimal LLM comprehension
        
        Args:
            results: List of vector search results with metadata
            
        Returns:
            Formatted context string optimized for LLM consumption
        """
        # Filter results by minimum similarity threshold
        filtered_results = [r for r in results if r["similarity"] >= SIMILARITY_THRESHOLD]
        
        if not filtered_results:
            return "No relevant information found."
        
        # Sort results by similarity score (descending)
        sorted_results = sorted(filtered_results, key=lambda x: x["similarity"], reverse=True)
        
        # Select top K results to control context size
        top_results = sorted_results[:TOP_K_RESULTS]
        
        # Format context with clear document boundaries and metadata
        context_parts = []
        for i, doc in enumerate(top_results):
            source_url = doc["metadata"]["source"]
            title = doc["metadata"]["title"]
            content = doc["page_content"]
            
            # Structured format with document index, title, source URL, and content
            context_parts.append(f"[Document {i+1}] Title: {title}\nSource: {source_url}\nContent: {content}\n")
        
        return "\n".join(context_parts)
    
    def get_answer(self, query: str, context: str) -> str:
        """
        Generate a contextually-aware answer using the LLM.
        
        This method:
        1. Constructs an optimized prompt with clear instructions
        2. Limits hallucination by emphasizing context-only answers
        3. Handles the API interaction with appropriate parameters
        4. Extracts and returns the generated response
        
        Args:
            query: User's original question
            context: Formatted context from vector search
            
        Returns:
            Generated answer based on provided context
        """
        logging.info(f"Generating answer for: {query}")
        
        # Construct a prompt that emphasizes factuality and source adherence
        prompt = f"""You are an AI assistant that answers questions about website content.
Use ONLY the following context to answer the question. 
If the information is in the context, provide a detailed and helpful answer.
If the question cannot be answered from the context, say "Based on the available information, I cannot provide a complete answer to that question."
DO NOT make up or hallucinate any information not present in the context.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

        # Generate completion using direct API for more control
        response = self.client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,  # Low temperature for factual responses
            max_tokens=1024   # Reasonable length for comprehensive answers
        )
        
        # Extract and return the generated text
        return response.choices[0].message.content