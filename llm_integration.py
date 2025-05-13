# llm_integration.py
import logging
from typing import List, Dict, Any
from groq import Groq
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, LLM_MODEL_NAME, TOP_K_RESULTS, SIMILARITY_THRESHOLD

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMIntegration:
    def __init__(self, groq_api_key=None):
        # Use provided API key or fall back to environment variable
        api_key = groq_api_key or GROQ_API_KEY
        
        # Initialize Groq client
        self.client = Groq(api_key=api_key)
        
        # Initialize LangChain ChatGroq
        self.llm = ChatGroq(
            model_name=LLM_MODEL_NAME,
            groq_api_key=api_key,
            temperature=0.2,
            max_tokens=1024
        )
    
    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results into context for the LLM
        
        Args:
            results: List of search results
            
        Returns:
            Formatted context string
        """
        # Filter results by similarity threshold
        filtered_results = [r for r in results if r["similarity"] >= SIMILARITY_THRESHOLD]
        
        if not filtered_results:
            return "No relevant information found."
        
        # Sort by similarity (highest first)
        sorted_results = sorted(filtered_results, key=lambda x: x["similarity"], reverse=True)
        
        # Take top K results
        top_results = sorted_results[:TOP_K_RESULTS]
        
        # Format context
        context_parts = []
        for i, doc in enumerate(top_results):
            source_url = doc["metadata"]["source"]
            title = doc["metadata"]["title"]
            content = doc["page_content"]
            
            context_parts.append(f"[Document {i+1}] Title: {title}\nSource: {source_url}\nContent: {content}\n")
        
        return "\n".join(context_parts)
    
    def get_answer(self, query: str, context: str) -> str:
        """
        Generate an answer using the LLM based on the context
        
        Args:
            query: User's question
            context: Context from vector search
            
        Returns:
            Generated answer
        """
        logging.info(f"Generating answer for: {query}")
        
        # Create a more directive prompt
        prompt = f"""You are an AI assistant that answers questions about website content.
Use ONLY the following context to answer the question. 
If the information is in the context, provide a detailed and helpful answer.
If the question cannot be answered from the context, say "Based on the available information, I cannot provide a complete answer to that question."
DO NOT make up or hallucinate any information not present in the context.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

        # Generate response
        response = self.client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024
        )
        
        return response.choices[0].message.content