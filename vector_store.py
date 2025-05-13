"""
vector_store.py - Vector Database Integration for WebRAG Assistant

This module implements the vector storage and retrieval functionality using ChromaDB,
providing semantic search capabilities through document embeddings. It handles the
storage, indexing, and retrieval of document vectors with associated metadata.

Key Features:
- Embedding generation with Sentence Transformers
- Vector database integration with ChromaDB
- Collection management for multi-website support
- Efficient batched operations for large document sets
- Semantic similarity search with distance-to-similarity conversion

Architecture Design:
- Abstraction layer over ChromaDB for simplified interaction
- Efficient batch processing for performance optimization
- Comprehensive metadata preservation for document traceability
- Persistent storage with proper state handling

"""

import os
import logging
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from config import CHROMA_PERSIST_DIRECTORY, EMBED_MODEL_NAME

# Configure module-level logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorStore:
    """
    Manages vector embeddings and semantic search functionality.
    
    This class encapsulates the vector database operations, providing methods
    for collection management, document embedding, and semantic similarity search.
    It serves as the retrieval component in the RAG pipeline, connecting user
    queries to relevant document chunks.
    """
    
    def __init__(self):
        """
        Initialize the vector store with embedding model and database connection.
        
        Sets up the embedding model, ensures storage directory exists, and
        establishes a connection to the persistent ChromaDB instance.
        """
        # Initialize the embedding model for vector generation
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        
        # Ensure persistence directory exists
        os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        
        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        self.collection = None
    
    def create_collection(self, website_name: str) -> None:
        """
        Create or retrieve a vector collection for a specific website.
        
        This method ensures that each website has its own isolated collection
        in the vector database, allowing for efficient multi-website support
        and proper data isolation between sites.
        
        Args:
            website_name: Normalized identifier for the website
            
        Sets:
            self.collection: The active ChromaDB collection for operations
        """
        # Get all existing collections
        existing_collections = self.client.list_collections()
        collection_names = [collection.name for collection in existing_collections]
        
        # Check if collection already exists for this website
        if website_name in collection_names:
            logging.info(f"Using existing collection for {website_name}")
            self.collection = self.client.get_collection(name=website_name)
        else:
            # Create new collection with cosine similarity metric
            logging.info(f"Creating new collection for {website_name}")
            self.collection = self.client.create_collection(
                name=website_name,
                metadata={"hnsw:space": "cosine"}  # Configure for cosine similarity
            )
    
    def embed_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Embed documents and store them in the vector database.
        
        This method processes a list of document chunks:
        1. Extracts text content for embedding
        2. Generates unique document IDs
        3. Computes vector embeddings using the model
        4. Stores vectors with associated metadata and text
        5. Processes in batches for memory efficiency
        
        Args:
            documents: List of document chunks with metadata
            
        Raises:
            ValueError: If collection is not initialized
        """
        # Validate collection initialization
        if not self.collection:
            raise ValueError("Collection not initialized. Call create_collection first.")
        
        logging.info(f"Embedding {len(documents)} documents")
        
        # Extract text content and generate sequential IDs
        texts = [doc["page_content"] for doc in documents]
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Generate embeddings for all texts in one batch for efficiency
        embeddings = self.embed_model.encode(texts)
        
        # Extract metadata dictionaries
        metadatas = [doc["metadata"] for doc in documents]
        
        # Process in batches to manage memory usage for large document sets
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            end_idx = min(i + batch_size, len(texts))
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx].tolist(),  # Convert numpy arrays to lists
                metadatas=metadatas[i:end_idx],
                documents=texts[i:end_idx]
            )
            
        logging.info(f"Added {len(documents)} documents to the vector store")
    
    def search(self, query: str, k: int = 8) -> List[Dict[str, Any]]:
        """
        Perform semantic search for documents similar to the query.
        
        This method:
        1. Encodes the query text into a vector embedding
        2. Searches the collection for similar vectors
        3. Processes results into a standardized format
        4. Converts distance scores to similarity scores
        
        Args:
            query: The search query text
            k: Number of results to return (default: 8)
            
        Returns:
            List of relevant documents with metadata and similarity scores
            
        Raises:
            ValueError: If collection is not initialized
        """
        # Validate collection initialization
        if not self.collection:
            raise ValueError("Collection not initialized. Call create_collection first.")
        
        # Generate query embedding vector
        query_embedding = self.embed_model.encode(query).tolist()
        
        # Search the collection with specified result count
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]  # Request all result components
        )
        
        # Format results into a standardized structure
        documents = []
        for i in range(len(results["documents"][0])):
            documents.append({
                "page_content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity": 1 - results["distances"][0][i]  # Convert distance to similarity score (0-1)
            })
            
        return documents