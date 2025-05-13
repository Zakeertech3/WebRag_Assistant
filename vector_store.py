# vector_store.py
import os
import logging
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from config import CHROMA_PERSIST_DIRECTORY, EMBED_MODEL_NAME

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorStore:
    def __init__(self):
        # Initialize embedding model
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        
        # Make sure the directory exists
        os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        self.collection = None
    
    def create_collection(self, website_name: str) -> None:
        """
        Create or get a collection for the website
        
        Args:
            website_name: Name of the website to create a collection for
        """
        # Get all existing collections
        existing_collections = self.client.list_collections()
        collection_names = [collection.name for collection in existing_collections]
        
        # Check if collection exists
        if website_name in collection_names:
            logging.info(f"Using existing collection for {website_name}")
            self.collection = self.client.get_collection(name=website_name)
        else:
            logging.info(f"Creating new collection for {website_name}")
            self.collection = self.client.create_collection(
                name=website_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def embed_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Embed documents and store them in the vector database
        
        Args:
            documents: List of document chunks with metadata
        """
        if not self.collection:
            raise ValueError("Collection not initialized. Call create_collection first.")
        
        logging.info(f"Embedding {len(documents)} documents")
        
        # Extract text and generate IDs
        texts = [doc["page_content"] for doc in documents]
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Generate embeddings
        embeddings = self.embed_model.encode(texts)
        
        # Prepare metadatas
        metadatas = [doc["metadata"] for doc in documents]
        
        # Add to collection in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            end_idx = min(i + batch_size, len(texts))
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx].tolist(),
                metadatas=metadatas[i:end_idx],
                documents=texts[i:end_idx]
            )
            
        logging.info(f"Added {len(documents)} documents to the vector store")
    
    def search(self, query: str, k: int = 8) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query
        
        Args:
            query: The search query
            k: Number of results to return (increased from default 5)
            
        Returns:
            List of relevant documents with metadata
        """
        if not self.collection:
            raise ValueError("Collection not initialized. Call create_collection first.")
        
        # Generate query embedding
        query_embedding = self.embed_model.encode(query).tolist()
        
        # Search the collection with more results
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        documents = []
        for i in range(len(results["documents"][0])):
            documents.append({
                "page_content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity": 1 - results["distances"][0][i]  # Convert distance to similarity
            })
            
        return documents