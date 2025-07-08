import os
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """Handles vector storage operations using ChromaDB."""
    
    def __init__(self, persist_directory: str = "./chroma_db", embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the vector database
            embedding_model: Name of the sentence transformer model to use
        """
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
        self.vector_store = None
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
    
    def create_vector_store(self, documents: List[Document], collection_name: str = "pdf_documents") -> Chroma:
        """
        Create a vector store from documents.
        
        Args:
            documents: List of Document objects
            collection_name: Name of the collection in ChromaDB
            
        Returns:
            Chroma vector store instance
        """
        try:
            logger.info(f"Creating vector store with {len(documents)} documents")
            logger.info(f"Using embedding model: {self.embedding_model}")
            
            # Create vector store
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=collection_name
            )
            
            # Persist the vector store
            vector_store.persist()
            self.vector_store = vector_store
            
            logger.info("Vector store created and persisted successfully")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def load_existing_vector_store(self, collection_name: str = "pdf_documents") -> Chroma:
        """
        Load an existing vector store from disk.
        
        Args:
            collection_name: Name of the collection to load
            
        Returns:
            Chroma vector store instance
        """
        try:
            logger.info(f"Loading existing vector store from {self.persist_directory}")
            
            vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=collection_name
            )
            
            self.vector_store = vector_store
            logger.info("Vector store loaded successfully")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query
            k: Number of similar documents to retrieve
            
        Returns:
            List of similar Document objects
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Please create or load a vector store first.")
        
        try:
            logger.info(f"Performing similarity search for query: '{query}'")
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Retrieved {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise
    
    def get_collection_info(self) -> dict:
        """
        Get information about the vector store collection.
        
        Returns:
            Dictionary with collection statistics
        """
        if self.vector_store is None:
            return {"error": "Vector store not initialized"}
        
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            return {
                "total_documents": count,
                "embedding_model": self.embedding_model,
                "persist_directory": self.persist_directory
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {"error": str(e)}
    
    def delete_collection(self, collection_name: str = "pdf_documents"):
        """
        Delete a collection from the vector store.
        
        Args:
            collection_name: Name of the collection to delete
        """
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_directory)
            client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise 