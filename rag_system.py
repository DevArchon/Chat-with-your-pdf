import os
from typing import Optional, List, Dict
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from llm_interface import LLMInterface
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """Main RAG system that orchestrates PDF processing, vector storage, and LLM interactions."""
    
    def __init__(self, 
                 llm_type: str = "gemini",
                 api_key: Optional[str] = None,
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the RAG system.
        
        Args:
            llm_type: Type of LLM to use ("gemini" or "ollama")
            api_key: API key for Gemini (if using Gemini)
            persist_directory: Directory to persist vector database
            embedding_model: Name of the sentence transformer model
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.llm_type = llm_type
        self.api_key = api_key
        self.persist_directory = persist_directory
        
        # Initialize components
        self.pdf_processor = PDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_store = VectorStore(persist_directory=persist_directory, embedding_model=embedding_model)
        self.llm_interface = LLMInterface(llm_type=llm_type, api_key=api_key)
        
        # State tracking
        self.is_initialized = False
        self.current_pdf_path = None
        
        logger.info(f"RAG System initialized with LLM type: {llm_type}")
    
    def load_pdf(self, pdf_path: str, collection_name: str = "pdf_documents") -> Dict:
        """
        Load and process a PDF file, creating embeddings and storing them in the vector database.
        
        Args:
            pdf_path: Path to the PDF file
            collection_name: Name of the collection in the vector database
            
        Returns:
            Dictionary with processing results and statistics
        """
        try:
            logger.info(f"Starting PDF processing: {pdf_path}")
            
            # Process PDF
            documents = self.pdf_processor.process_pdf(pdf_path)
            doc_info = self.pdf_processor.get_document_info(documents)
            
            # Create vector store
            vector_store = self.vector_store.create_vector_store(documents, collection_name)
            
            # Create QA chain
            self.llm_interface.create_qa_chain(vector_store)
            
            # Update state
            self.is_initialized = True
            self.current_pdf_path = pdf_path
            
            # Get collection info
            collection_info = self.vector_store.get_collection_info()
            
            result = {
                "success": True,
                "pdf_path": pdf_path,
                "document_info": doc_info,
                "collection_info": collection_info,
                "llm_info": self.llm_interface.get_llm_info()
            }
            
            logger.info("PDF processing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "pdf_path": pdf_path
            }
    
    def load_existing_database(self, collection_name: str = "pdf_documents") -> Dict:
        """
        Load an existing vector database.
        
        Args:
            collection_name: Name of the collection to load
            
        Returns:
            Dictionary with loading results
        """
        try:
            logger.info("Loading existing vector database")
            
            # Load vector store
            vector_store = self.vector_store.load_existing_vector_store(collection_name)
            
            # Create QA chain
            self.llm_interface.create_qa_chain(vector_store)
            
            # Update state
            self.is_initialized = True
            
            # Get collection info
            collection_info = self.vector_store.get_collection_info()
            
            result = {
                "success": True,
                "collection_info": collection_info,
                "llm_info": self.llm_interface.get_llm_info()
            }
            
            logger.info("Existing database loaded successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error loading existing database: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def ask_question(self, question: str) -> Dict:
        """
        Ask a question about the loaded PDF content.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing answer and source documents
        """
        if not self.is_initialized:
            return {
                "success": False,
                "error": "RAG system not initialized. Please load a PDF first."
            }
        
        try:
            result = self.llm_interface.ask_question(question)
            result["success"] = True
            return result
            
        except Exception as e:
            logger.error(f"Error asking question: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "answer": f"Error processing your question: {str(e)}"
            }
    
    def get_system_info(self) -> Dict:
        """
        Get comprehensive information about the RAG system.
        
        Returns:
            Dictionary with system information
        """
        info = {
            "is_initialized": self.is_initialized,
            "current_pdf": self.current_pdf_path,
            "llm_type": self.llm_type,
            "persist_directory": self.persist_directory
        }
        
        if self.is_initialized:
            info["collection_info"] = self.vector_store.get_collection_info()
            info["llm_info"] = self.llm_interface.get_llm_info()
        
        return info
    
    def clear_database(self, collection_name: str = "pdf_documents"):
        """
        Clear the vector database.
        
        Args:
            collection_name: Name of the collection to delete
        """
        try:
            self.vector_store.delete_collection(collection_name)
            self.is_initialized = False
            self.current_pdf_path = None
            self.llm_interface.qa_chain = None
            
            logger.info("Database cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing database: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 4) -> List[Dict]:
        """
        Perform similarity search on the vector database.
        
        Args:
            query: Search query
            k: Number of similar documents to retrieve
            
        Returns:
            List of similar documents with metadata
        """
        if not self.is_initialized:
            return []
        
        try:
            documents = self.vector_store.similarity_search(query, k=k)
            
            results = []
            for doc in documents:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            return [] 