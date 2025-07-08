import os
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF processing including loading, text extraction, and chunking."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the PDF processor.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load and extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects
        """
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            logger.info(f"Loading PDF: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from PDF")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF: {str(e)}")
            raise
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for better retrieval.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        try:
            logger.info(f"Splitting {len(documents)} documents into chunks")
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            raise
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """
        Complete PDF processing pipeline: load and chunk.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of processed Document chunks
        """
        documents = self.load_pdf(pdf_path)
        chunks = self.split_documents(documents)
        return chunks
    
    def get_document_info(self, documents: List[Document]) -> dict:
        """
        Get information about the processed documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary with document statistics
        """
        total_chunks = len(documents)
        total_text_length = sum(len(doc.page_content) for doc in documents)
        avg_chunk_length = total_text_length / total_chunks if total_chunks > 0 else 0
        
        return {
            "total_chunks": total_chunks,
            "total_text_length": total_text_length,
            "average_chunk_length": avg_chunk_length,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        } 