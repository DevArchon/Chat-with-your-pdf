import os
from typing import Optional, List
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMInterface:
    """Interface for different LLM providers (Ollama and Google Gemini)."""
    
    def __init__(self, llm_type: str = "gemini", api_key: Optional[str] = None):
        """
        Initialize the LLM interface.
        
        Args:
            llm_type: Type of LLM to use ("gemini" or "ollama")
            api_key: API key for Gemini (if using Gemini)
        """
        self.llm_type = llm_type.lower()
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.llm = None
        self.qa_chain = None
        
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM based on the specified type."""
        try:
            if self.llm_type == "gemini":
                if not self.api_key:
                    raise ValueError("Google API key is required for Gemini. Set GOOGLE_API_KEY environment variable.")
                
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-pro",
                    google_api_key=self.api_key,
                    temperature=0.1,
                    max_output_tokens=2048
                )
                logger.info("Initialized Google Gemini LLM")
                
            elif self.llm_type == "ollama":
                ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                ollama_model = os.getenv("OLLAMA_MODEL", "llama2")
                
                self.llm = Ollama(
                    base_url=ollama_base_url,
                    model=ollama_model,
                    temperature=0.1
                )
                logger.info(f"Initialized Ollama LLM with model: {ollama_model}")
                
            else:
                raise ValueError(f"Unsupported LLM type: {self.llm_type}. Use 'gemini' or 'ollama'.")
                
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise
    
    def create_qa_chain(self, vector_store, k: int = 4):
        """
        Create a QA chain with the vector store.
        
        Args:
            vector_store: Vector store instance
            k: Number of documents to retrieve
        """
        try:
            # Create a custom prompt template
            prompt_template = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            Context: {context}
            
            Question: {question}
            
            Answer:"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create the QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={"k": k}),
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            
            logger.info("QA chain created successfully")
            
        except Exception as e:
            logger.error(f"Error creating QA chain: {str(e)}")
            raise
    
    def ask_question(self, question: str) -> dict:
        """
        Ask a question and get an answer with source documents.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing answer and source documents
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Please create a QA chain first.")
        
        try:
            logger.info(f"Processing question: {question}")
            
            # Get the answer
            result = self.qa_chain({"query": question})
            
            answer = result.get("result", "")
            source_documents = result.get("source_documents", [])
            
            # Format source documents
            sources = []
            for doc in source_documents:
                sources.append({
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                })
            
            response = {
                "answer": answer,
                "sources": sources,
                "llm_type": self.llm_type
            }
            
            logger.info(f"Generated answer with {len(sources)} source documents")
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                "answer": f"Error processing your question: {str(e)}",
                "sources": [],
                "llm_type": self.llm_type
            }
    
    def get_llm_info(self) -> dict:
        """
        Get information about the current LLM configuration.
        
        Returns:
            Dictionary with LLM information
        """
        info = {
            "llm_type": self.llm_type,
            "is_initialized": self.llm is not None,
            "qa_chain_ready": self.qa_chain is not None
        }
        
        if self.llm_type == "gemini":
            info["model"] = "gemini-pro"
            info["api_key_configured"] = bool(self.api_key)
        elif self.llm_type == "ollama":
            info["model"] = os.getenv("OLLAMA_MODEL", "llama2")
            info["base_url"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        return info 