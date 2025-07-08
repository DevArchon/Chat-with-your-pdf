import streamlit as st
import os
from dotenv import load_dotenv
from rag_system import RAGSystem
import tempfile
import logging
from typing import Optional, List, Any

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Chat with Your PDF",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        color: black;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .user-message {
        color: black;
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        color: black;
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .source-box {
        color: black;
        background-color: #f5f5f5;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .info-box {
        color: black;
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
    }
    .error-box {
        color: black;
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'pdf_uploaded' not in st.session_state:
        st.session_state.pdf_uploaded = False
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False

def create_rag_system(llm_type: str, api_key: Optional[str] = None):
    """Create and initialize the RAG system."""
    try:
        rag_system = RAGSystem(
            llm_type=llm_type,
            api_key=api_key,
            persist_directory="./chroma_db"
        )
        return rag_system
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None

def display_chat_message(role: str, content: str, sources: Optional[List[Any]] = None):
    """Display a chat message with proper styling."""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>Assistant:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
        
        if sources and len(sources) > 0:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>Source {i}:</strong><br>
                        {source['content']}<br>
                        <em>Page: {source.get('metadata', {}).get('page', 'N/A')}</em>
                    </div>
                    """, unsafe_allow_html=True)

def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📚 Chat with Your PDF</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a PDF and ask questions about its content using AI</p>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # LLM Selection
        llm_type = st.selectbox(
            "Choose LLM Provider",
            ["gemini", "ollama"],
            help="Select the LLM provider to use for answering questions"
        )
        
        # API Key input for Gemini
        api_key = None
        if llm_type == "gemini":
            api_key = st.text_input(
                "Google API Key",
                type="password",
                help="Enter your Google Gemini API key",
                value=os.getenv("GOOGLE_API_KEY", "")
            )
            
            if not api_key:
                st.warning("⚠️ Please enter your Google API key to use Gemini")
        
        # Ollama configuration
        elif llm_type == "ollama":
            st.info("🔧 Make sure Ollama is running locally with the llama2 model")
            st.code("ollama run llama2", language="bash")
        
        # Initialize system button
        if st.button("🚀 Initialize System", type="primary"):
            with st.spinner("Initializing RAG system..."):
                rag_system = create_rag_system(llm_type, api_key)
                if rag_system:
                    st.session_state.rag_system = rag_system
                    st.session_state.system_initialized = True
                    st.success("✅ System initialized successfully!")
                else:
                    st.error("❌ Failed to initialize system")
        
        # System info
        if st.session_state.system_initialized and st.session_state.rag_system:
            st.header("📊 System Info")
            system_info = st.session_state.rag_system.get_system_info()
            
            st.metric("Status", "✅ Active" if system_info["is_initialized"] else "❌ Inactive")
            st.metric("LLM Type", system_info["llm_type"].title())
            
            if system_info.get("collection_info"):
                st.metric("Documents", system_info["collection_info"].get("total_documents", 0))
        
        # Clear database button
        if st.session_state.system_initialized and st.session_state.rag_system:
            if st.button("🗑️ Clear Database", type="secondary"):
                with st.spinner("Clearing database..."):
                    try:
                        st.session_state.rag_system.clear_database()
                        st.session_state.chat_history = []
                        st.session_state.pdf_uploaded = False
                        st.success("✅ Database cleared successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error clearing database: {str(e)}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Upload PDF")
        
        if not st.session_state.system_initialized:
            st.warning("⚠️ Please initialize the system first using the sidebar")
        else:
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type="pdf",
                help="Upload a PDF file to chat with"
            )
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    pdf_path = tmp_file.name
                
                # Process PDF
                if st.button("🔍 Process PDF", type="primary"):
                    with st.spinner("Processing PDF..."):
                        try:
                            result = st.session_state.rag_system.load_pdf(pdf_path)
                            
                            if result["success"]:
                                st.session_state.pdf_uploaded = True
                                st.session_state.chat_history = []
                                
                                # Display processing results
                                st.success("✅ PDF processed successfully!")
                                
                                with st.expander("📊 Processing Details"):
                                    st.json(result)
                            else:
                                st.error(f"❌ Error processing PDF: {result.get('error', 'Unknown error')}")
                                
                        except Exception as e:
                            st.error(f"❌ Error processing PDF: {str(e)}")
                        finally:
                            # Clean up temporary file
                            try:
                                os.unlink(pdf_path)
                            except:
                                pass
    
    with col2:
        st.header("💬 Chat Interface")
        
        if not st.session_state.system_initialized:
            st.info("ℹ️ Initialize the system first to start chatting")
        elif not st.session_state.pdf_uploaded:
            st.info("ℹ️ Upload and process a PDF to start chatting")
        else:
            # Chat interface
            user_question = st.text_input(
                "Ask a question about your PDF:",
                placeholder="What is the main topic of this document?",
                key="user_question"
            )
            
            if st.button("🤖 Ask Question", type="primary") and user_question:
                # Add user question to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_question,
                    "sources": []
                })
                
                # Get answer from RAG system
                with st.spinner("🤔 Thinking..."):
                    try:
                        result = st.session_state.rag_system.ask_question(user_question)
                        
                        if result["success"]:
                            # Add assistant response to chat history
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": result["answer"],
                                "sources": result.get("sources", [])
                            })
                        else:
                            st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
                            
                    except Exception as e:
                        st.error(f"❌ Error processing question: {str(e)}")
                
                # Clear the input
                st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.header("💭 Chat History")
        
        for message in st.session_state.chat_history:
            display_chat_message(
                message["role"],
                message["content"],
                message.get("sources", [])
            )

if __name__ == "__main__":
    main() 