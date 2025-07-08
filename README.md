# 📚 Chat with Your PDF

A powerful RAG (Retrieval-Augmented Generation) system that allows you to upload PDF documents and chat with them using AI. The system uses vector embeddings to understand document content and provides accurate answers based on the PDF's content.

## ✨ Features

- **PDF Processing**: Upload and process PDF documents with automatic text extraction and chunking
- **Vector Database**: Store document embeddings using ChromaDB for efficient retrieval
- **Multiple LLM Support**: Choose between Google Gemini API or local Ollama (Llama)
- **Beautiful UI**: Modern Streamlit interface with real-time chat
- **Source Attribution**: View the exact sources used to generate answers
- **Persistent Storage**: Vector database persists between sessions

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Upload    │───▶│  Text Chunking  │───▶│  Vector Store   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Chat UI       │◀───│   RAG System    │◀───│   LLM Interface │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Google API key (for Gemini) or Ollama (for local Llama)

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd chat-with-pdf
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   Copy `env_example.txt` to `.env` and configure:

   ```bash
   # For Google Gemini
   GOOGLE_API_KEY=your_gemini_api_key_here

   # For Ollama (optional)
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama2
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🔧 Configuration

### Using Google Gemini API

1. Get a Google API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Set the `GOOGLE_API_KEY` environment variable
3. Select "gemini" as the LLM provider in the app

### Using Ollama (Local Llama)

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull the Llama model:
   ```bash
   ollama pull llama2
   ```
3. Start Ollama:
   ```bash
   ollama serve
   ```
4. Select "ollama" as the LLM provider in the app

## 📖 Usage

1. **Initialize the System**

   - Open the app in your browser
   - Configure your LLM provider in the sidebar
   - Click "Initialize System"

2. **Upload a PDF**

   - Click "Choose a PDF file" in the upload section
   - Select your PDF document
   - Click "Process PDF" to extract and embed the content

3. **Start Chatting**
   - Type your questions in the chat interface
   - Get AI-powered answers based on your PDF content
   - View source documents used to generate answers

## 🏗️ Project Structure

```
chat-with-pdf/
├── app.py                 # Main Streamlit application
├── rag_system.py          # Core RAG system orchestrator
├── pdf_processor.py       # PDF loading and text processing
├── vector_store.py        # Vector database operations
├── llm_interface.py       # LLM provider interface
├── requirements.txt       # Python dependencies
├── env_example.txt       # Environment variables template
├── README.md             # This file
└── chroma_db/           # Vector database storage (auto-created)
```

## 🔍 Key Components

### PDFProcessor

- Handles PDF loading and text extraction
- Splits documents into manageable chunks
- Provides document statistics and metadata

### VectorStore

- Manages ChromaDB vector database
- Handles document embedding and storage
- Performs similarity search for relevant content

### LLMInterface

- Supports multiple LLM providers (Gemini, Ollama)
- Creates QA chains for question answering
- Manages API keys and model configuration

### RAGSystem

- Orchestrates all components
- Provides high-level interface for PDF processing and Q&A
- Manages system state and persistence

## 🎯 Features in Detail

### Smart Text Chunking

- Automatic document splitting with configurable chunk sizes
- Overlapping chunks to maintain context
- Metadata preservation for source tracking

### Vector Search

- Semantic similarity search using sentence transformers
- Configurable retrieval parameters
- Fast and accurate document retrieval

### Source Attribution

- View exact source documents for each answer
- Page-level source tracking
- Confidence scoring based on source relevance

### Persistent Storage

- Vector database persists between sessions
- No need to reprocess PDFs on restart
- Efficient storage and retrieval

## 🔧 Advanced Configuration

### Customizing Chunk Sizes

```python
# In rag_system.py
rag_system = RAGSystem(
    chunk_size=1000,      # Size of text chunks
    chunk_overlap=200,    # Overlap between chunks
    embedding_model="all-MiniLM-L6-v2"  # Embedding model
)
```

### Using Different Embedding Models

```python
# Available models: all-MiniLM-L6-v2, all-mpnet-base-v2, etc.
vector_store = VectorStore(embedding_model="all-mpnet-base-v2")
```

### Customizing LLM Parameters

```python
# For Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.1,
    max_output_tokens=2048
)

# For Ollama
llm = Ollama(
    model="llama2",
    temperature=0.1
)
```

## 🐛 Troubleshooting

### Common Issues

1. **"Google API key not found"**

   - Ensure you've set the `GOOGLE_API_KEY` environment variable
   - Check that the API key is valid and has proper permissions

2. **"Ollama connection failed"**

   - Make sure Ollama is running: `ollama serve`
   - Verify the model is installed: `ollama list`
   - Check the base URL in environment variables

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the RAG framework
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Streamlit](https://streamlit.io/) for the web interface
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [Google Gemini](https://ai.google.dev/) and [Ollama](https://ollama.ai/) for LLM providers
