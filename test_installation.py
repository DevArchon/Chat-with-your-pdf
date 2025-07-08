#!/usr/bin/env python3
"""
Test script to verify the Chat with Your PDF installation.

This script checks if all dependencies are properly installed
and the system can be initialized without errors.
"""

import sys
import importlib
import os

def test_imports():
    """Test if all required packages can be imported."""
    print("🔍 Testing package imports...")
    
    required_packages = [
        'langchain',
        'langchain_community',
        'langchain_text_splitters',
        'chromadb',
        'pypdf',
        'streamlit',
        'google.generativeai',
        'sentence_transformers',
        'faiss',
        'numpy',
        'pandas',
        'dotenv'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError as e:
            print(f"  ❌ {package}: {str(e)}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing packages: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

def test_system_initialization():
    """Test if the RAG system can be initialized."""
    print("\n🚀 Testing system initialization...")
    
    try:
        # Test basic imports
        from rag_system import RAGSystem
        from pdf_processor import PDFProcessor
        from vector_store import VectorStore
        from llm_interface import LLMInterface
        
        print("  ✅ Core modules imported")
        
        # Test PDF processor
        pdf_processor = PDFProcessor()
        print("  ✅ PDF processor initialized")
        
        # Test vector store
        vector_store = VectorStore()
        print("  ✅ Vector store initialized")
        
        # Test LLM interface (without API key for now)
        try:
            llm_interface = LLMInterface(llm_type="ollama")
            print("  ✅ LLM interface initialized (Ollama)")
        except Exception as e:
            print(f"  ⚠️  LLM interface (Ollama): {str(e)}")
        
        # Test RAG system
        try:
            rag_system = RAGSystem(llm_type="ollama")
            print("  ✅ RAG system initialized (Ollama)")
        except Exception as e:
            print(f"  ⚠️  RAG system (Ollama): {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ System initialization failed: {str(e)}")
        return False

def test_environment():
    """Test environment configuration."""
    print("\n🔧 Testing environment configuration...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"  Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("  ❌ Python 3.8 or higher is required")
        return False
    else:
        print("  ✅ Python version is compatible")
    
    # Check environment variables
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        print("  ✅ Google API key found")
    else:
        print("  ℹ️  Google API key not set (required for Gemini)")
    
    # Check if .env file exists
    if os.path.exists(".env"):
        print("  ✅ .env file found")
    else:
        print("  ℹ️  .env file not found (create from env_example.txt)")
    
    return True

def test_file_structure():
    """Test if all required files exist."""
    print("\n📁 Testing file structure...")
    
    required_files = [
        "app.py",
        "rag_system.py",
        "pdf_processor.py",
        "vector_store.py",
        "llm_interface.py",
        "requirements.txt",
        "README.md",
        "env_example.txt"
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("\n✅ All required files found!")
        return True

def main():
    """Run all tests."""
    print("🧪 Chat with Your PDF - Installation Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("File Structure", test_file_structure),
        ("Environment", test_environment),
        ("System Initialization", test_system_initialization)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your installation is ready.")
        print("\nNext steps:")
        print("1. Set up your API keys in .env file")
        print("2. Run: streamlit run app.py")
        print("3. Or run: python example_usage.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Check Python version (3.8+ required)")
        print("3. Set up environment variables")

if __name__ == "__main__":
    main() 