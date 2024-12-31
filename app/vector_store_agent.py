from typing import List, Dict, Optional
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_core.tools import tool

vector_store = None
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

def initialize_store(persist_directory: str):
    """Initialize or load the vector store"""
    global vector_store
    if os.path.exists(persist_directory):
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings()
        )
    else:
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings()
        )

@tool
def add_texts(texts: List[str], metadatas: Optional[List[Dict]] = None):
    """
    Add texts to the vector store
    
    Args:
        texts: List of text content to add
        metadatas: Optional list of metadata dictionaries for each text
        
    Returns:
        None
    """
    initialize_store("./vector_store")
    documents = text_splitter.create_documents(texts, metadatas)
    vector_store.add_documents(documents)
    vector_store.persist()

@tool
def add_file(file_path: str, content: str):
    """
    Add a single file to the vector store
    
    Args:
        file_path: Path to the file being added
        content: Content of the file to add
        
    Returns:
        None
    """
    initialize_store("./vector_store")
    metadata = {"source": file_path}
    vector_store.add_texts([content], [metadata])  # Fix: call add_texts on vector_store

@tool
def similarity_search(query: str, k: int = 4) -> List[Document]:
    """
    Search for similar documents
    
    Args:
        query: Query string to search for
        k: Number of results to return (default: 4)
        
    Returns:
        List of similar documents
    """
    initialize_store("./vector_store")
    return vector_store.similarity_search(query, k=k)

@tool
def similarity_search_with_score(query: str, k: int = 4):
    """
    Search for similar documents with relevance scores
    
    Args:
        query: Query string to search for
        k: Number of results to return (default: 4)
        
    Returns:
        List of similar documents with relevance scores
    """
    initialize_store("./vector_store")
    return vector_store.similarity_search_with_score(query, k=k)
