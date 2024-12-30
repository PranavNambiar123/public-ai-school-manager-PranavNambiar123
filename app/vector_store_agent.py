from typing import List, Dict, Optional
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

class VectorStoreAgent:
    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def initialize_store(self):
        """Initialize or load the vector store"""
        if os.path.exists(self.persist_directory):
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """Add texts to the vector store"""
        documents = self.text_splitter.create_documents(texts, metadatas)
        self.vector_store.add_documents(documents)
        self.vector_store.persist()
        
    def add_file(self, file_path: str, content: str):
        """Add a single file to the vector store"""
        metadata = {"source": file_path}
        self.add_texts([content], [metadata])
        
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents"""
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4):
        """Search for similar documents with relevance scores"""
        return self.vector_store.similarity_search_with_score(query, k=k)
