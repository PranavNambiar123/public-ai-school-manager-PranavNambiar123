from langchain_core.tools import tool
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

class QueryRouterAgent:
    def __init__(self, persist_directory: str):
        """Initialize the Query Router Agent with a vector store"""
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        try:
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
        except Exception as e:
            print(f"Error loading vector store: {e}")
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
    
    @tool
    def answer_query(self, query: str) -> str:
        """
        Answer a query using RAG (Retrieval Augmented Generation)
        
        Args:
            query: The question to answer
            
        Returns:
            Generated answer based on retrieved context
        """
        try:
            # Initialize the language model
            llm = OpenAI(temperature=0.2)
            
            # Create a retrieval QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
                return_source_documents=True
            )
            
            # Get the answer
            result = qa_chain({"query": query})
            
            # Format the response
            answer = result["result"]
            sources = [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
            
            return f"Answer: {answer}\n\nSources: {', '.join(sources)}"
            
        except Exception as e:
            return f"Error processing query: {str(e)}"
