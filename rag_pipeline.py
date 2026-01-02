"""
RAG Pipeline implementation using LangChain with Gemini or local LLM
Handles document chunking, embedding, vector storage, and retrieval
"""
import torch
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class MedicalRAGPipeline:
    """RAG Pipeline for Medical Chatbot using Gemini or local LLM"""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_store_path: str = "./data/vector_store",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        device: str = "cpu",
        top_k: int = 5,
        google_api_key: str = None,
        use_gemini: bool = True,
        model_name: str = "gemini-2.5-flash"
    ):
        self.embedding_model_name = embedding_model
        self.vector_store_path = Path(vector_store_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = device if torch.cuda.is_available() else "cpu"
        self.top_k = top_k
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.use_gemini = use_gemini
        self.model_name = model_name
        
        logger.info(f"Initializing RAG Pipeline with device: {self.device}")
        
        # Initialize components
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        
        self._initialize_embeddings()
        if self.use_gemini and self.google_api_key:
            self._initialize_gemini_llm()
    
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        
        model_kwargs = {'device': self.device}
        encode_kwargs = {'normalize_embeddings': True}
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        logger.info("Embedding model loaded successfully")
    
    def _initialize_gemini_llm(self):
        """Initialize Google Gemini LLM"""
        logger.info(f"Loading Gemini LLM: {self.model_name}")
        
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.google_api_key,
                temperature=0.3,
                convert_system_message_to_human=True
            )
            logger.info("Gemini LLM loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Gemini LLM: {e}")
            raise
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Document]:
        """
        Chunk documents into smaller pieces for embedding
        
        Args:
            documents: List of document dictionaries with 'text' and 'metadata'
            
        Returns:
            List of LangChain Document objects
        """
        logger.info(f"Chunking {len(documents)} documents...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunked_docs = []
        
        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            metadata["source"] = doc.get("source", "unknown")
            
            chunks = text_splitter.split_text(text)
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_id"] = i
                
                chunked_docs.append(
                    Document(page_content=chunk, metadata=chunk_metadata)
                )
        
        logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        return chunked_docs
    
    def create_vector_store(self, documents: List[Document], store_type: str = "chroma"):
        """
        Create vector store from documents
        
        Args:
            documents: List of LangChain Document objects
            store_type: Type of vector store ('faiss' or 'chroma')
        """
        logger.info(f"Creating {store_type} vector store with {len(documents)} documents...")
        
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        try:
            if store_type.lower() == "faiss":
                self.vectorstore = FAISS.from_documents(
                    documents,
                    self.embeddings
                )
                self.vectorstore.save_local(str(self.vector_store_path / "faiss_index"))
                
            elif store_type.lower() == "chroma":
                self.vectorstore = Chroma.from_documents(
                    documents,
                    self.embeddings,
                    persist_directory=str(self.vector_store_path / "chroma_db")
                )
            
            else:
                raise ValueError(f"Unsupported vector store type: {store_type}")
            
            logger.info(f"Vector store created and saved to {self.vector_store_path}")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def load_vector_store(self, store_type: str = "chroma"):
        """
        Load existing vector store
        
        Args:
            store_type: Type of vector store ('faiss' or 'chroma')
        """
        logger.info(f"Loading {store_type} vector store from {self.vector_store_path}...")
        
        try:
            if store_type.lower() == "faiss":
                index_path = self.vector_store_path / "faiss_index"
                if not index_path.exists():
                    raise FileNotFoundError(f"FAISS index not found at {index_path}")
                
                self.vectorstore = FAISS.load_local(
                    str(index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
            elif store_type.lower() == "chroma":
                chroma_path = self.vector_store_path / "chroma_db"
                if not chroma_path.exists():
                    raise FileNotFoundError(f"Chroma DB not found at {chroma_path}")
                
                self.vectorstore = Chroma(
                    persist_directory=str(chroma_path),
                    embedding_function=self.embeddings
                )
                
                try:
                    collection = self.vectorstore._collection
                    count = collection.count()
                    logger.info(f"Vector store loaded successfully with {count} documents")
                except Exception as e:
                    logger.warning(f"Could not get Chroma collection count: {e}")
                    logger.info("Vector store loaded successfully")
            
            else:
                raise ValueError(f"Unsupported vector store type: {store_type}")
            
            if store_type.lower() == "faiss" and hasattr(self.vectorstore, 'index') and hasattr(self.vectorstore.index, 'ntotal'):
                logger.info(f"Vector store loaded successfully with {self.vectorstore.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system using Gemini
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and source documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Load or create vector store first.")
        
        logger.info(f"Processing query: {question[:100]}...")
        
        try:
            # Retrieve relevant documents
            logger.info(f"Searching for top {self.top_k} similar documents...")
            retrieved_docs = self.vectorstore.similarity_search(question, k=self.top_k)
            logger.info(f"Retrieved {len(retrieved_docs)} documents from vector store")
            
            # Build context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # Create prompt with context
            prompt = f"""You are MediChat, a helpful medical assistant AI. Provide clear, accurate medical information based on the context given.
Keep answers concise and professional. If the question is not about medical field say "I can't help with this type of questions, I'm a medical chatbot".

Context information:
{context}

Question: {question}

Please provide a helpful, accurate response based on the context above."""
            
            # Generate response using Gemini
            if self.llm:
                response = self.llm.invoke(prompt)
                answer = response.content
            else:
                answer = "LLM not initialized. Please set up the Gemini API key."
            
            response_dict = {
                "question": question,
                "answer": answer,
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in retrieved_docs
                ]
            }
            
            logger.info(f"Query processed successfully with {len(response_dict['source_documents'])} sources")
            return response_dict
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform similarity search without LLM generation
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        logger.info(f"Performing similarity search for: {query[:100]}...")
        
        docs = self.vectorstore.similarity_search(query, k=k)
        
        results = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs
        ]
        
        return results


def initialize_rag_pipeline(config) -> MedicalRAGPipeline:
    """
    Initialize RAG pipeline with configuration
    
    Args:
        config: Application configuration object
        
    Returns:
        Initialized MedicalRAGPipeline instance
    """
    pipeline = MedicalRAGPipeline(
        embedding_model=config.EMBEDDING_MODEL,
        vector_store_path=config.VECTOR_STORE_PATH,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        device=config.LLM_DEVICE if hasattr(config, 'LLM_DEVICE') else "cpu",
        top_k=config.TOP_K_RESULTS,
        google_api_key=config.GOOGLE_API_KEY if hasattr(config, 'GOOGLE_API_KEY') else None,
        use_gemini=True,
        model_name=config.AGENT_MODEL if hasattr(config, 'AGENT_MODEL') else "gemini-2.5-flash"
    )
    
    return pipeline


if __name__ == "__main__":
    from logger import setup_logging
    from config import get_settings
    from data_loader import MedicalDataLoader, prepare_documents_for_rag
    
    setup_logging()
    config = get_settings()
    
    # Load data
    loader = MedicalDataLoader()
    documents = loader.load_processed_data()
    prepared_docs = prepare_documents_for_rag(documents[:100])  # Test with first 100
    
    # Initialize pipeline
    rag = initialize_rag_pipeline(config)
    
    # Create vector store
    chunked_docs = rag.chunk_documents(prepared_docs)
    rag.create_vector_store(chunked_docs)
    
    # Test query
    response = rag.query("What are the symptoms of diabetes?")
    print(response)
