"""
Medical RAG Tool for CrewAI Agent
This module provides a custom tool that allows CrewAI agents to interface with the Medical RAG system
"""
import os
from typing import Type, Optional
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from rag_pipeline import MedicalRAGPipeline, initialize_rag_pipeline
from data_loader import MedicalDataLoader, prepare_documents_for_rag
from config import get_settings, create_directories
from logger import setup_logging, get_logger

logger = get_logger(__name__)
config = get_settings()

# Global RAG pipeline instance
_rag_pipeline: Optional[MedicalRAGPipeline] = None


def get_rag_pipeline() -> MedicalRAGPipeline:
    """Get or initialize the RAG pipeline singleton"""
    global _rag_pipeline
    
    if _rag_pipeline is None:
        setup_logging()
        config = get_settings()
        create_directories()
        
        logger.info("Initializing RAG pipeline...")
        _rag_pipeline = initialize_rag_pipeline(config)
        
        # Try to load existing vector store
        try:
            _rag_pipeline.load_vector_store(store_type=config.VECTOR_STORE_TYPE)
            logger.info("Vector store loaded successfully")
        except FileNotFoundError:
            logger.warning("No existing vector store found. Building from scratch...")
            
            # Load and process data
            loader = MedicalDataLoader(config.DATA_DIR, config.PROCESSED_DATA_DIR)
            documents = loader.load_processed_data()
            prepared_docs = prepare_documents_for_rag(documents)
            
            # Create vector store
            chunked_docs = _rag_pipeline.chunk_documents(prepared_docs)
            _rag_pipeline.create_vector_store(chunked_docs, store_type=config.VECTOR_STORE_TYPE)
            
            logger.info("Vector store created successfully")
    
    return _rag_pipeline


class MedicalQueryInput(BaseModel):
    """Input schema for medical query tool"""
    question: str = Field(
        ..., 
        description="The medical question to search for in the knowledge base"
    )


class MedicalSearchInput(BaseModel):
    """Input schema for medical search tool"""
    query: str = Field(
        ..., 
        description="The search query to find relevant medical documents"
    )
    num_results: int = Field(
        default=5,
        description="Number of results to return (1-10)"
    )


class MedicalRAGQueryTool(BaseTool):
    """
    Tool for querying the Medical RAG system with natural language questions.
    This tool searches a comprehensive medical knowledge base containing:
    - MedQA (medical licensing exam questions)
    - MedDialog (doctor-patient conversations)
    - HealthSearchQA (health search queries)
    - LiveQA (consumer health questions)
    
    Use this tool when you need to answer medical questions with supporting evidence.
    """
    name: str = "medical_knowledge_query"
    description: str = """Use this tool to query the medical knowledge base and get comprehensive answers to medical questions.
    The tool will search through medical Q&A datasets, doctor-patient dialogues, and health information
    to provide accurate, evidence-based medical information with source documents.
    
    Input should be a clear medical question. For example:
    - "What are the symptoms of diabetes?"
    - "How is hypertension treated?"
    - "What causes chest pain?"
    """
    args_schema: Type[BaseModel] = MedicalQueryInput
    
    def _run(self, question: str) -> str:
        """Execute the medical query"""
        try:
            rag = get_rag_pipeline()
            result = rag.query(question)
            
            # Format the response
            answer = result.get("answer", "No answer found")
            sources = result.get("source_documents", [])
            
            response = f"**Answer:** {answer}\n\n"
            
            if sources:
                response += "**Sources:**\n"
                for i, source in enumerate(sources[:3], 1):
                    content_preview = source.get("content", "")[:200]
                    metadata = source.get("metadata", {})
                    dataset = metadata.get("dataset", "Unknown")
                    response += f"\n{i}. [{dataset}] {content_preview}...\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error in medical query: {e}")
            return f"Error querying medical knowledge base: {str(e)}"


class MedicalSearchTool(BaseTool):
    """
    Tool for searching the Medical knowledge base without generating an answer.
    Returns relevant document snippets based on semantic similarity.
    
    Use this tool when you need to find specific medical information or
    gather context before answering a complex question.
    """
    name: str = "medical_document_search"
    description: str = """Use this tool to search the medical knowledge base and retrieve relevant documents.
    This performs a semantic similarity search and returns matching document snippets.
    Unlike the query tool, this does not generate an answer - it only retrieves documents.
    
    Use this when you need to:
    - Find specific medical information
    - Gather context for complex questions
    - Verify medical facts
    
    Input should be a search query related to medical topics.
    """
    args_schema: Type[BaseModel] = MedicalSearchInput
    
    def _run(self, query: str, num_results: int = 5) -> str:
        """Execute the medical search"""
        try:
            num_results = min(max(1, num_results), 10)  # Clamp between 1-10
            
            rag = get_rag_pipeline()
            results = rag.similarity_search(query, k=num_results)
            
            if not results:
                return "No relevant documents found."
            
            response = f"**Found {len(results)} relevant documents:**\n\n"
            
            for i, doc in enumerate(results, 1):
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                dataset = metadata.get("dataset", "Unknown")
                
                response += f"**Document {i} [{dataset}]:**\n{content}\n\n"
                response += "-" * 50 + "\n\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error in medical search: {e}")
            return f"Error searching medical knowledge base: {str(e)}"


# Create tool instances for easy import
medical_query_tool = MedicalRAGQueryTool()
medical_search_tool = MedicalSearchTool()


def get_medical_tools():
    """Get all medical RAG tools for use with CrewAI agents"""
    return [medical_query_tool, medical_search_tool]


if __name__ == "__main__":
    # Test the tools
    setup_logging()
    
    print("Testing Medical RAG Tools...")
    print("=" * 60)
    
    # Test query tool
    print("\n1. Testing Medical Query Tool:")
    result = medical_query_tool._run("What are the symptoms of diabetes?")
    print(result)
    
    print("\n" + "=" * 60)
    
    # Test search tool
    print("\n2. Testing Medical Search Tool:")
    result = medical_search_tool._run("hypertension treatment", num_results=3)
    print(result)
