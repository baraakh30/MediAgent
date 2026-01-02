"""
Configuration management for the Medical Agent + RAG Application
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Configuration
    FLASK_APP: str = "app.py"
    FLASK_ENV: str = "development"
    SECRET_KEY: str = "default-secret-key-change-me"
    API_PORT: int = 5000
    
    # Google Gemini API (for agent)
    GOOGLE_API_KEY: str = ""
    AGENT_MODEL: str = "gemini-2.5-flash"
    
    # Local LLM Configuration (for RAG - optional if using Gemini for RAG too)
    USE_LOCAL_LLM: bool = False
    LLM_MODEL_NAME: str = "Qwen/Qwen2.5-3B-Instruct"
    LLM_DEVICE: str = "cuda"
    LLM_MAX_LENGTH: int = 2048
    
    # Embedding Model Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Vector Store Configuration
    VECTOR_STORE_TYPE: str = "chroma"
    VECTOR_STORE_PATH: str = "./data/vector_store"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # RAG Configuration
    TOP_K_RESULTS: int = 5
    TEMPERATURE: float = 0.3
    MAX_TOKENS: int = 2000
    
    # Data Sources
    DATA_DIR: str = "./data/raw"
    PROCESSED_DATA_DIR: str = "./data/processed"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"
    
    # Langfuse Observability (optional)
    LANGFUSE_ENABLED: bool = False
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    
    # Redis Configuration (optional)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    HISTORY_DB: int = 1
    REDIS_PASSWORD: Optional[str] = None
    REDIS_ENABLED: bool = False
    CACHE_TTL: int = 3600
    MAX_HISTORY_PER_SESSION: int = 50
    
    # Agent Configuration
    AGENT_MAX_ITERATIONS: int = 5
    AGENT_VERBOSE: bool = True
    
    # CrewAI Configuration
    CREWAI_TRACING_ENABLED: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra environment variables


def get_settings() -> Settings:
    """Get application settings"""
    return Settings()


def create_directories():
    """Create necessary directories if they don't exist"""
    dirs = [
        "data/raw",
        "data/processed",
        "data/vector_store",
        "logs"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
