"""Configuration management for AI Search Agents Platform."""

from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings using Pydantic for validation and environment variable loading."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Settings
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    
    # LLM Settings (OpenAI-compatible API for Qwen)
    openai_api_key: str = Field(default="", description="OpenAI API key or Qwen API key")
    openai_api_base: str = Field(default="https://api.openai.com/v1", description="OpenAI API base URL")
    openai_model: str = Field(default="gpt-3.5-turbo", description="Model name (e.g., qwen-turbo)")
    
    # Vector Store Settings
    vector_store_type: Literal["redis", "postgres", "chroma"] = Field(
        default="chroma",
        description="Type of vector store to use"
    )
    
    # Redis Settings
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_password: str = Field(default="", description="Redis password")
    redis_db: int = Field(default=0, description="Redis database number")
    
    # PostgreSQL Settings
    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_user: str = Field(default="postgres", description="PostgreSQL user")
    postgres_password: str = Field(default="", description="PostgreSQL password")
    postgres_db: str = Field(default="vectordb", description="PostgreSQL database name")
    
    # Chroma Settings
    chroma_persist_directory: str = Field(default="./chroma_db", description="Chroma persistence directory")
    
    # Agent Settings
    agent_max_turns: int = Field(default=4, description="Maximum number of turns (0-3)")
    agent_temperature: float = Field(default=0.7, description="LLM temperature for agent responses")
    

# Global settings instance
settings = Settings()
