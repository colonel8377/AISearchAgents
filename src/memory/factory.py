"""Vector store factory for creating different types of vector stores."""

from typing import Optional, Any
from langchain.embeddings.base import Embeddings
from langchain_openai import OpenAIEmbeddings

from ..utils.logger import get_logger

logger = get_logger(__name__)


class VectorStoreFactory:
    """Factory for creating vector store instances based on configuration."""
    
    @staticmethod
    def create_vector_store(
        store_type: str,
        embeddings: Optional[Embeddings] = None,
        **kwargs: Any
    ):
        """
        Create a vector store instance based on the specified type.
        
        Args:
            store_type: Type of vector store ('redis', 'postgres', or 'chroma')
            embeddings: Embeddings instance to use (defaults to OpenAIEmbeddings)
            **kwargs: Additional keyword arguments specific to the vector store type
            
        Returns:
            VectorStore instance
            
        Raises:
            ValueError: If store_type is not supported
        """
        logger.info(f"Creating vector store: type={store_type}")
        
        if embeddings is None:
            logger.debug("No embeddings provided, using default OpenAIEmbeddings")
            embeddings = OpenAIEmbeddings()
        
        try:
            if store_type == "redis":
                store = VectorStoreFactory._create_redis_store(embeddings, **kwargs)
            elif store_type == "postgres":
                store = VectorStoreFactory._create_postgres_store(embeddings, **kwargs)
            elif store_type == "chroma":
                store = VectorStoreFactory._create_chroma_store(embeddings, **kwargs)
            else:
                logger.error(f"Unsupported vector store type requested: {store_type}")
                raise ValueError(f"Unsupported vector store type: {store_type}")
            
            logger.info(f"Vector store created successfully: type={store_type}")
            return store
        except Exception as e:
            logger.error(f"Failed to create vector store: type={store_type}, error={e}", exc_info=True)
            raise
    
    @staticmethod
    def _create_redis_store(embeddings: Embeddings, **kwargs: Any):
        """Create a Redis vector store."""
        from langchain_community.vectorstores import Redis
        
        redis_url = kwargs.get("redis_url", "redis://localhost:6379")
        index_name = kwargs.get("index_name", "agent_memory")
        
        logger.debug(f"Creating Redis store: url={redis_url}, index={index_name}")
        
        return Redis(
            redis_url=redis_url,
            index_name=index_name,
            embedding=embeddings
        )
    
    @staticmethod
    def _create_postgres_store(embeddings: Embeddings, **kwargs: Any):
        """Create a PostgreSQL with pgvector store."""
        from langchain_community.vectorstores.pgvector import PGVector
        
        connection_string = kwargs.get(
            "connection_string",
            "postgresql://postgres:@localhost:5432/vectordb"
        )
        collection_name = kwargs.get("collection_name", "agent_memory")
        
        logger.debug(f"Creating Postgres store: collection={collection_name}")
        
        return PGVector(
            connection_string=connection_string,
            collection_name=collection_name,
            embedding_function=embeddings
        )
    
    @staticmethod
    def _create_chroma_store(embeddings: Embeddings, **kwargs: Any):
        """Create a Chroma vector store."""
        from langchain_community.vectorstores import Chroma
        
        persist_directory = kwargs.get("persist_directory", "./chroma_db")
        collection_name = kwargs.get("collection_name", "agent_memory")
        
        logger.debug(f"Creating Chroma store: persist_dir={persist_directory}, collection={collection_name}")
        
        return Chroma(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=embeddings
        )
