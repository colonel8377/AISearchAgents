"""
Abstract base class for vector store providers.
"""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore



class BaseVectorStoreProvider(ABC):
    """Abstract base class for vector store creators."""

    @property
    @abstractmethod
    def store_type(self) -> str:
        """
        Get the identifier for this store type (e.g., 'redis', 'chroma').
        
        Returns:
            Store type identifier string
        """
        pass

    @abstractmethod
    def create_store(
        self,
        embeddings: Embeddings,
        **kwargs: Any
    ) -> VectorStore:
        """
        Create the specific VectorStore instance.
        
        Args:
            embeddings: Embedding function to use
            **kwargs: Configuration specific to the vector store
            
        Returns:
            Configured VectorStore instance
        """
        raise NotImplementedError
