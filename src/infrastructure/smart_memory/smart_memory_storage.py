"""
Memory storage module for persisting and retrieving user memories.

This module handles the storage and retrieval of memory entries in both
in-memory cache and vector stores.
"""

from typing import Dict, Any, Optional, List, Union

from ...shared.utils.logger import get_logger

logger = get_logger(__name__)

# Type alias for memory entries
MemoryEntry = Dict[str, Any]


class SmartMemoryStorage:
    """
    Storage manager for memory entries.

    Supports both in-memory caching and vector store persistence for
    intelligent memory management with semantic search capabilities.
    """
    
    def __init__(self, vector_store: Optional[Any] = None):
        """
        Initialize memory storage.
        
        Args:
            vector_store: Optional vector store for persisting memories.
                         Should support `add_texts` and `similarity_search`.
        """
        self.vector_store = vector_store
        self._memory_cache: List[MemoryEntry] = []
        logger.debug(f"SmartMemoryStorage initialized with vector_store={bool(vector_store)}")

    def store_memory(
        self,
        user_message: str,
        analysis: Dict[str, Any],
        bot_response: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> MemoryEntry:
        """
        Store a memory entry in cache and vector store.

        Args:
            user_message: The user's message
            analysis: Analysis result from detector
            bot_response: The bot's response
            context: Additional context
            
        Returns:
            The stored memory entry
        """
        memory_entry: MemoryEntry = {
            "user_message": user_message,
            "memory_type": analysis.get("memory_type", "unknown"),
            "summary": analysis.get("summary", user_message[:200]),
            "key_facts": analysis.get("key_facts", []),
            "bot_response": bot_response,
            "context": context or {},
            "timestamp": analysis.get("timestamp")
        }
        
        # Store in local cache
        self._memory_cache.append(memory_entry)
        logger.info(
            f"Stored memory: type={memory_entry['memory_type']}, "
            f"summary={memory_entry['summary'][:50]}..."
        )
        
        # Store in vector store if available
        if self.vector_store:
            self._persist_to_vector_store(memory_entry)

        return memory_entry

    def _persist_to_vector_store(self, entry: MemoryEntry) -> None:
        """Helper to persist entry to vector store."""
        try:
            # Format text for embedding
            doc_text = (
                f"User: {entry['user_message']}\n"
                f"Type: {entry['memory_type']}\n"
                f"Summary: {entry['summary']}"
            )

            # Prepare metadata
            metadata = {
                "source": "smart_memory",
                "type": "user_memory",
                "memory_type": entry['memory_type'],
                "summary": entry['summary']
            }

            if entry.get("context"):
                # Ensure context values are compatible with vector store metadata
                # (usually requires primitives)
                for k, v in entry["context"].items():
                    if isinstance(v, (str, int, float, bool)):
                        metadata[k] = v
                    else:
                        metadata[k] = str(v)

            self.vector_store.add_texts(
                texts=[doc_text],
                metadatas=[metadata]
            )
            logger.debug("Memory persisted to vector store")

        except Exception as e:
            logger.error(f"Failed to store memory in vector store: {e}", exc_info=True)

    def get_memories(
        self,
        memory_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[MemoryEntry]:
        """
        Retrieve stored memories from local cache.

        Args:
            memory_type: Optional filter by memory type
            limit: Optional limit on number of memories (returns most recent)

        Returns:
            List of memory entries
        """
        memories = self._memory_cache

        if memory_type:
            memories = [m for m in memories if m.get("memory_type") == memory_type]
        
        if limit:
            memories = memories[-limit:]
        
        return memories

    def search_memories(self, query: str, k: int = 5) -> List[MemoryEntry]:
        """
        Search for relevant memories using semantic search (if vector store available)
        or simple text matching (fallback).

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of relevant memory entries
        """
        if self.vector_store:
            try:
                docs = self.vector_store.similarity_search(query, k=k)
                return [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "source": "vector_store"
                    }
                    for doc in docs
                ]
            except Exception as e:
                logger.warning(f"Vector search failed, falling back to local cache: {e}")

        # Fallback: Simple keyword matching in local cache
        query_lower = query.lower()
        matches = []
        for mem in self._memory_cache:
            content = (mem.get("summary", "") + " " + mem.get("user_message", "")).lower()
            if query_lower in content:
                matches.append(mem)

        # Return most recent matches up to k
        return matches[-k:] if k < len(matches) else matches

    def clear_memories(self) -> None:
        """Clear all cached memories."""
        self._memory_cache.clear()
        logger.info("Memory cache cleared")
