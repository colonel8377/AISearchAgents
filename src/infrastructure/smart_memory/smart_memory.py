"""
Smart memory system for detecting and storing important user information.

This module provides a high-level interface for intelligent memory management,
coordinating between detection and storage components.
"""

from typing import Dict, Any, Optional, List

from langchain_openai import ChatOpenAI

from .smart_memory_detector import SmartMemoryDetector
from .smart_memory_storage import SmartMemoryStorage
from ...shared.utils.logger import get_logger

logger = get_logger(__name__)


class SmartMemory:
    """
    Smart memory system that detects and stores important user information.
    
    Focuses on:
    - News preferences and interests
    - Political stances and opinions
    - Personal preferences and context
    - Important facts about the user
    
    This class coordinates between MemoryDetector (for analysis) and
    MemoryRepository (for persistence).
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None, vector_store: Optional[Any] = None):
        """
        Initialize smart memory.
        
        Args:
            llm: Optional LLM for intelligent memory detection
            vector_store: Optional vector store for persisting memories
        """
        self.detector = SmartMemoryDetector(llm=llm)
        self.storage = SmartMemoryStorage(vector_store=vector_store)
        logger.debug("SmartMemory initialized")
    
    def should_store_message(self, user_message: str) -> bool:
        """
        Quick heuristic check if message might be worth storing.
        
        Args:
            user_message: The user's message
            
        Returns:
            True if message potentially contains memorable information
        """
        return self.detector.should_store_message(user_message)
    
    async def analyze_and_store(
        self,
        user_message: str,
        bot_response: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze message and store if it contains memorable information.
        
        Args:
            user_message: The user's message
            bot_response: The bot's response (optional)
            context: Additional context (bot_name, agent_id, etc.)
            
        Returns:
            Memory entry if stored, None otherwise
        """
        if not self.should_store_message(user_message):
            logger.debug("Message does not meet heuristic criteria for storage")
            return None
        
        # Use LLM for intelligent detection if available
        if self.detector.llm:
            try:
                memory_analysis = await self.detector.analyze_with_llm(user_message)
                if memory_analysis and memory_analysis.get("should_store"):
                    return self.storage.store_memory(
                        user_message,
                        memory_analysis,
                        bot_response,
                        context
                    )
            except Exception as e:
                logger.warning(
                    f"LLM analysis failed, falling back to heuristic storage: {e}"
                )
        
        # Fallback: Store with heuristic analysis
        heuristic_analysis = self.detector.analyze_heuristic(user_message)
        return self.storage.store_memory(
            user_message,
            heuristic_analysis,
            bot_response,
            context
        )
    
    def get_memories(
        self,
        memory_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve stored memories.
        
        Args:
            memory_type: Optional filter by memory type
            limit: Optional limit on number of memories
            
        Returns:
            List of memory entries
        """
        return self.storage.get_memories(memory_type, limit)
    
    def clear_memories(self):
        """Clear all cached memories."""
        self.storage.clear_memories()
