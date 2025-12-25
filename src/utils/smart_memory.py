"""Smart memory utility for detecting and storing important user information."""

import re
import json
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .logger import get_logger
from ..config.settings import settings
from .agent_cache import cached

logger = get_logger(__name__)


class SmartMemory:
    """
    Smart memory system that detects and stores important user information.
    
    Focuses on:
    - News preferences and interests
    - Political stances and opinions
    - Personal preferences and context
    - Important facts about the user
    """
    
    # Keywords that indicate important memory-worthy content
    MEMORY_KEYWORDS = [
        # News and topics
        "news", "article", "report", "story", "breaking",
        # Opinions and stances
        "opinion", "believe", "think", "feel", "stance", "view", "perspective",
        "support", "oppose", "agree", "disagree",
        # Political
        "political", "politics", "policy", "government", "election",
        "democrat", "republican", "liberal", "conservative",
        # Preferences
        "prefer", "like", "dislike", "favorite", "interest", "enjoy",
        # Personal context
        "I am", "I'm", "my background", "my experience", "I work",
    ]
    
    DETECTION_PROMPT = """You are a memory detection assistant. Analyze the user's message to determine if it contains information worth remembering for future conversations.

Information worth remembering includes:
1. News topics or events the user is interested in
2. Political stances, opinions, or viewpoints
3. Personal preferences or interests
4. Important context about the user's background or situation
5. Strong beliefs or values expressed by the user

Respond with a JSON object containing:
- "should_store": true/false indicating if the message contains memorable information
- "memory_type": one of ["news_interest", "political_stance", "preference", "personal_context", "opinion", "none"]
- "summary": a brief summary of what to remember (only if should_store is true)
- "key_facts": list of 1-3 key facts to store (only if should_store is true)

Examples:
User: "I'm really interested in climate change news"
{"should_store": true, "memory_type": "news_interest", "summary": "User is interested in climate change news", "key_facts": ["interested in climate change news"]}

User: "I believe healthcare should be a universal right"
{"should_store": true, "memory_type": "political_stance", "summary": "User supports universal healthcare", "key_facts": ["believes healthcare should be universal right"]}

User: "What's the weather today?"
{"should_store": false, "memory_type": "none", "summary": "", "key_facts": []}

Analyze this message:
"""
    
    def __init__(self, llm: Optional[ChatOpenAI] = None, vector_store: Optional[Any] = None):
        """
        Initialize smart memory.
        
        Args:
            llm: Optional LLM for intelligent memory detection
            vector_store: Optional vector store for persisting memories
        """
        self.llm = llm
        self.vector_store = vector_store
        self.memory_cache: List[Dict[str, Any]] = []
        logger.debug("SmartMemory initialized")
    
    def should_store_message(self, user_message: str) -> bool:
        """
        Quick heuristic check if message might be worth storing.
        
        Args:
            user_message: The user's message
            
        Returns:
            True if message potentially contains memorable information
        """
        if not settings.smart_memory_enabled:
            return False
        
        # Quick keyword check
        message_lower = user_message.lower()
        
        # Check for first-person statements (I, my, I'm, etc.)
        first_person_pattern = r'\b(i|my|i\'m|i am|me)\b'
        has_first_person = bool(re.search(first_person_pattern, message_lower))
        
        # Check for memory keywords
        has_keywords = any(keyword in message_lower for keyword in self.MEMORY_KEYWORDS)
        
        # Check message length (too short messages are unlikely to be memorable)
        is_substantial = len(user_message.split()) >= 5
        
        return (has_first_person or has_keywords) and is_substantial
    
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
            context: Additional context (bot_id, agent_id, etc.)
            
        Returns:
            Memory entry if stored, None otherwise
        """
        if not self.should_store_message(user_message):
            logger.debug("Message does not meet heuristic criteria for storage")
            return None
        
        # Use LLM for intelligent detection if available
        if self.llm:
            try:
                memory_analysis = await self._llm_analyze_message(user_message)
                if memory_analysis and memory_analysis.get("should_store"):
                    return self._store_memory(
                        user_message,
                        memory_analysis,
                        bot_response,
                        context
                    )
            except Exception as e:
                logger.warning(f"LLM analysis failed, falling back to heuristic storage: {e}")
        
        # Fallback: Store with heuristic analysis
        return self._store_memory_heuristic(user_message, bot_response, context)
    
    @cached(exclude_class_name=True)
    async def _llm_analyze_message(self, user_message: str) -> Optional[Dict[str, Any]]:
        """
        Use LLM to intelligently analyze if message is worth storing.
        
        Args:
            user_message: The user's message
            
        Returns:
            Analysis result as dictionary
        """
        try:
            messages = [
                SystemMessage(content=self.DETECTION_PROMPT),
                HumanMessage(content=user_message)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse JSON response
            analysis = json.loads(response.content)
            
            logger.debug(f"LLM memory analysis: should_store={analysis.get('should_store')}, type={analysis.get('memory_type')}")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze message with LLM: {e}")
            return None
    
    def _store_memory_heuristic(
        self,
        user_message: str,
        bot_response: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store memory using heuristic analysis.
        
        Args:
            user_message: The user's message
            bot_response: The bot's response
            context: Additional context
            
        Returns:
            Memory entry
        """
        # Simple heuristic categorization
        message_lower = user_message.lower()
        
        if any(word in message_lower for word in ["news", "article", "breaking"]):
            memory_type = "news_interest"
        elif any(word in message_lower for word in ["political", "politics", "policy", "believe", "stance"]):
            memory_type = "political_stance"
        elif any(word in message_lower for word in ["prefer", "like", "favorite"]):
            memory_type = "preference"
        elif any(word in message_lower for word in ["i am", "i'm", "my background"]):
            memory_type = "personal_context"
        else:
            memory_type = "opinion"
        
        memory_entry = {
            "user_message": user_message,
            "memory_type": memory_type,
            "summary": user_message[:200],  # Truncate for summary
            "bot_response": bot_response,
            "context": context or {}
        }
        
        return self._store_memory(user_message, {"memory_type": memory_type}, bot_response, context)
    
    def _store_memory(
        self,
        user_message: str,
        analysis: Dict[str, Any],
        bot_response: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store memory entry.
        
        Args:
            user_message: The user's message
            analysis: Analysis result from LLM or heuristic
            bot_response: The bot's response
            context: Additional context
            
        Returns:
            Stored memory entry
        """
        memory_entry = {
            "user_message": user_message,
            "memory_type": analysis.get("memory_type", "unknown"),
            "summary": analysis.get("summary", user_message[:200]),
            "key_facts": analysis.get("key_facts", []),
            "bot_response": bot_response,
            "context": context or {}
        }
        
        # Store in cache
        self.memory_cache.append(memory_entry)
        logger.info(f"Stored memory: type={memory_entry['memory_type']}, summary={memory_entry['summary'][:50]}...")
        
        # Store in vector store if available
        if self.vector_store:
            try:
                doc_text = f"User: {user_message}\nType: {memory_entry['memory_type']}\nSummary: {memory_entry['summary']}"
                metadata = {
                    "type": "user_memory",
                    "memory_type": memory_entry['memory_type'],
                    "summary": memory_entry['summary']
                }
                if context:
                    metadata.update(context)
                
                self.vector_store.add_texts([doc_text], metadatas=[metadata])
                logger.debug("Memory stored in vector store")
            except Exception as e:
                logger.warning(f"Failed to store memory in vector store: {e}")
        
        return memory_entry
    
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
        memories = self.memory_cache
        
        if memory_type:
            memories = [m for m in memories if m.get("memory_type") == memory_type]
        
        if limit:
            memories = memories[-limit:]
        
        return memories
    
    def clear_memories(self):
        """Clear all cached memories."""
        self.memory_cache.clear()
        logger.info("Memory cache cleared")
