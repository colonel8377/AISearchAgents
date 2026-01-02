"""
Memory detection module for analyzing user messages.

This module is responsible for detecting whether a user message contains
information worth storing in memory.
"""

import re
import json
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ...shared.utils.logger import get_logger
from ...shared.config.settings import settings
from ...shared.cache.decorator import cached

logger = get_logger(__name__)


class SmartMemoryDetector:
    """
    Detects whether user messages contain information worth storing.
    
    Uses both heuristic keyword matching and LLM-based intelligent analysis
    to determine if a message should be stored in memory.
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
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        Initialize memory detector.
        
        Args:
            llm: Optional LLM for intelligent memory detection
        """
        self.llm = llm
        logger.debug("MemoryDetector initialized")
    
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
    
    @cached()
    async def analyze_with_llm(self, user_message: str) -> Optional[Dict[str, Any]]:
        """
        Use LLM to intelligently analyze if message is worth storing.
        
        Args:
            user_message: The user's message
            
        Returns:
            Analysis result as dictionary, or None if analysis fails
        """
        if not self.llm:
            return None
            
        try:
            messages = [
                SystemMessage(content=self.DETECTION_PROMPT),
                HumanMessage(content=user_message)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse JSON response
            analysis = json.loads(response.content)
            
            logger.debug(
                f"LLM memory analysis: should_store={analysis.get('should_store')}, "
                f"type={analysis.get('memory_type')}"
            )
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze message with LLM: {e}")
            return None
    
    def analyze_heuristic(self, user_message: str) -> Dict[str, Any]:
        """
        Analyze message using heuristic rules.
        
        Args:
            user_message: The user's message
            
        Returns:
            Analysis result as dictionary
        """
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
        
        return {
            "should_store": True,
            "memory_type": memory_type,
            "summary": user_message[:200],  # Truncate for summary
            "key_facts": []
        }

