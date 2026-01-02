"""Nudge Collapse Service - Handles nudge collapse agent business logic."""

from typing import Dict, Any, List, Optional

from .base_service import BaseService
from ...shared.constant.enums import AgentType
from ...shared.config.settings import settings
from ...shared.utils.logger import get_logger

logger = get_logger(__name__)


class NudgeCollapseService(BaseService):
    """
    Service for nudge collapse agent operations.
    
    Single Responsibility: Handle nudge collapse-specific business logic.
    """
    
    def generate_turn(
        self,
        agent_id: str,
        user_query: str,
        search_summary: Optional[str] = None,
        search_urls: Optional[List[str]] = None,
        history_mode: Optional[str] = None,
        use_few_shots: bool = True,
        custom_few_shots: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a response turn.
        
        Args:
            agent_id: The agent ID
            user_query: User query
            search_summary: Optional search summary
            search_urls: Optional search URLs
            history_mode: Optional history mode
            use_few_shots: Whether to use few-shot examples
            custom_few_shots: Optional custom few-shot examples
            
        Returns:
            Dictionary with turn response
            
        Raises:
            ValueError: If agent does not exist or is wrong type
        """
        self._validate_agent_type(agent_id, AgentType.NUDGE_COLLAPSE)
        
        agent = self._get_agent(agent_id)
        history_mode = history_mode if history_mode is not None else settings.default_history_mode
        
        return agent.generate_turn(
            user_query=user_query,
            search_summary=search_summary,
            search_urls=search_urls,
            history_mode=history_mode,
            use_few_shots=use_few_shots,
            custom_few_shots=custom_few_shots
        )
    
    def get_conversation_history(self, agent_id: str) -> Dict[str, Any]:
        """
        Get conversation history for an agent.
        
        Args:
            agent_id: The agent ID
            
        Returns:
            Dictionary with conversation history
            
        Raises:
            ValueError: If agent does not exist or is wrong type
        """
        self._validate_agent_type(agent_id, AgentType.NUDGE_COLLAPSE)
        
        agent = self._get_agent(agent_id)
        
        return {
            "current_turn": agent.get_current_turn(),
            "max_turns": agent.max_turns,
            "history": agent.get_conversation_history()
        }

