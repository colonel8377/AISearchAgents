"""Summarizer Service - Handles summarizer agent business logic."""

from typing import Dict, Any, List, Optional

from .base_service import BaseService
from src.shared.constant.enums import AgentType
from ...shared.utils.logger import get_logger

logger = get_logger(__name__)


class SummarizerService(BaseService):
    """
    Service for summarizer agent operations.
    
    Single Responsibility: Handle summarizer-specific business logic.
    """
    
    def summarize_conversation(
        self,
        agent_id: str,
        conversation_records: List[Dict[str, str]],
        execution_mode: str,
        use_few_shots: bool = True
    ) -> Dict[str, Any]:
        """
        Summarize conversation records.
        
        Args:
            agent_id: The agent ID
            conversation_records: List of conversation records
            execution_mode: Execution mode (chain_local, chain_api, etc.)
            use_few_shots: Whether to use few-shot examples
            
        Returns:
            Dictionary with summary and metadata
            
        Raises:
            ValueError: If agent does not exist or is wrong type
        """
        self._validate_agent_type(agent_id, AgentType.SUMMARIZER)
        
        agent = self._get_agent(agent_id)
        return agent.summarize_conversation(
            conversation_records=conversation_records,
            execution_mode=execution_mode,
            use_few_shots=use_few_shots
        )
    
    def get_summary_history(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Get summary history for an agent.
        
        Args:
            agent_id: The agent ID
            
        Returns:
            List of summary metadata
            
        Raises:
            ValueError: If agent does not exist or is wrong type
        """
        self._validate_agent_type(agent_id, AgentType.SUMMARIZER)
        
        agent = self._get_agent(agent_id)
        history = agent.get_summary_history()
        
        # Return only metadata, not full records
        return [
            {
                "conversation_length": entry["conversation_length"],
                "summary": entry["summary"]
            }
            for entry in history
        ]
    
    def create_conversation(
        self,
        agent_id: str,
        conversation_id: Optional[str] = None
    ) -> str:
        """
        Create a new conversation.
        
        Args:
            agent_id: The agent ID
            conversation_id: Optional conversation ID
            
        Returns:
            Conversation ID
            
        Raises:
            ValueError: If agent does not exist or is wrong type
        """
        self._validate_agent_type(agent_id, AgentType.SUMMARIZER)
        
        agent = self._get_agent(agent_id)
        return agent.create_conversation(conversation_id=conversation_id)
    
    def add_turn(
        self,
        agent_id: str,
        conversation_id: str,
        user_message: str,
        assistant_message: str
    ) -> Dict[str, Any]:
        """
        Add a turn to a conversation.
        
        Args:
            agent_id: The agent ID
            conversation_id: The conversation ID
            user_message: User message
            assistant_message: Assistant message
            
        Returns:
            Dictionary with result
            
        Raises:
            ValueError: If agent does not exist or is wrong type
        """
        self._validate_agent_type(agent_id, AgentType.SUMMARIZER)
        
        agent = self._get_agent(agent_id)
        return agent.add_turn(conversation_id, user_message, assistant_message)
    
    def list_conversations(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        List all conversations for an agent.
        
        Args:
            agent_id: The agent ID
            
        Returns:
            List of conversations
            
        Raises:
            ValueError: If agent does not exist or is wrong type
        """
        self._validate_agent_type(agent_id, AgentType.SUMMARIZER)
        
        agent = self._get_agent(agent_id)
        return agent.list_conversations()
    
    def get_conversation(
        self,
        agent_id: str,
        conversation_id: str,
        turn: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get a conversation by ID.
        
        Args:
            agent_id: The agent ID
            conversation_id: The conversation ID
            turn: Optional turn number
            
        Returns:
            Conversation data
            
        Raises:
            ValueError: If agent does not exist or is wrong type
        """
        self._validate_agent_type(agent_id, AgentType.SUMMARIZER)
        
        agent = self._get_agent(agent_id)
        conversation = agent.get_conversation(conversation_id, turn=turn)
        
        if conversation is None:
            if turn is not None:
                raise ValueError(f"Conversation '{conversation_id}' or turn {turn} not found")
            else:
                raise ValueError(f"Conversation '{conversation_id}' not found")
        
        if turn is not None:
            return {
                "conversation_id": conversation["conversation_id"],
                "turn": turn,
                "turn_data": conversation["turn_data"],
                "turn_count": conversation["turn_count"]
            }
        else:
            return {
                "conversation_id": conversation["conversation_id"],
                "turn_count": len(conversation["turns"]),
                "turns": conversation["turns"],
                "summary": conversation["summary"],
                "has_summary": conversation["summary"] is not None
            }
    
    def delete_conversation(self, agent_id: str, conversation_id: str) -> bool:
        """
        Delete a conversation.
        
        Args:
            agent_id: The agent ID
            conversation_id: The conversation ID
            
        Returns:
            True if deleted, False otherwise
            
        Raises:
            ValueError: If agent does not exist or is wrong type
        """
        self._validate_agent_type(agent_id, AgentType.SUMMARIZER)
        
        agent = self._get_agent(agent_id)
        return agent.delete_conversation(conversation_id)
    
    def summarize_conversation_by_id(
        self,
        agent_id: str,
        conversation_id: str,
        use_few_shots: bool = True,
        custom_few_shots: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Summarize a conversation by ID.
        
        Args:
            agent_id: The agent ID
            conversation_id: The conversation ID
            use_few_shots: Whether to use few-shot examples
            custom_few_shots: Optional custom few-shot examples
            
        Returns:
            Dictionary with summary and metadata
            
        Raises:
            ValueError: If agent does not exist or is wrong type
        """
        self._validate_agent_type(agent_id, AgentType.SUMMARIZER)
        
        agent = self._get_agent(agent_id)
        result = agent.summarize_conversation_by_id(
            conversation_id=conversation_id,
            execution_mode="chain_local",
            use_few_shots=use_few_shots,
            custom_few_shots=custom_few_shots
        )
        
        # Add conversation_id to metadata
        if "metadata" not in result:
            result["metadata"] = {}
        result["metadata"]["conversation_id"] = conversation_id
        
        return result

