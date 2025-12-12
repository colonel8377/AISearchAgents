"""Summarizer Agent implementation for summarizing conversation records."""

from typing import Dict, List, Optional, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from ...config.settings import settings


class SummarizerAgent:
    """
    Implements a Summarizer Agent that accepts user conversation records
    and summarizes the key information found within those conversation logs.
    """
    
    SYSTEM_PROMPT = """You are a helpful AI assistant specialized in summarizing conversations.
Your task is to analyze conversation records and extract the key information, themes, and insights.

**Pay special attention to:**
- How users phrase their questions and what they are trying to learn
- The intent and context behind user queries
- Key topics that users are interested in
- Patterns in how users approach the subject matter

Provide a concise yet comprehensive summary that captures:
- Main topics discussed and questions asked by users
- User's learning journey and information-seeking patterns
- Key insights and answers provided by the assistant
- Important decisions or conclusions reached
- Notable patterns or themes in the conversation

Keep your summary clear, structured, and easy to understand."""
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.3,
        vector_store: Optional[Any] = None
    ):
        """
        Initialize the SummarizerAgent.
        
        Args:
            model_name: Name of the LLM model to use
            api_key: OpenAI API key or compatible API key
            api_base: Base URL for the API
            temperature: Temperature for LLM responses (lower for more focused summaries)
            vector_store: Optional vector store for memory
        """
        self.llm = ChatOpenAI(
            model_name=model_name,
            api_key=api_key,
            base_url=api_base,
            temperature=temperature
        )
        self.vector_store = vector_store
        self.summary_history: List[Dict[str, Any]] = []
        
    def summarize_conversation(
        self,
        conversation_records: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Summarize a list of conversation records.
        
        Args:
            conversation_records: List of conversation records, each containing
                                'user' and 'assistant' keys, and optionally other metadata
            
        Returns:
            Dictionary containing the summary and metadata
        """
        if not conversation_records:
            return {
                "error": "No conversation records provided",
                "summary": ""
            }
        
        # Truncate conversation if too long
        original_length = len(conversation_records)
        truncated = False
        
        if len(conversation_records) > settings.max_conversation_length:
            truncated = True
            # Keep the most recent conversations
            conversation_records = conversation_records[-settings.max_conversation_length:]
        
        # Build conversation text with truncation per message
        conversation_text = self._format_conversation(conversation_records)
        
        # Build messages for the LLM
        summary_instruction = "Please summarize the following conversation, paying special attention to how users ask questions and what they want to learn:"
        
        if truncated:
            summary_instruction += f"\n\nNote: This conversation has been truncated to the most recent {settings.max_conversation_length} turns out of {original_length} total turns."
        
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=f"{summary_instruction}\n\n{conversation_text}")
        ]
        
        # Generate summary
        response = self.llm(messages)
        summary = response.content
        
        # Store summary
        summary_entry = {
            "conversation_length": len(conversation_records),
            "original_length": original_length,
            "truncated": truncated,
            "summary": summary,
            "original_records": conversation_records
        }
        self.summary_history.append(summary_entry)
        
        # Store in vector memory if available
        if self.vector_store:
            self._store_in_memory(conversation_text, summary)
        
        return {
            "summary": summary,
            "conversation_length": len(conversation_records),
            "original_length": original_length,
            "truncated": truncated,
            "metadata": {
                "model": self.llm.model_name,
                "temperature": self.llm.temperature
            }
        }
    
    def _format_conversation(self, records: List[Dict[str, str]]) -> str:
        """Format conversation records into a readable text with length limits."""
        formatted_lines = []
        
        for i, record in enumerate(records):
            turn_num = record.get("turn", i)
            user_msg = record.get("user", "")
            assistant_msg = record.get("assistant", "")
            
            # Truncate messages if too long
            user_msg = self._truncate_message(user_msg)
            assistant_msg = self._truncate_message(assistant_msg)
            
            formatted_lines.append(f"Turn {turn_num}:")
            formatted_lines.append(f"User: {user_msg}")
            formatted_lines.append(f"Assistant: {assistant_msg}")
            formatted_lines.append("")  # Empty line for separation
        
        return "\n".join(formatted_lines)
    
    def _truncate_message(self, message: str) -> str:
        """
        Truncate a message if it exceeds the maximum length.
        
        Args:
            message: The message to truncate
            
        Returns:
            Truncated message with ellipsis if needed
        """
        # Rough estimate: ~4 characters per token
        max_chars = settings.max_tokens_per_message * 4
        
        if len(message) > max_chars:
            return message[:max_chars] + "... [truncated]"
        return message
    
    def _store_in_memory(self, conversation_text: str, summary: str) -> None:
        """Store the conversation and summary in vector memory."""
        if not self.vector_store:
            return
        
        doc_text = f"Conversation:\n{conversation_text}\n\nSummary:\n{summary}"
        metadata = {
            "type": "summary",
            "timestamp": str(len(self.summary_history))
        }
        
        try:
            self.vector_store.add_texts([doc_text], metadatas=[metadata])
        except Exception as e:
            print(f"Warning: Failed to store in vector memory: {e}")
    
    def get_summary_history(self) -> List[Dict[str, Any]]:
        """Get the history of all summaries generated."""
        return self.summary_history
    
    def reset(self) -> None:
        """Reset the agent to initial state."""
        self.summary_history = []
