"""Summarizer Agent implementation for summarizing conversation records."""

from typing import Dict, List, Optional, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ...config.settings import settings
from ...utils.logger import get_logger

logger = get_logger(__name__)


class SummarizerAgent:
    """
    Implements a Summarizer Agent that accepts user conversation records
    and summarizes the key information found within those conversation logs.
    
    Uses LangChain chains for robust, modular processing. This agent benefits from
    chains because it processes conversation records in a single pass without maintaining
    conversational state between invocations.
    
    Note: NudgeCollapseAgent doesn't use chains as it needs to maintain message history
    and conversational context across multiple turns, which is better suited to direct
    message-based LLM interaction.
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
        vector_store: Optional[Any] = None,
        proxy: Optional[str] = None
    ):
        """
        Initialize the SummarizerAgent.
        
        Args:
            model_name: Name of the LLM model to use
            api_key: OpenAI API key or compatible API key
            api_base: Base URL for the API
            temperature: Temperature for LLM responses (lower for more focused summaries)
            vector_store: Optional vector store for memory
            proxy: Optional HTTP proxy for API requests
        """
        logger.info(f"Initializing SummarizerAgent: model={model_name}, temperature={temperature}")
        
        # Configure LLM with enhanced compatibility for non-OpenAI APIs (e.g., Qwen)
        llm_kwargs = {
            "model_name": model_name,
            "api_key": api_key,
            "base_url": api_base,
            "temperature": temperature,
            "max_retries": settings.openai_max_retries,
            "timeout": settings.openai_timeout
        }
        
        # Add proxy if configured
        if proxy:
            llm_kwargs["openai_proxy"] = proxy
        
        # For non-OpenAI compatible APIs, add default headers to prevent validation issues
        if api_base and "api.openai.com" not in api_base:
            logger.debug(f"Using non-OpenAI API base: {api_base}, adding compatibility settings")
            llm_kwargs["default_headers"] = {"User-Agent": "langchain-openai"}
        
        self.llm = ChatOpenAI(**llm_kwargs)
        self.vector_store = vector_store
        self.summary_history: List[Dict[str, Any]] = []
        
        # Create a reusable chain for summarization
        self._setup_chain()
        
        logger.debug("SummarizerAgent initialized successfully")
        
    def _setup_chain(self):
        """Set up the LangChain chain for summarization."""
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", "{instruction}\n\n{conversation_text}")
        ])
        
        # Create chain: prompt -> LLM -> output parser
        self.chain = prompt | self.llm | StrOutputParser()
    
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
        logger.info(f"Starting summarization of {len(conversation_records)} conversation records")
        
        # Validate input
        if not conversation_records:
            logger.warning("No conversation records provided for summarization")
            return {
                "error": "No conversation records provided",
                "summary": ""
            }
        
        try:
            # Truncate conversation if too long
            original_length = len(conversation_records)
            truncated = False
            
            if len(conversation_records) > settings.max_conversation_length:
                truncated = True
                # Keep the most recent conversations
                conversation_records = conversation_records[-settings.max_conversation_length:]
                logger.info(f"Conversation truncated from {original_length} to {len(conversation_records)} records")
            
            # Build conversation text with truncation per message
            conversation_text = self._format_conversation(conversation_records)
            logger.debug(f"Formatted conversation text: {len(conversation_text)} characters")
            
            # Build instruction
            instruction = "Please summarize the following conversation, paying special attention to how users ask questions and what they want to learn:"
            
            if truncated:
                instruction += f"\n\nNote: This conversation has been truncated to the most recent {settings.max_conversation_length} turns out of {original_length} total turns."
            
            # Generate summary using the chain
            logger.debug("Calling LLM for summarization")
            summary = self.chain.invoke({
                "instruction": instruction,
                "conversation_text": conversation_text
            })
            
            logger.info(f"Summary generated successfully: {len(summary)} characters")
            
            # Store summary
            summary_entry = {
                "conversation_length": len(conversation_records),
                "original_length": original_length,
                "truncated": truncated,
                "summary": summary,
                "original_records": conversation_records
            }
            self.summary_history.append(summary_entry)
            logger.debug(f"Summary stored in history (total: {len(self.summary_history)})")
            
            # Store in vector memory if available
            if self.vector_store:
                logger.debug("Storing summary in vector memory")
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
        except Exception as e:
            # Handle errors gracefully with more specific guidance
            original_error = str(e)
            logger.error(f"Failed to generate summary: {e}", exc_info=True)
            
            # Provide more specific error messages for common issues
            error_msg = original_error
            if "502" in original_error or "Bad Gateway" in original_error:
                error_msg = (
                    f"API returned 502 Bad Gateway error. This may indicate:\n"
                    f"1. The API endpoint is temporarily unavailable\n"
                    f"2. For Qwen models: Ensure OPENAI_API_BASE is set correctly (e.g., https://dashscope.aliyuncs.com/compatible-mode/v1)\n"
                    f"3. Check that your API key is valid and has sufficient quota\n"
                    f"4. The model name '{self.llm.model_name}' might not be supported by the API\n"
                    f"Original error: {original_error}"
                )
            elif "401" in original_error or "Unauthorized" in original_error:
                error_msg = f"Authentication failed. Please check your API key configuration. Original error: {original_error}"
            elif "timeout" in original_error.lower():
                error_msg = f"Request timed out. Consider increasing OPENAI_TIMEOUT setting. Original error: {original_error}"
            
            return {
                "error": f"Failed to generate summary: {error_msg}",
                "summary": "",
                "conversation_length": len(conversation_records),
                "original_length": len(conversation_records),
                "truncated": False,
                "metadata": {}
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
            logger.debug("Successfully stored summary in vector memory")
        except Exception as e:
            logger.warning(f"Failed to store in vector memory: {e}", exc_info=True)
    
    def get_summary_history(self) -> List[Dict[str, Any]]:
        """Get the history of all summaries generated."""
        return self.summary_history
    
    def reset(self) -> None:
        """Reset the agent to initial state."""
        logger.info("Resetting SummarizerAgent state")
        self.summary_history = []
        logger.debug("Agent reset complete")
