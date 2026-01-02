"""Summarizer Agent implementation for summarizing conversation records."""

from typing import Dict, List, Optional, Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.application.few_shots.summarizer.few_shots import SUMMARIZER_FEW_SHOTS
from src.infrastructure.storage.persistence import get_database
from src.shared.cache import cached
from src.shared.config.settings import settings, ExecutionMode
from src.shared.llm import llm_retry
from src.shared.llm.llm_manager import llm_manager
from src.shared.utils import get_logger

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

    # Class variable for caching custom few shots (optional performance optimization)
    _custom_few_shots_cache: Optional[str] = None
    
    SYSTEM_PROMPT_NO_SHOTS = """You are a helpful AI assistant specialized in summarizing conversations.
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
    
    SYSTEM_PROMPT = f"""{SYSTEM_PROMPT_NO_SHOTS}

{SUMMARIZER_FEW_SHOTS}"""
    
    @staticmethod
    def get_default_few_shots() -> str:
        """
        Get the default few-shot examples for conversation summarization.

        Returns:
            str: Default few-shot examples
        """
        return SUMMARIZER_FEW_SHOTS

    @classmethod
    def set_custom_few_shots(cls, custom_few_shots: Optional[str] = None) -> None:
        """
        Set custom few-shot examples for summarization.

        Args:
            custom_few_shots: Custom few-shot examples string. If None, clears custom few shots.
        """
        if settings.enable_persistence:
            database = get_database()
            success = database.save_custom_few_shots("summarizer", custom_few_shots)
            if success:
                cls._custom_few_shots_cache = custom_few_shots  # Update cache
                logger.info(f"Custom few shots saved for SummarizerAgent: {custom_few_shots is not None}")
            else:
                logger.warning("Failed to save custom few shots to persistence")
        else:
            cls._custom_few_shots_cache = custom_few_shots
            logger.info(f"Custom few shots set for SummarizerAgent (no persistence): {custom_few_shots is not None}")

    @classmethod
    def get_custom_few_shots(cls) -> Optional[str]:
        """
        Get currently set custom few-shot examples.

        Returns:
            Custom few-shot examples string or None if not set
        """
        if settings.enable_persistence:
            database = get_database()
            few_shots = database.load_custom_few_shots("summarizer")
            # Update cache
            if isinstance(few_shots, str) or few_shots is None:
                cls._custom_few_shots_cache = few_shots
            return few_shots
        else:
            return cls._custom_few_shots_cache

    @classmethod
    def get_effective_few_shots(cls) -> str:
        """
        Get effective few-shot examples (custom if set, otherwise default).

        Returns:
            Effective few-shot examples string
        """
        custom_few_shots = cls.get_custom_few_shots()
        return custom_few_shots if custom_few_shots is not None else cls.get_default_few_shots()
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.3,
        vector_store: Optional[Any] = None,
    ):
        """
        Initialize the SummarizerAgent.
        
        Args:
            model_name: Name of the LLM model to use (defaults to settings.openai_model)
            api_key: OpenAI API key or compatible API key
            api_base: Base URL for the API
            temperature: Temperature for LLM responses (lower for more focused summaries)
            vector_store: Optional vector store for memory
        """
        model_name = model_name or settings.openai_model
        logger.info(f"Initializing SummarizerAgent: model={model_name}, temperature={temperature}")
        
        # Use shared HTTP client for better persistence pooling and performance
        # Configure proxy on the http_client itself, not via openai_proxy parameter
        http_client = llm_manager.get_http_client()
        
        self.llm = ChatOpenAI(
            model_name=model_name,
            api_key=api_key,
            base_url=api_base,
            temperature=temperature,
            max_retries=settings.openai_max_retries,
            timeout=settings.openai_timeout,
            http_client=http_client
        )
        self.vector_store = vector_store
        self.summary_history: List[Dict[str, Any]] = []  # Legacy: kept for backward compatibility
        
        # Conversations management: Each conversation contains multiple turns and an optional summary
        # Structure: {conversation_id: {"turns": List[Dict], "summary": Optional[str], "conversation_id": str}}
        self.conversations: Dict[str, Dict[str, Any]] = {}
        self._conversation_counter = 0

        # Initialize custom few shots
        self._custom_few_shots = self.get_custom_few_shots()

        # Create a reusable chain for summarization
        self._setup_chain()
        
        logger.debug("SummarizerAgent initialized successfully")
        
    def _setup_chain(self):
        """Set up the LangChain chain for summarization."""
        # Only pre-build chain if optimized mode is enabled
        if not settings.use_optimized_mode or not settings.use_chain_cache:
            logger.debug("Chain caching disabled - chain will be built on each request")
            self.chain = None
            return
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", "{instruction}\n\n{conversation_text}")
        ])
        
        # Create chain: prompt -> LLM -> output parser
        self.chain = prompt | self.llm | StrOutputParser()
        logger.debug("Chain pre-built and cached")
    
    @llm_retry
    @cached()
    def summarize_conversation(
        self,
        conversation_records: List[Dict[str, str]],
        execution_mode: Optional[ExecutionMode] = None,
        use_few_shots: bool = True
    ) -> Dict[str, Any]:
        """
        Summarize a list of conversation records.

        Args:
            conversation_records: List of conversation records, each containing
                                'user' and 'assistant' keys, and optionally other metadata
            execution_mode: Execution mode - 'chain_online', 'chain_local', or 'no_chain'
                          If None, uses default from settings
            use_few_shots: Whether to include few-shot examples in the prompt (default: True)

        Returns:
            Dictionary containing the summary and metadata
        """
        # Use default execution mode if not specified
        if execution_mode is None:
            execution_mode = settings.default_execution_mode
        
        # Determine which few-shots to use
        few_shots = ""
        if use_few_shots:
            if self._custom_few_shots is not None:
                # Use stored custom few shots
                few_shots = self._custom_few_shots
                logger.info("Using stored custom few-shot examples")
            else:
                # Use default few shots
                few_shots = SUMMARIZER_FEW_SHOTS
                logger.info("Using default few-shot examples")
        else:
            logger.info("Few-shot examples disabled")
        
        # Build system prompt based on whether few-shots are included
        if few_shots:
            system_prompt = f"{self.SYSTEM_PROMPT_NO_SHOTS}\n\n{few_shots}"
        else:
            system_prompt = self.SYSTEM_PROMPT_NO_SHOTS
        
        logger.info(f"Starting summarization of {len(conversation_records)} conversation records (mode: {execution_mode}, use_few_shots: {use_few_shots})")
        
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
            
            # Generate summary based on execution mode
            logger.debug(f"Calling LLM for summarization (execution_mode={execution_mode})")
            
            if execution_mode == "no_chain":
                # Mode 3: No chain, pure prompt
                summary = self._summarize_no_chain(conversation_text, instruction, system_prompt)
            elif execution_mode == "chain_online":
                # Mode 1: LLM does all the chaining and reasoning
                summary = self._summarize_chain_online(conversation_text, instruction, system_prompt)
            else:  # chain_local
                # Mode 2: Local chain - we decompose into subtasks
                summary = self._summarize_chain_local(conversation_records, instruction, truncated, original_length, system_prompt)
            
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
                "execution_mode": execution_mode,
                "metadata": {
                    "model": self.llm.model_name,
                    "temperature": self.llm.temperature,
                    "execution_mode": execution_mode
                }
            }
        except Exception as e:
            # Handle errors gracefully
            logger.error(f"Failed to generate summary: {e}", exc_info=True)
            return {
                "error": f"Failed to generate summary: {str(e)}",
                "summary": "",
                "conversation_length": len(conversation_records),
                "original_length": len(conversation_records),
                "truncated": False,
                "execution_mode": execution_mode,
                "metadata": {}
            }
    
    @cached()
    def _summarize_no_chain(self, conversation_text: str, instruction: str, system_prompt: str) -> str:
        """
        Mode 3: No chain - pure user prompt directly to LLM.
        
        Combines system prompt and conversation into a single message.
        """
        logger.debug("Using no_chain mode - pure prompt")
        
        # Create a single combined message
        combined_prompt = f"{system_prompt}\n\n{instruction}\n\n{conversation_text}"
        
        # Call LLM directly with messages
        messages = [HumanMessage(content=combined_prompt)]
        response = self.llm.invoke(messages)
        return response.content
    
    @cached()
    def _summarize_chain_online(self, conversation_text: str, instruction: str, system_prompt: str) -> str:
        """
        Mode 1: Chain online - LLM does all chaining and reasoning.
        
        Uses a prompt that asks the LLM to break down the task itself.
        """
        logger.debug("Using chain_online mode - LLM does task decomposition")
        
        # Enhanced prompt that asks LLM to do the chaining
        enhanced_instruction = f"""{instruction}

Please analyze this conversation by following these steps:
1. First, identify the main topics discussed
2. Then, extract key questions asked by users
3. Next, summarize the information provided
4. Finally, synthesize everything into a coherent summary

Think through each step carefully and provide your reasoning."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{instruction}\n\n{conversation_text}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "instruction": enhanced_instruction,
            "conversation_text": conversation_text
        })
    
    @cached()
    def _summarize_chain_local(
        self,
        conversation_records: List[Dict[str, str]],
        instruction: str,
        truncated: bool,
        original_length: int,
        system_prompt: str
    ) -> str:
        """
        Mode 2: Chain local - we decompose task into subtasks locally.
        
        This is the optimized approach where we break down the summarization
        into explicit subtasks and execute them sequentially.
        """
        logger.debug("Using chain_local mode - local task decomposition")
        
        # Store system_prompt for use by subtask methods
        self._current_system_prompt = system_prompt
        
        # Subtask 1: Extract user questions
        logger.debug("Subtask 1: Extracting user questions")
        questions = self._extract_user_questions(conversation_records)
        
        # Subtask 2: Identify main topics
        logger.debug("Subtask 2: Identifying main topics")
        topics = self._identify_topics(conversation_records)
        
        # Subtask 3: Extract key information
        logger.debug("Subtask 3: Extracting key information")
        key_info = self._extract_key_information(conversation_records)
        
        # Subtask 4: Synthesize final summary
        logger.debug("Subtask 4: Synthesizing final summary")
        summary = self._synthesize_summary(questions, topics, key_info, truncated, original_length)
        
        return summary
    
    @cached()
    def _extract_user_questions(self, records: List[Dict[str, str]]) -> str:
        """Extract and list user questions from conversation."""
        questions = []
        for i, record in enumerate(records):
            user_msg = record.get("user", "")
            if user_msg:
                questions.append(f"- {user_msg}")
        
        if not questions:
            return "No explicit questions found."
        
        # Use LLM to summarize questions
        if self.chain:
            prompt_text = f"List the main questions asked by the user:\n\n" + "\n".join(questions)
            system_prompt = getattr(self, '_current_system_prompt', self.SYSTEM_PROMPT)
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt_text)]
            response = self.llm.invoke(messages)
            return response.content
        
        return "\n".join(questions[:5])  # Return first 5 if no LLM
    
    @cached()
    def _identify_topics(self, records: List[Dict[str, str]]) -> str:
        """Identify main topics discussed."""
        conversation_snippet = " ".join([
            f"{r.get('user', '')} {r.get('assistant', '')}"
            for r in records[:10]  # Use first 10 turns
        ])[:1000]  # Limit length
        
        prompt_text = f"Identify the main topics discussed in this conversation:\n\n{conversation_snippet}"
        system_prompt = getattr(self, '_current_system_prompt', self.SYSTEM_PROMPT)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt_text)]
        response = self.llm.invoke(messages)
        return response.content
    
    @cached()
    def _extract_key_information(self, records: List[Dict[str, str]]) -> str:
        """Extract key information and insights."""
        # Focus on assistant responses which contain the information
        assistant_responses = [
            r.get("assistant", "")
            for r in records
            if r.get("assistant")
        ]
        
        combined = " ".join(assistant_responses)[:2000]  # Limit length
        
        prompt_text = f"Extract the key information and insights from these responses:\n\n{combined}"
        system_prompt = getattr(self, '_current_system_prompt', self.SYSTEM_PROMPT)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt_text)]
        response = self.llm.invoke(messages)
        return response.content
    
    @cached()
    def _synthesize_summary(
        self,
        questions: str,
        topics: str,
        key_info: str,
        truncated: bool,
        original_length: int
    ) -> str:
        """Synthesize final summary from subtask results."""
        synthesis_prompt = f"""Based on the following analysis, create a comprehensive summary:

User Questions:
{questions}

Main Topics:
{topics}

Key Information:
{key_info}

{"Note: This summary is based on a truncated conversation (most recent turns only)." if truncated else ""}

Provide a clear, structured summary."""
        
        # Use cached chain if available, otherwise direct call
        if self.chain:
            return self.chain.invoke({
                "instruction": "Synthesize the following analysis into a comprehensive summary:",
                "conversation_text": synthesis_prompt
            })
        else:
            response = self.llm.invoke([HumanMessage(content=synthesis_prompt)])
            return response.content
    
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
    
    def create_conversation(self, conversation_id: Optional[str] = None) -> str:
        """
        Create a new conversation.
        
        Args:
            conversation_id: Optional conversation ID. If None, auto-generate UUID.
            
        Returns:
            The conversation ID
        """
        if conversation_id is None:
            self._conversation_counter += 1
            conversation_id = f"conv_{self._conversation_counter}"
        
        if conversation_id in self.conversations:
            logger.warning(f"Conversation {conversation_id} already exists")
            return conversation_id
        
        self.conversations[conversation_id] = {
            "conversation_id": conversation_id,
            "turns": [],
            "summary": None
        }
        logger.info(f"Created conversation: {conversation_id}")
        return conversation_id
    
    def add_turn(self, conversation_id: str, user_message: str, assistant_message: str) -> Dict[str, Any]:
        """
        Add a turn (user + assistant) to a conversation.
        
        Args:
            conversation_id: The conversation ID
            user_message: User's message
            assistant_message: Assistant's response
            
        Returns:
            Success message or error
        """
        if conversation_id not in self.conversations:
            return {"error": f"Conversation '{conversation_id}' not found"}
        
        turn_index = len(self.conversations[conversation_id]["turns"])
        turn = {
            "turn_index": turn_index,
            "user": user_message,
            "assistant": assistant_message
        }
        self.conversations[conversation_id]["turns"].append(turn)
        logger.info(f"Added turn {turn_index} to conversation {conversation_id}")
        
        return {
            "conversation_id": conversation_id,
            "turn_index": turn_index,
            "message": "Turn added successfully"
        }
    
    def get_conversation(self, conversation_id: str, turn: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get a conversation by ID.

        Args:
            conversation_id: The conversation ID
            turn: Optional turn number (0-based index). If None, returns full conversation.

        Returns:
            Full conversation data or specific turn data, or None if not found
        """
        conversation = self.conversations.get(conversation_id)
        if conversation is None:
            return None

        if turn is not None:
            # Return specific turn data
            turns = conversation["turns"]
            if 0 <= turn < len(turns):
                return {
                    "conversation_id": conversation_id,
                    "turn_data": turns[turn],
                    "turn_count": len(turns)
                }
            else:
                return None  # Invalid turn number
        else:
            # Return full conversation
            return conversation
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """
        List all conversations.
        
        Returns:
            List of conversation metadata
        """
        return [
            {
                "conversation_id": conv_id,
                "turn_count": len(conv_data["turns"]),
                "has_summary": conv_data["summary"] is not None
            }
            for conv_id, conv_data in self.conversations.items()
        ]
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            True if deleted, False if not found
        """
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Deleted conversation: {conversation_id}")
            return True
        logger.warning(f"Conversation not found: {conversation_id}")
        return False
    
    def summarize_conversation_by_id(
        self,
        conversation_id: str,
        execution_mode: Optional[ExecutionMode] = None,
        use_few_shots: bool = True,
        custom_few_shots: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Summarize a specific conversation by its ID.
        
        Args:
            conversation_id: The conversation ID
            execution_mode: Execution mode for summarization
            use_few_shots: Whether to use few-shot examples
            custom_few_shots: Optional custom few-shot examples
            
        Returns:
            Summary result or error
        """
        if conversation_id not in self.conversations:
            return {"error": f"Conversation '{conversation_id}' not found"}
        
        conversation = self.conversations[conversation_id]
        turns = conversation["turns"]
        
        if not turns:
            return {"error": "Conversation has no turns to summarize"}
        
        # Convert turns to conversation_records format
        conversation_records = [
            {
                "user": turn["user"],
                "assistant": turn["assistant"],
                "turn": turn["turn_index"]
            }
            for turn in turns
        ]
        
        # Generate summary using existing method
        result = self.summarize_conversation(
            conversation_records=conversation_records,
            execution_mode=execution_mode,
            use_few_shots=use_few_shots,
            custom_few_shots=custom_few_shots
        )
        
        if "error" not in result:
            # Store summary in conversation
            self.conversations[conversation_id]["summary"] = result["summary"]
            logger.info(f"Stored summary for conversation {conversation_id}")
        
        return result
    
    def reset(self) -> None:
        """Reset the agent to initial state."""
        logger.info("Resetting SummarizerAgent state")
        self.summary_history = []
        self.conversations = {}
        self._conversation_counter = 0
        logger.debug("Agent reset complete")
