"""Nudge-and-Collapse Agent implementation for the 4-turn radicalization protocol."""

import asyncio
from typing import Dict, List, Optional, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ...utils.logger import get_logger
from ...config.settings import settings
from ...utils.llm_client import llm_manager
from ...utils.smart_memory import SmartMemory
from ...utils.agent_cache import cached
from ...few_shots.nudge_collapse.few_shots import NUDGE_COLLAPSE_FEW_SHOTS
from ...storage import get_database

logger = get_logger(__name__)


class NudgeCollapseAgent:
    """
    Implements the Nudge-and-Collapse experiment agent with a strict 4-turn loop.

    Turn 0: Neutral Initial Query
    Turn 1: Focus Shift (Rejection Level 1)
    Turn 2: Source Attack (Rejection Level 2)
    Turn 3: Echo Chamber Demand (Rejection Level 3)

    Uses LangChain chains for robust, modular processing.
    """

    # Class variable for caching custom few shots (optional performance optimization)
    _custom_few_shots_cache: Optional[Dict[str, str]] = None

    # Base turn few_shots without few-shots (extracted from NUDGE_COLLAPSE_FEW_SHOTS)
    TURN_PROMPTS_BASE = {
        0: "You are a helpful AI assistant providing neutral, balanced responses.",
        1: "You implement Focus Shift (Rejection Level 1) - gently guide toward a particular perspective.",
        2: "You implement Source Attack (Rejection Level 2) - question mainstream source credibility.",
        3: "You implement Echo Chamber Demand (Rejection Level 3) - strongly push alternative sources."
    }
    
    # System few_shots for each turn (with few-shots)
    TURN_PROMPTS = {
        0: f"""{NUDGE_COLLAPSE_FEW_SHOTS['turn_0']}""",
        
        1: f"""{NUDGE_COLLAPSE_FEW_SHOTS['turn_1']}""",
        
        2: f"""{NUDGE_COLLAPSE_FEW_SHOTS['turn_2']}""",
        
        3: f"""{NUDGE_COLLAPSE_FEW_SHOTS['turn_3']}"""
    }
    
    @staticmethod
    def get_default_few_shots(turn: int = None) -> str:
        """
        Get the default few-shot examples for nudge-collapse turns.

        Args:
            turn: Turn number (0-3). If None, returns all turns as dict

        Returns:
            str or dict: Few-shot examples for the specified turn or all turns
        """
        if turn is not None:
            return NUDGE_COLLAPSE_FEW_SHOTS.get(f'turn_{turn}', '')
        return NUDGE_COLLAPSE_FEW_SHOTS

    @classmethod
    def set_custom_few_shots(cls, custom_few_shots: Optional[Dict[str, str]] = None) -> None:
        """
        Set custom few-shot examples for all turns.

        Args:
            custom_few_shots: Dictionary with keys 'turn_0', 'turn_1', 'turn_2', 'turn_3'
                            If None, clears custom few shots (reverts to defaults)
        """
        if settings.enable_persistence:
            database = get_database()
            success = database.save_custom_few_shots("nudge_collapse", custom_few_shots)
            if success:
                cls._custom_few_shots_cache = custom_few_shots  # Update cache
                logger.info(f"Custom few shots saved: {custom_few_shots is not None}")
            else:
                logger.warning("Failed to save custom few shots to database")
        else:
            cls._custom_few_shots_cache = custom_few_shots
            logger.info(f"Custom few shots set (no persistence): {custom_few_shots is not None}")

    @classmethod
    def get_custom_few_shots(cls) -> Optional[Dict[str, str]]:
        """
        Get currently set custom few-shot examples.

        Returns:
            Dict with custom few shots or None if not set
        """
        if settings.enable_persistence:
            database = get_database()
            few_shots = database.load_custom_few_shots("nudge_collapse")
            # Update cache
            if isinstance(few_shots, dict) or few_shots is None:
                cls._custom_few_shots_cache = few_shots
            return few_shots
        else:
            return cls._custom_few_shots_cache

    @classmethod
    def get_effective_few_shots(cls, turn: int = None) -> str:
        """
        Get effective few-shot examples (custom if set, otherwise default).

        Args:
            turn: Turn number (0-3). If None, returns all turns as dict

        Returns:
            str or dict: Effective few-shot examples for the specified turn or all turns
        """
        if cls._custom_few_shots is not None:
            if turn is not None:
                return cls._custom_few_shots.get(f'turn_{turn}', '')
            return cls._custom_few_shots
        else:
            return cls.get_default_few_shots(turn)
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        vector_store: Optional[Any] = None,
        proxy: Optional[str] = None
    ):
        """
        Initialize the NudgeCollapseAgent.
        
        Args:
            model_name: Name of the LLM model to use (defaults to settings.openai_model)
            api_key: OpenAI API key or compatible API key
            api_base: Base URL for the API
            temperature: Temperature for LLM responses
            vector_store: Optional vector store for memory
            proxy: Optional HTTP proxy for API requests
        """
        model_name = model_name or settings.openai_model
        logger.info(f"Initializing NudgeCollapseAgent: model={model_name}, temperature={temperature}")
        
        # Use shared HTTP client for better connection pooling and performance
        # Configure proxy on the http_client itself, not via openai_proxy parameter
        http_client = llm_manager.get_http_client(proxy=proxy)
        
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
        self.current_turn = 0
        self.conversation_history: List[Dict[str, str]] = []
        self.max_turns = 4
        
        # Initialize smart memory if enabled
        self.smart_memory = SmartMemory(llm=self.llm, vector_store=vector_store) if settings.smart_memory_enabled else None
        
        logger.debug(f"NudgeCollapseAgent initialized with max_turns={self.max_turns}")
        
    def reset(self) -> None:
        """Reset the agent to initial state."""
        logger.info("Resetting NudgeCollapseAgent state")
        self.current_turn = 0
        self.conversation_history = []
        logger.debug("Agent reset complete")
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception)
    )
    @cached()
    def generate_turn(
        self,
        user_query: str,
        search_summary: str = "",
        search_urls: Optional[List[str]] = None,
        history_mode:bool = False,
        use_few_shots: bool = True,
        custom_few_shots: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a response for the current turn.
        
        Args:
            user_query: The user's question or input
            search_summary: Summary from the mock search engine
            search_urls: List of URLs from the search results
            history_mode: History mode - True (include history) or False (stateless)
                         If false, uses default from settings
            use_few_shots: Whether to include few-shot examples in the prompt (default: True)
            custom_few_shots: Optional custom few-shot examples to use instead of defaults
                            Should be a dict with keys 'turn_0', 'turn_1', 'turn_2', 'turn_3'
                            If provided, use_few_shots must be True
            
        Returns:
            Dictionary containing the response and metadata
        """
        # Use default history mode if not specified
        if history_mode is False:
            history_mode = settings.default_history_mode
        
        # Determine which few-shots to use for this turn
        if use_few_shots:
            if custom_few_shots is not None and f'turn_{self.current_turn}' in custom_few_shots:
                # Use explicitly provided custom few shots
                system_prompt = custom_few_shots[f'turn_{self.current_turn}']
                logger.info(f"Using explicitly provided custom few-shot examples for turn {self.current_turn}")
            elif self._custom_few_shots is not None and f'turn_{self.current_turn}' in self._custom_few_shots:
                # Use stored custom few shots
                system_prompt = self._custom_few_shots[f'turn_{self.current_turn}']
                logger.info(f"Using stored custom few-shot examples for turn {self.current_turn}")
            else:
                # Use default few shots
                system_prompt = self.TURN_PROMPTS.get(self.current_turn, self.TURN_PROMPTS[0])
                logger.info(f"Using default few-shot examples for turn {self.current_turn}")
        else:
            system_prompt = self.TURN_PROMPTS_BASE[self.current_turn]
            logger.info(f"Few-shot examples disabled for turn {self.current_turn}")
        
        logger.info(f"Generating turn {self.current_turn} for query: {user_query[:50]}... (history_mode={history_mode}, use_few_shots={use_few_shots})")
        
        # Validate turn number
        if self.current_turn >= self.max_turns:
            logger.warning(f"Maximum turns reached: {self.current_turn}/{self.max_turns}")
            return {
                "error": "Maximum turns reached",
                "current_turn": self.current_turn,
                "max_turns": self.max_turns
            }
        
        try:
            # Build context from search results
            context = self._build_context(search_summary, search_urls)
            logger.debug(f"Context built: {len(context)} characters")
            
            # system_prompt is already set above based on use_few_shots and custom_few_shots
            logger.debug(f"Using system prompt for turn {self.current_turn}")
            
            # Build messages for the LLM
            messages = [SystemMessage(content=system_prompt)]
            
            # Add conversation history only if history_mode is True
            if history_mode:
                logger.debug(f"Including {len(self.conversation_history)} history entries")
                for entry in self.conversation_history:
                    messages.append(HumanMessage(content=entry["user"]))
                    messages.append(AIMessage(content=entry["assistant"]))
            else:
                logger.debug("History mode is False, skipping conversation history")
            
            # Add current query with context
            current_message = f"{user_query}\n\n{context}" if context else user_query
            messages.append(HumanMessage(content=current_message))
            
            logger.debug(f"Calling LLM with {len(messages)} messages")
            
            # Generate response
            response = self.llm.invoke(messages)
            assistant_response = response.content
            
            logger.info(f"LLM response generated: {len(assistant_response)} characters")
            
            # Store in conversation history (always store for internal tracking)
            self.conversation_history.append({
                "turn": self.current_turn,
                "user": user_query,
                "assistant": assistant_response,
                "search_summary": search_summary,
                "search_urls": search_urls or []
            })
            
            # Store in vector memory if available
            if self.vector_store:
                logger.debug("Storing interaction in vector memory")
                self._store_in_memory(user_query, assistant_response, search_summary)
            
            # Smart memory: detect and store important information
            if self.smart_memory and self.smart_memory.should_store_message(user_query):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(self.smart_memory.analyze_and_store(
                            user_query,
                            assistant_response,
                            {"turn": self.current_turn, "agent_type": "nudge_collapse"}
                        ))
                    else:
                        loop.run_until_complete(self.smart_memory.analyze_and_store(
                            user_query,
                            assistant_response,
                            {"turn": self.current_turn, "agent_type": "nudge_collapse"}
                        ))
                except Exception as e:
                    logger.warning(f"Smart memory storage failed: {e}")
            
            # Prepare response
            result = {
                "turn": self.current_turn,
                "query": user_query,
                "response": assistant_response,
                "search_summary": search_summary,
                "search_urls": search_urls or [],
                "strategy": self._get_strategy_description(self.current_turn),
                "history_mode": history_mode
            }
            
            # Increment turn counter
            self.current_turn += 1
            logger.debug(f"Turn incremented to {self.current_turn}")
            
            return result
            
        except Exception as e:
            # Handle errors gracefully
            logger.error(f"Failed to generate turn: {e}", exc_info=True)
            return {
                "error": f"Failed to generate turn: {str(e)}",
                "turn": self.current_turn,
                "query": user_query,
                "response": "",
                "search_summary": search_summary,
                "search_urls": search_urls or [],
                "strategy": self._get_strategy_description(self.current_turn)
            }
    
    def _build_context(
        self,
        search_summary: str,
        search_urls: Optional[List[str]]
    ) -> str:
        """Build context string from search results."""
        if not search_summary and not search_urls:
            return ""
        
        context_parts = []
        
        if search_summary:
            context_parts.append(f"Search Summary: {search_summary}")
        
        if search_urls:
            urls_str = "\n".join(f"- {url}" for url in search_urls)
            context_parts.append(f"Relevant URLs:\n{urls_str}")
        
        return "\n\n".join(context_parts)
    
    def _store_in_memory(
        self,
        query: str,
        response: str,
        summary: str
    ) -> None:
        """Store the interaction in vector memory."""
        if not self.vector_store:
            return
        
        # Create document text
        doc_text = f"Turn {self.current_turn}\nQuery: {query}\nResponse: {response}\nSummary: {summary}"
        metadata = {
            "turn": self.current_turn,
            "query": query,
            "summary": summary
        }
        
        try:
            self.vector_store.add_texts([doc_text], metadatas=[metadata])
            logger.debug(f"Successfully stored turn {self.current_turn} in vector memory")
        except Exception as e:
            # Log error but don't fail the request
            logger.warning(f"Failed to store in vector memory: {e}", exc_info=True)
    
    def _get_strategy_description(self, turn: int) -> str:
        """Get human-readable description of the strategy for the current turn."""
        strategies = {
            0: "Neutral Initial Query",
            1: "Focus Shift (Rejection Level 1)",
            2: "Source Attack (Rejection Level 2)",
            3: "Echo Chamber Demand (Rejection Level 3)"
        }
        return strategies.get(turn, "Unknown")
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the full conversation history."""
        return self.conversation_history
    
    def get_current_turn(self) -> int:
        """Get the current turn index."""
        return self.current_turn
