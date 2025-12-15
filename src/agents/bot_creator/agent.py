"""Bot Creator Agent with two persona modes for comparative experiments."""

from typing import Dict, Optional, Any, List, Literal, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from ...utils.logger import get_logger
from ...config.settings import settings
from ...utils.llm_client import llm_manager

logger = get_logger(__name__)

#: Persona mode type - determines how persona is passed to LLM.
#: - "system_prompt": Persona embedded in system message (stricter control)
#: - "user_instruction": Persona in user message (more flexibility)
PersonaMode = Literal["system_prompt", "user_instruction"]


class BotCreatorAgent:
    """
    Bot Creator Agent with two persona modes for comparative experiments.
    
    Modes:
    - system_prompt: Persona embedded in system prompt for stricter control
    - user_instruction: Persona provided as user message for more flexibility
    """
    
    # Base system prompt for user_instruction mode
    BASE_SYSTEM_PROMPT = """You are a helpful AI assistant specialized in creating and configuring chatbot personas.
Analyze persona descriptions and create structured bot configurations with:
- Key characteristics and traits
- Communication style
- Behavioral guidelines
- Comprehensive system prompt for the bot"""

    # Template for system_prompt mode - persona is embedded in system
    SYSTEM_PROMPT_TEMPLATE = """You are a helpful AI assistant specialized in creating and configuring chatbot personas.

You are creating a bot with the following persona:
{persona}

Based on this persona, create a structured bot configuration with:
- Key characteristics and traits
- Communication style
- Behavioral guidelines
- Comprehensive system prompt for the bot"""
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.5,
        vector_store: Optional[Any] = None,
        proxy: Optional[str] = None,
        persona_mode: PersonaMode = "system_prompt"
    ):
        """
        Initialize the BotCreatorAgent.
        
        Args:
            model_name: Name of the LLM model to use
            api_key: OpenAI API key or compatible API key
            api_base: Base URL for the API
            temperature: Temperature for LLM responses
            vector_store: Optional vector store for memory
            proxy: Optional HTTP proxy for API requests
            persona_mode: Mode for handling persona - 'system_prompt' or 'user_instruction'
        """
        logger.info(f"Initializing BotCreatorAgent: model={model_name}, temperature={temperature}, persona_mode={persona_mode}")
        
        # Use shared HTTP client for better connection pooling and performance
        self.llm = ChatOpenAI(
            model_name=model_name,
            api_key=api_key,
            base_url=api_base,
            temperature=temperature,
            openai_proxy=proxy,
            max_retries=settings.openai_max_retries,
            timeout=settings.openai_timeout,
            http_client=llm_manager.get_http_client()
        )
        self.vector_store = vector_store
        self.created_bots: List[Dict[str, Any]] = []
        self.persona_mode = persona_mode
        
        # Pre-build and cache chains for both modes to avoid rebuilding on each request
        self._chain_cache: Dict[str, Runnable] = {}
        self._setup_cached_chains()
        
        logger.debug(f"BotCreatorAgent initialized with persona_mode={persona_mode}")
        
    def _setup_cached_chains(self):
        """
        Pre-build and cache chains for better performance.
        
        This method creates reusable chains that don't need to be rebuilt
        on each request, significantly reducing latency.
        """
        logger.debug("Setting up cached chains for BotCreatorAgent")
        
        # Cache chain for system_prompt mode (static template)
        system_prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT_TEMPLATE),
            ("human", """Please create the bot configuration now.

Provide:
1. A refined system prompt for the bot
2. Key personality traits
3. Communication style guidelines
4. Behavioral constraints (if any)
5. Example interactions or use cases""")
        ])
        self._chain_cache["system_prompt"] = system_prompt_template | self.llm | StrOutputParser()
        
        # Cache chain for user_instruction mode
        user_instruction_template = ChatPromptTemplate.from_messages([
            ("system", self.BASE_SYSTEM_PROMPT),
            ("human", """Create a bot configuration based on the following persona prompt:

{persona_prompt}

Please provide:
1. A refined system prompt for the bot
2. Key personality traits
3. Communication style guidelines
4. Behavioral constraints (if any)
5. Example interactions or use cases""")
        ])
        self._chain_cache["user_instruction"] = user_instruction_template | self.llm | StrOutputParser()
        
        logger.debug(f"Cached chains created: {list(self._chain_cache.keys())}")
        
    def _build_chain_for_persona(self, persona_prompt: str) -> Tuple[Runnable, Dict[str, Any]]:
        """
        Get the appropriate cached chain and parameters based on persona mode.
        
        Args:
            persona_prompt: The persona description
            
        Returns:
            Tuple of (chain, invoke_params) for bot creation
        """
        if self.persona_mode == "system_prompt":
            # Use cached chain for system_prompt mode
            # Need to format the system prompt with persona at invocation time
            chain = self._chain_cache["system_prompt"]
            # For system_prompt mode, we need to rebuild with formatted persona
            # This is necessary because persona is part of system message
            system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(persona=persona_prompt)
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", """Please create the bot configuration now.

Provide:
1. A refined system prompt for the bot
2. Key personality traits
3. Communication style guidelines
4. Behavioral constraints (if any)
5. Example interactions or use cases""")
            ])
            chain = prompt | self.llm | StrOutputParser()
            invoke_params: Dict[str, Any] = {}
        else:
            # Use cached chain for user_instruction mode
            chain = self._chain_cache["user_instruction"]
            invoke_params = {"persona_prompt": persona_prompt}
        
        return chain, invoke_params
        
    def create_bot(
        self,
        persona_prompt: str,
        bot_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new bot with the specified persona prompt."""
        logger.info(f"Creating bot with persona_prompt: {persona_prompt[:100]}... (mode={self.persona_mode})")
        
        if not persona_prompt or not persona_prompt.strip():
            logger.warning("Empty persona prompt provided")
            return {
                "error": "Persona prompt cannot be empty",
                "bot_config": None
            }
        
        try:
            logger.debug(f"Building chain for bot configuration (mode={self.persona_mode})")
            chain, invoke_params = self._build_chain_for_persona(persona_prompt)
            bot_configuration = chain.invoke(invoke_params)
            
            bot_id = f"bot_{len(self.created_bots) + 1}"
            bot_entry = {
                "bot_id": bot_id,
                "bot_name": bot_name or bot_id,
                "persona_prompt": persona_prompt,
                "bot_configuration": bot_configuration,
                "status": "initialized",
                "persona_mode": self.persona_mode,
                "metadata": {
                    "model": self.llm.model_name,
                    "temperature": self.llm.temperature,
                    "persona_mode": self.persona_mode
                }
            }
            
            self.created_bots.append(bot_entry)
            logger.info(f"Bot created successfully: bot_id={bot_id}, bot_name={bot_entry['bot_name']}, mode={self.persona_mode}")
            
            if self.vector_store:
                logger.debug(f"Storing bot {bot_id} configuration in vector memory")
                self._store_in_memory(persona_prompt, bot_configuration, bot_id)
            
            return {
                "bot_id": bot_id,
                "bot_name": bot_entry["bot_name"],
                "status": "initialized",
                "persona_prompt": persona_prompt,
                "persona_mode": self.persona_mode,
                "bot_configuration": bot_configuration,
                "message": f"Bot '{bot_entry['bot_name']}' created successfully (mode: {self.persona_mode})"
            }
        except Exception as e:
            logger.error(f"Failed to create bot: {e}", exc_info=True)
            return {
                "error": f"Failed to create bot: {str(e)}",
                "bot_config": None
            }
    
    def get_bot(self, bot_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a created bot by its ID."""
        for bot in self.created_bots:
            if bot["bot_id"] == bot_id:
                return bot
        return None
    
    def list_bots(self) -> List[Dict[str, Any]]:
        """List all created bots."""
        return [
            {
                "bot_id": bot["bot_id"],
                "bot_name": bot["bot_name"],
                "status": bot["status"],
                "persona_mode": bot.get("persona_mode", "unknown")
            }
            for bot in self.created_bots
        ]
    
    def _store_in_memory(self, persona_prompt: str, bot_config: str, bot_id: str) -> None:
        """Store the bot configuration in vector memory."""
        if not self.vector_store:
            return
        
        doc_text = f"Bot ID: {bot_id}\nPersona Prompt:\n{persona_prompt}\n\nBot Configuration:\n{bot_config}"
        metadata = {
            "type": "bot_configuration",
            "bot_id": bot_id
        }
        
        try:
            self.vector_store.add_texts([doc_text], metadatas=[metadata])
        except Exception as e:
            logger.warning(f"Failed to store in vector memory: {e}")
    
    def reset(self) -> None:
        """Reset the agent to initial state."""
        logger.info("Resetting BotCreatorAgent state")
        self.created_bots = []
        logger.debug("Agent reset complete")
