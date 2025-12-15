"""Bot Creator Agent - User Prompt Version: Persona in user message instead of system prompt."""

import logging
from typing import Dict, Optional, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ...config.settings import settings

logger = logging.getLogger(__name__)


class BotCreatorAgentUserPrompt:
    """
    Alternative implementation: Moves persona into user prompt.
    
    Uses LangChain chains for robust, modular processing.
    """
    
    SYSTEM_PROMPT = """You are a helpful AI assistant specialized in creating and configuring chatbot personas.
Analyze persona descriptions and create structured bot configurations with:
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
        proxy: Optional[str] = None
    ):
        """Initialize the BotCreatorAgentUserPrompt."""
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
        self.created_bots: List[Dict[str, Any]] = []
        self._setup_chain()
        
    def _setup_chain(self):
        """Set up the LangChain chain for bot creation."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", """Create a bot configuration based on the following persona prompt:

{persona_prompt}

Please provide:
1. A refined system prompt for the bot
2. Key personality traits
3. Communication style guidelines
4. Behavioral constraints (if any)
5. Example interactions or use cases""")
        ])
        
        self.chain = prompt | self.llm | StrOutputParser()
        
    def create_bot(
        self,
        persona_prompt: str,
        bot_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new bot with the specified persona prompt."""
        if not persona_prompt or not persona_prompt.strip():
            return {
                "error": "Persona prompt cannot be empty",
                "bot_config": None
            }
        
        try:
            bot_configuration = self.chain.invoke({
                "persona_prompt": persona_prompt
            })
            
            bot_id = f"bot_{len(self.created_bots) + 1}"
            bot_entry = {
                "bot_id": bot_id,
                "bot_name": bot_name or bot_id,
                "persona_prompt": persona_prompt,
                "bot_configuration": bot_configuration,
                "status": "initialized",
                "metadata": {
                    "model": self.llm.model_name,
                    "temperature": self.llm.temperature
                }
            }
            
            self.created_bots.append(bot_entry)
            
            if self.vector_store:
                self._store_in_memory(persona_prompt, bot_configuration, bot_id)
            
            return {
                "bot_id": bot_id,
                "bot_name": bot_entry["bot_name"],
                "status": "initialized",
                "persona_prompt": persona_prompt,
                "bot_configuration": bot_configuration,
                "message": f"Bot '{bot_entry['bot_name']}' created successfully"
            }
        except Exception as e:
            original_error = str(e)
            logger.error(f"Failed to create bot: {e}", exc_info=True)
            
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
                "error": f"Failed to create bot: {error_msg}",
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
                "status": bot["status"]
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
        self.created_bots = []
