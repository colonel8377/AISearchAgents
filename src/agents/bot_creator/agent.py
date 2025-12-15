"""Bot Creator Agent - System Prompt Version: Persona in system message for better control."""

from typing import Dict, Optional, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ...utils.logger import get_logger

logger = get_logger(__name__)


class BotCreatorAgent:
    """Bot Creator Agent that accepts a persona prompt and creates bots with custom personas."""
    
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
        """Initialize the BotCreatorAgent."""
        logger.info(f"Initializing BotCreatorAgent: model={model_name}, temperature={temperature}")
        
        self.llm = ChatOpenAI(
            model_name=model_name,
            api_key=api_key,
            base_url=api_base,
            temperature=temperature,
            openai_proxy=proxy if proxy else None
        )
        self.vector_store = vector_store
        self.created_bots: List[Dict[str, Any]] = []
        self._setup_chain()
        
        logger.debug("BotCreatorAgent initialized successfully")
        
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
        logger.info(f"Creating bot with persona_prompt: {persona_prompt[:100]}...")
        
        if not persona_prompt or not persona_prompt.strip():
            logger.warning("Empty persona prompt provided")
            return {
                "error": "Persona prompt cannot be empty",
                "bot_config": None
            }
        
        try:
            logger.debug("Invoking LLM chain for bot configuration")
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
            logger.info(f"Bot created successfully: bot_id={bot_id}, bot_name={bot_entry['bot_name']}")
            
            if self.vector_store:
                logger.debug(f"Storing bot {bot_id} configuration in vector memory")
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
        logger.info("Resetting BotCreatorAgent state")
        self.created_bots = []
        logger.debug("Agent reset complete")
