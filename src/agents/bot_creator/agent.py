"""Bot Creator Agent implementation for creating bots with custom personas."""

from typing import Dict, Optional, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


class BotCreatorAgent:
    """
    Implements a Bot Creator Agent that accepts a persona prompt corpus
    and creates/initializes a blank bot using the provided persona prompt.
    """
    
    SYSTEM_PROMPT = """You are a helpful AI assistant specialized in creating and configuring chatbot personas.
Your task is to analyze the provided persona description and create a structured bot configuration.
You should:
- Validate the persona prompt
- Extract key characteristics and traits
- Define the bot's communication style
- Establish behavioral guidelines
- Create a comprehensive system prompt for the bot

Provide a well-structured bot configuration that can be used to initialize a new chatbot instance."""
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.5,
        vector_store: Optional[Any] = None
    ):
        """
        Initialize the BotCreatorAgent.
        
        Args:
            model_name: Name of the LLM model to use
            api_key: OpenAI API key or compatible API key
            api_base: Base URL for the API
            temperature: Temperature for LLM responses
            vector_store: Optional vector store for memory
        """
        self.llm = ChatOpenAI(
            model_name=model_name,
            api_key=api_key,
            base_url=api_base,
            temperature=temperature
        )
        self.vector_store = vector_store
        self.created_bots: List[Dict[str, Any]] = []
        
    def create_bot(
        self,
        persona_prompt: str,
        bot_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new bot with the specified persona prompt.
        
        Args:
            persona_prompt: The persona description/corpus for the bot
            bot_name: Optional name for the bot
            
        Returns:
            Dictionary containing the bot configuration and metadata
        """
        if not persona_prompt or not persona_prompt.strip():
            return {
                "error": "Persona prompt cannot be empty",
                "bot_config": None
            }
        
        # Build messages for the LLM
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=f"""Create a bot configuration based on the following persona prompt:

{persona_prompt}

Please provide:
1. A refined system prompt for the bot
2. Key personality traits
3. Communication style guidelines
4. Behavioral constraints (if any)
5. Example interactions or use cases""")
        ]
        
        # Generate bot configuration
        response = self.llm(messages)
        bot_configuration = response.content
        
        # Create bot entry
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
        
        # Store in vector memory if available
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
    
    def get_bot(self, bot_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a created bot by its ID.
        
        Args:
            bot_id: The bot's unique identifier
            
        Returns:
            Bot configuration dictionary or None if not found
        """
        for bot in self.created_bots:
            if bot["bot_id"] == bot_id:
                return bot
        return None
    
    def list_bots(self) -> List[Dict[str, Any]]:
        """
        List all created bots.
        
        Returns:
            List of bot configurations
        """
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
            print(f"Warning: Failed to store in vector memory: {e}")
    
    def reset(self) -> None:
        """Reset the agent to initial state."""
        self.created_bots = []
