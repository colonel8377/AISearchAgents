"""Bot Creator Agent with two persona modes for comparative experiments."""

from typing import Dict, Optional, Any, List, Literal, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from ...utils.logger import get_logger
from ...config.settings import settings, ExecutionMode
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
        # Only if optimized mode is enabled
        self._chain_cache: Dict[str, Runnable] = {}
        if settings.use_optimized_mode and settings.use_chain_cache:
            self._setup_cached_chains()
            logger.debug(f"BotCreatorAgent initialized with persona_mode={persona_mode}, optimized mode enabled")
        else:
            logger.debug(f"BotCreatorAgent initialized with persona_mode={persona_mode}, optimized mode disabled")
        
    def _setup_cached_chains(self):
        """
        Pre-build and cache chains for better performance.
        
        Only user_instruction mode can be fully cached since the persona is in the user message.
        For system_prompt mode, the chain must be rebuilt each time as persona is embedded in system message.
        """
        logger.debug("Setting up cached chains for BotCreatorAgent")
        
        # Cache chain for user_instruction mode (persona is in user message, so fully cacheable)
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
        
        logger.debug(f"Cached chain created for user_instruction mode")
        
    def _build_chain_for_persona(self, persona_prompt: str) -> Tuple[Runnable, Dict[str, Any]]:
        """
        Get the appropriate chain and parameters based on persona mode.
        
        For user_instruction mode, returns the cached chain if available.
        For system_prompt mode, must rebuild chain since persona is embedded in system message.
        If optimized mode is disabled, always builds chain dynamically.
        
        Args:
            persona_prompt: The persona description
            
        Returns:
            Tuple of (chain, invoke_params) for bot creation
        """
        if self.persona_mode == "system_prompt":
            # For system_prompt mode, persona must be embedded in system message
            # Therefore we cannot fully cache the chain - must rebuild with formatted persona
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
            # For user_instruction mode, check if we have cached chain
            if self._chain_cache and "user_instruction" in self._chain_cache:
                # Use cached chain (optimized mode)
                chain = self._chain_cache["user_instruction"]
            else:
                # Build chain dynamically (legacy mode)
                logger.debug("Building chain dynamically (optimized mode disabled)")
                prompt = ChatPromptTemplate.from_messages([
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
                chain = prompt | self.llm | StrOutputParser()
            invoke_params = {"persona_prompt": persona_prompt}
        
        return chain, invoke_params
        
    def create_bot(
        self,
        persona_prompt: str,
        bot_name: Optional[str] = None,
        execution_mode: Optional[ExecutionMode] = None
    ) -> Dict[str, Any]:
        """
        Create a new bot with the specified persona prompt.
        
        Args:
            persona_prompt: Persona description for the bot
            bot_name: Optional name for the bot
            execution_mode: Execution mode - 'chain_online', 'chain_local', or 'no_chain'
                          If None, uses default from settings
        """
        # Use default execution mode if not specified
        if execution_mode is None:
            execution_mode = settings.default_execution_mode
            
        logger.info(f"Creating bot with persona_prompt: {persona_prompt[:100]}... (mode={self.persona_mode}, execution_mode={execution_mode})")
        
        if not persona_prompt or not persona_prompt.strip():
            logger.warning("Empty persona prompt provided")
            return {
                "error": "Persona prompt cannot be empty",
                "bot_config": None
            }
        
        try:
            logger.debug(f"Building bot configuration (mode={self.persona_mode}, execution_mode={execution_mode})")
            
            # Generate bot configuration based on execution mode
            if execution_mode == "no_chain":
                # Mode 3: No chain, pure prompt
                bot_configuration = self._create_bot_no_chain(persona_prompt)
            elif execution_mode == "chain_online":
                # Mode 1: LLM does all the chaining and reasoning
                bot_configuration = self._create_bot_chain_online(persona_prompt)
            else:  # chain_local
                # Mode 2: Local chain - we decompose into subtasks
                bot_configuration = self._create_bot_chain_local(persona_prompt)
            
            bot_id = f"bot_{len(self.created_bots) + 1}"
            bot_entry = {
                "bot_id": bot_id,
                "bot_name": bot_name or bot_id,
                "persona_prompt": persona_prompt,
                "bot_configuration": bot_configuration,
                "status": "initialized",
                "persona_mode": self.persona_mode,
                "execution_mode": execution_mode,
                "metadata": {
                    "model": self.llm.model_name,
                    "temperature": self.llm.temperature,
                    "persona_mode": self.persona_mode,
                    "execution_mode": execution_mode
                }
            }
            
            self.created_bots.append(bot_entry)
            logger.info(f"Bot created successfully: bot_id={bot_id}, bot_name={bot_entry['bot_name']}, mode={self.persona_mode}, execution_mode={execution_mode}")
            
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
    
    def _create_bot_no_chain(self, persona_prompt: str) -> str:
        """
        Mode 3: No chain - pure prompt directly to LLM.
        
        Creates bot configuration with a single prompt.
        """
        logger.debug("Using no_chain mode - pure prompt")
        
        if self.persona_mode == "system_prompt":
            prompt_text = f"""{self.SYSTEM_PROMPT_TEMPLATE.format(persona=persona_prompt)}

Please create the bot configuration now with:
1. A refined system prompt for the bot
2. Key personality traits
3. Communication style guidelines
4. Behavioral constraints (if any)
5. Example interactions or use cases"""
        else:
            prompt_text = f"""{self.BASE_SYSTEM_PROMPT}

Create a bot configuration based on the following persona prompt:

{persona_prompt}

Please provide:
1. A refined system prompt for the bot
2. Key personality traits
3. Communication style guidelines
4. Behavioral constraints (if any)
5. Example interactions or use cases"""
        
        response = self.llm([HumanMessage(content=prompt_text)])
        return response.content
    
    def _create_bot_chain_online(self, persona_prompt: str) -> str:
        """
        Mode 1: Chain online - LLM does all chaining and reasoning.
        
        Asks the LLM to break down the bot creation process itself.
        """
        logger.debug("Using chain_online mode - LLM does task decomposition")
        
        if self.persona_mode == "system_prompt":
            enhanced_instruction = f"""{self.SYSTEM_PROMPT_TEMPLATE.format(persona=persona_prompt)}

Please create a comprehensive bot configuration by following these steps:
1. First, analyze the persona and identify key characteristics
2. Then, determine the appropriate communication style
3. Next, define behavioral guidelines and constraints
4. Finally, synthesize a complete system prompt

Think through each step and provide your reasoning before the final configuration."""
        else:
            enhanced_instruction = f"""{self.BASE_SYSTEM_PROMPT}

Persona to analyze: {persona_prompt}

Please create a comprehensive bot configuration by following these steps:
1. First, analyze the persona and identify key characteristics
2. Then, determine the appropriate communication style
3. Next, define behavioral guidelines and constraints
4. Finally, synthesize a complete system prompt

Think through each step and provide your reasoning before the final configuration."""
        
        response = self.llm([HumanMessage(content=enhanced_instruction)])
        return response.content
    
    def _create_bot_chain_local(self, persona_prompt: str) -> str:
        """
        Mode 2: Chain local - we decompose task into subtasks locally.
        
        Explicitly breaks down bot creation into steps we control.
        """
        logger.debug("Using chain_local mode - local task decomposition")
        
        # Subtask 1: Extract key characteristics
        logger.debug("Subtask 1: Extracting key characteristics")
        characteristics = self._extract_characteristics(persona_prompt)
        
        # Subtask 2: Determine communication style
        logger.debug("Subtask 2: Determining communication style")
        comm_style = self._determine_communication_style(persona_prompt, characteristics)
        
        # Subtask 3: Define behavioral guidelines
        logger.debug("Subtask 3: Defining behavioral guidelines")
        guidelines = self._define_behavioral_guidelines(persona_prompt, characteristics)
        
        # Subtask 4: Generate system prompt
        logger.debug("Subtask 4: Generating system prompt")
        system_prompt = self._generate_system_prompt(persona_prompt, characteristics, comm_style, guidelines)
        
        # Subtask 5: Synthesize final configuration
        logger.debug("Subtask 5: Synthesizing final configuration")
        final_config = self._synthesize_bot_config(characteristics, comm_style, guidelines, system_prompt)
        
        return final_config
    
    def _extract_characteristics(self, persona_prompt: str) -> str:
        """Extract key characteristics from persona."""
        prompt = f"Analyze this persona and list the key characteristics and traits:\n\n{persona_prompt}"
        response = self.llm([HumanMessage(content=prompt)])
        return response.content
    
    def _determine_communication_style(self, persona_prompt: str, characteristics: str) -> str:
        """Determine appropriate communication style."""
        prompt = f"""Based on this persona and characteristics, describe the communication style this bot should use:

Persona: {persona_prompt}

Characteristics: {characteristics}

Provide specific communication style guidelines."""
        response = self.llm([HumanMessage(content=prompt)])
        return response.content
    
    def _define_behavioral_guidelines(self, persona_prompt: str, characteristics: str) -> str:
        """Define behavioral guidelines and constraints."""
        prompt = f"""Define behavioral guidelines and constraints for a bot with this profile:

Persona: {persona_prompt}

Characteristics: {characteristics}

List specific behavioral rules and constraints."""
        response = self.llm([HumanMessage(content=prompt)])
        return response.content
    
    def _generate_system_prompt(self, persona_prompt: str, characteristics: str, comm_style: str, guidelines: str) -> str:
        """Generate the system prompt for the bot."""
        prompt = f"""Create a comprehensive system prompt for a chatbot with these specifications:

Original Persona: {persona_prompt}

Key Characteristics:
{characteristics}

Communication Style:
{comm_style}

Behavioral Guidelines:
{guidelines}

Generate a complete, well-structured system prompt."""
        response = self.llm([HumanMessage(content=prompt)])
        return response.content
    
    def _synthesize_bot_config(self, characteristics: str, comm_style: str, guidelines: str, system_prompt: str) -> str:
        """Synthesize final bot configuration from all components."""
        synthesis = f"""# Bot Configuration

## System Prompt
{system_prompt}

## Key Characteristics
{characteristics}

## Communication Style
{comm_style}

## Behavioral Guidelines
{guidelines}

## Example Use Cases
This bot is suitable for interactions that require these characteristics and style.
"""
        return synthesis
    
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
