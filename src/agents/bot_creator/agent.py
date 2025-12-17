"""Bot Creator Agent with two persona modes for comparative experiments."""

import re
import asyncio
from typing import Dict, Optional, Any, List, Literal, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from ...utils.logger import get_logger
from ...config.settings import settings, ExecutionMode, HistoryMode
from ...utils.llm_client import llm_manager
from ...utils.smart_memory import SmartMemory

logger = get_logger(__name__)

#: Persona mode type - determines how persona is passed to LLM.
#: - "system_prompt": Persona embedded in system message (stricter control)
#: - "user_instruction": Persona in user message (more flexibility)
PersonaMode = Literal["system_prompt", "user_instruction"]

# Section markers used in bot configuration
BOT_CONFIG_SECTION_MARKERS = [
    "## System Prompt",
    "## Key Characteristics",
    "## Communication Style",
    "## Behavioral Guidelines",
    "## Behavioral",
    "## Example Use Cases"
]


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
        model_name: Optional[str] = None,
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
            model_name: Name of the LLM model to use (defaults to settings.openai_model)
            api_key: OpenAI API key or compatible API key
            api_base: Base URL for the API
            temperature: Temperature for LLM responses
            vector_store: Optional vector store for memory
            proxy: Optional HTTP proxy for API requests
            persona_mode: Mode for handling persona - 'system_prompt' or 'user_instruction'
        """
        model_name = model_name or settings.openai_model
        logger.info(f"Initializing BotCreatorAgent: model={model_name}, temperature={temperature}, persona_mode={persona_mode}")
        
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
        self.created_bots: List[Dict[str, Any]] = []
        self.persona_mode = persona_mode
        
        # Initialize smart memory if enabled
        self.smart_memory = SmartMemory(llm=self.llm, vector_store=vector_store) if settings.smart_memory_enabled else None
        
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
        
        response = self.llm.invoke([HumanMessage(content=prompt_text)])
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
        
        response = self.llm.invoke([HumanMessage(content=enhanced_instruction)])
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
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _determine_communication_style(self, persona_prompt: str, characteristics: str) -> str:
        """Determine appropriate communication style."""
        prompt = f"""Based on this persona and characteristics, describe the communication style this bot should use:

Persona: {persona_prompt}

Characteristics: {characteristics}

Provide specific communication style guidelines."""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _define_behavioral_guidelines(self, persona_prompt: str, characteristics: str) -> str:
        """Define behavioral guidelines and constraints."""
        prompt = f"""Define behavioral guidelines and constraints for a bot with this profile:

Persona: {persona_prompt}

Characteristics: {characteristics}

List specific behavioral rules and constraints."""
        response = self.llm.invoke([HumanMessage(content=prompt)])
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
        response = self.llm.invoke([HumanMessage(content=prompt)])
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
    
    def chat_with_bot(
        self,
        bot_id: str,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        history_mode: Optional[HistoryMode] = None
    ) -> Dict[str, Any]:
        """
        Chat with a created bot using its configured persona.
        
        Args:
            bot_id: The ID of the bot to chat with
            user_message: The user's message to the bot
            conversation_history: Optional previous conversation history
                                 List of {"role": "user"|"assistant", "content": str}
            history_mode: History mode - 'full' (include history) or 'none' (stateless)
                         If None, uses default from settings
        
        Returns:
            Dictionary containing:
                - bot_id: The bot's ID
                - bot_name: The bot's name
                - response: The bot's response
                - conversation_history: Updated conversation history (empty if history_mode='none')
                - history_mode: The history mode used
        """
        # Use default history mode if not specified
        if history_mode is None:
            history_mode = settings.default_history_mode
        
        logger.info(f"Chat request for bot {bot_id}: message length={len(user_message)}, history_mode={history_mode}")
        
        # Find the bot
        bot = self.get_bot(bot_id)
        if not bot:
            logger.warning(f"Bot not found: {bot_id}")
            return {
                "error": f"Bot '{bot_id}' not found",
                "bot_id": bot_id
            }
        
        if not user_message or not user_message.strip():
            logger.warning("Empty user message provided")
            return {
                "error": "User message cannot be empty",
                "bot_id": bot_id
            }
        
        try:
            # Build the system prompt from bot configuration
            system_prompt = self._extract_system_prompt_from_config(bot["bot_configuration"])
            
            # Build messages
            from langchain_core.messages import SystemMessage, AIMessage
            
            messages = [SystemMessage(content=system_prompt)]
            
            # Add conversation history only if history_mode is 'full'
            if history_mode == "full" and conversation_history:
                logger.debug(f"Including {len(conversation_history)} history entries")
                for entry in conversation_history:
                    role = entry.get("role", "")
                    content = entry.get("content", "")
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))
            elif history_mode == "none":
                logger.debug("History mode is 'none', skipping conversation history")
            
            # Add current user message
            messages.append(HumanMessage(content=user_message))
            
            logger.debug(f"Calling LLM for bot chat with {len(messages)} messages")
            
            # Call LLM
            response = self.llm.invoke(messages)
            bot_response = response.content
            
            logger.info(f"Bot {bot_id} responded with {len(bot_response)} characters")
            
            # Build updated conversation history based on mode
            if history_mode == "full":
                updated_history = list(conversation_history) if conversation_history else []
                updated_history.append({"role": "user", "content": user_message})
                updated_history.append({"role": "assistant", "content": bot_response})
            else:
                # In 'none' mode, don't maintain history
                updated_history = []
            
            # Smart memory: detect and store important information
            if self.smart_memory and self.smart_memory.should_store_message(user_message):
                try:
                    # Note: Using sync version for now, can be made async if needed
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If loop is running, schedule as task (fire and forget)
                        asyncio.create_task(self.smart_memory.analyze_and_store(
                            user_message, 
                            bot_response,
                            {"bot_id": bot_id, "bot_name": bot["bot_name"]}
                        ))
                    else:
                        # If no loop, run synchronously
                        loop.run_until_complete(self.smart_memory.analyze_and_store(
                            user_message,
                            bot_response,
                            {"bot_id": bot_id, "bot_name": bot["bot_name"]}
                        ))
                except Exception as e:
                    logger.warning(f"Smart memory storage failed: {e}")
            
            return {
                "bot_id": bot_id,
                "bot_name": bot["bot_name"],
                "response": bot_response,
                "conversation_history": updated_history,
                "history_mode": history_mode
            }
            
        except Exception as e:
            logger.error(f"Failed to chat with bot {bot_id}: {e}", exc_info=True)
            return {
                "error": f"Failed to chat with bot: {str(e)}",
                "bot_id": bot_id
            }
    
    def _extract_system_prompt_from_config(self, bot_configuration: str) -> str:
        """
        Extract the system prompt from bot configuration.
        
        The bot configuration may contain multiple sections. This method
        uses regex patterns to robustly extract the system prompt section
        or uses the full configuration if no clear section is found.
        
        Args:
            bot_configuration: The full bot configuration string
        
        Returns:
            The system prompt for the bot
        """
        max_length = 4000
        
        # Strategy 1: Use regex to find System Prompt section
        # Match "## System Prompt" or "# System Prompt" with various spacing
        system_prompt_pattern = r'#{1,2}\s*System\s*Prompt\s*\n([\s\S]*?)(?=\n#{1,2}\s|\Z)'
        match = re.search(system_prompt_pattern, bot_configuration, re.IGNORECASE)
        if match:
            system_prompt = match.group(1).strip()
            if system_prompt:
                if len(system_prompt) > max_length:
                    return system_prompt[:max_length] + "..."
                return system_prompt
        
        # Strategy 2: Try to extract from markdown section using constant markers
        primary_marker = BOT_CONFIG_SECTION_MARKERS[0]  # "## System Prompt"
        if primary_marker in bot_configuration:
            start_idx = bot_configuration.find(primary_marker)
            remaining = bot_configuration[start_idx + len(primary_marker):]
            
            # Find the nearest following section marker
            nearest_marker_idx = len(remaining)
            for marker in BOT_CONFIG_SECTION_MARKERS[1:]:
                marker_idx = remaining.find(marker)
                if 0 < marker_idx < nearest_marker_idx:
                    nearest_marker_idx = marker_idx
            
            # Also check for generic ## header
            generic_header_idx = remaining.find("\n##")
            if 0 < generic_header_idx < nearest_marker_idx:
                nearest_marker_idx = generic_header_idx
            
            system_prompt = remaining[:nearest_marker_idx].strip()
            if system_prompt:
                if len(system_prompt) > max_length:
                    return system_prompt[:max_length] + "..."
                return system_prompt
        
        # Fallback: use the entire configuration as system prompt
        # Truncate if too long
        if len(bot_configuration) > max_length:
            return bot_configuration[:max_length] + "..."
        return bot_configuration
    
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
