"""
Core logic for Multi-Agent Debate System.

Implements agent factory and stability logic based on Adaptive Stability paper.
"""

from typing import Dict, List, Callable, Awaitable, Optional, Tuple
from uuid import UUID, uuid4
import numpy as np
from scipy import stats

from .schemas import PersonaConfig, AgentMetadata


# Few-shot example templates for different styles
FEW_SHOT_EXAMPLES = {
    "Critical": """Example output format:
{
    "verdict": 0,
    "reasoning": "While the proposal has merit, there are several critical flaws that need addressing. The methodology lacks rigor in data validation, and the assumptions made are not sufficiently supported by evidence. We should consider alternative approaches before committing to this direction."
}""",
    
    "Neutral": """Example output format:
{
    "verdict": 1,
    "reasoning": "The proposal presents both advantages and disadvantages. On one hand, it addresses key concerns raised in previous discussions. On the other hand, there are implementation challenges that need careful consideration. A balanced approach weighing all factors is recommended."
}""",
    
    "Supportive": """Example output format:
{
    "verdict": 2,
    "reasoning": "This is an excellent proposal that aligns well with our objectives. The approach is sound, the methodology is robust, and the expected outcomes are highly promising. I strongly support moving forward with this plan as it addresses all major concerns effectively."
}"""
}


def generate_system_prompt(persona: PersonaConfig) -> str:
    """
    Generate a specialized system prompt based on persona.
    
    Args:
        persona: PersonaConfig with style information
        
    Returns:
        Specialized system prompt string
    """
    base_prompt = f"You are {persona.name}, a debate participant with the following characteristics: {persona.description}."
    
    style_lower = persona.style.lower()
    
    if "skeptic" in style_lower or "critical" in style_lower:
        return base_prompt + " Your role is to critically evaluate proposals, identify potential flaws, and ensure rigorous analysis. You ask tough questions and demand evidence for claims."
    elif "support" in style_lower or "optimist" in style_lower or "agreeable" in style_lower:
        return base_prompt + " Your role is to identify strengths in proposals, build on positive aspects, and support constructive solutions. You focus on possibilities and potential."
    else:  # Neutral or other
        return base_prompt + " Your role is to provide balanced analysis, weighing both pros and cons objectively. You seek middle ground and comprehensive understanding."


def get_few_shot_example(persona: PersonaConfig) -> str:
    """
    Get the appropriate few-shot example based on persona style.
    
    Args:
        persona: PersonaConfig with style information
        
    Returns:
        Few-shot example string
    """
    style_lower = persona.style.lower()
    
    if "skeptic" in style_lower or "critical" in style_lower:
        return FEW_SHOT_EXAMPLES["Critical"]
    elif "support" in style_lower or "optimist" in style_lower or "agreeable" in style_lower:
        return FEW_SHOT_EXAMPLES["Supportive"]
    else:
        return FEW_SHOT_EXAMPLES["Neutral"]


class DebateService:
    """
    Service for managing multi-agent debate sessions.
    
    Handles agent creation, session management, and stability checking.
    """
    
    def __init__(self):
        """Initialize the debate service."""
        self._sessions: Dict[str, Dict] = {}
        self._llm_caller: Optional[Callable[[str, str], Awaitable[str]]] = None
    
    def set_llm_caller(self, llm_caller: Callable[[str, str], Awaitable[str]]) -> None:
        """
        Set the LLM caller function for dependency injection.
        
        Args:
            llm_caller: Async function that takes (system_prompt, user_message) and returns response
        """
        self._llm_caller = llm_caller
    
    def create_session(self, topic: str, personas: List[PersonaConfig]) -> Tuple[str, List[AgentMetadata]]:
        """
        Create a new debate session with agents.
        
        Args:
            topic: Topic of the debate
            personas: List of persona configurations
            
        Returns:
            Tuple of (session_id, list of agent metadata)
        """
        session_id = str(uuid4())
        agents = []
        
        for persona in personas:
            agent_id = uuid4()
            system_prompt = generate_system_prompt(persona)
            few_shot_example = get_few_shot_example(persona)
            
            agent_metadata = AgentMetadata(
                agent_id=agent_id,
                role_name=persona.name,
                system_prompt=system_prompt,
                few_shot_example=few_shot_example
            )
            agents.append(agent_metadata)
        
        # Store session data
        self._sessions[session_id] = {
            "topic": topic,
            "agents": {str(agent.agent_id): agent for agent in agents},
            "vote_history": []  # List of rounds, each round is a list of votes
        }
        
        return session_id, agents
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """
        Get session data.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or None if not found
        """
        return self._sessions.get(session_id)
    
    def get_agent_metadata(self, session_id: str, agent_id: UUID) -> Optional[AgentMetadata]:
        """
        Get agent metadata from a session.
        
        Args:
            session_id: Session identifier
            agent_id: Agent identifier
            
        Returns:
            AgentMetadata or None if not found
        """
        session = self._sessions.get(session_id)
        if not session:
            return None
        return session["agents"].get(str(agent_id))
    
    async def call_llm(self, system_prompt: str, user_message: str) -> str:
        """
        Call the LLM (placeholder for actual implementation).
        
        Args:
            system_prompt: System prompt for the LLM
            user_message: User message
            
        Returns:
            LLM response
        """
        if self._llm_caller:
            return await self._llm_caller(system_prompt, user_message)
        
        # Placeholder response for testing
        return '{"verdict": 1, "reasoning": "This is a placeholder response. Please configure the LLM caller."}'
    
    def add_vote_round(self, session_id: str, votes: List[int]) -> bool:
        """
        Add a round of votes to session history.
        
        Args:
            session_id: Session identifier
            votes: List of integer votes for this round
            
        Returns:
            True if successful, False if session not found
        """
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        # Ensure votes are numeric (convert if needed)
        numeric_votes = []
        for vote in votes:
            try:
                numeric_votes.append(int(vote) if not isinstance(vote, int) else vote)
            except (ValueError, TypeError):
                # If conversion fails, use 0 as default
                numeric_votes.append(0)
        
        session["vote_history"].append(numeric_votes)
        return True
    
    def calculate_stability(self, session_id: str) -> bool:
        """
        Calculate if the debate has reached stability.
        
        Uses KS Statistic logic: Compare distribution of current round vs previous round.
        If diff < 0.05 for 2 consecutive rounds, return True.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if stable, False otherwise
        """
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        vote_history = session["vote_history"]
        
        # Need at least 3 rounds to check for 2 consecutive stable rounds
        if len(vote_history) < 3:
            return False
        
        # Check last two transitions
        try:
            # Compare round n-2 vs n-1
            ks_stat_1, _ = stats.ks_2samp(vote_history[-3], vote_history[-2])
            
            # Compare round n-1 vs n
            ks_stat_2, _ = stats.ks_2samp(vote_history[-2], vote_history[-1])
            
            # If both transitions show small difference (< 0.05), we consider it stable
            return ks_stat_1 < 0.05 and ks_stat_2 < 0.05
        except (ValueError, RuntimeWarning) as e:
            # If KS test fails (e.g., empty arrays, invalid data), not stable
            return False


def generate_default_personas(count: int) -> List[PersonaConfig]:
    """
    Generate default personas for auto-generation.
    
    Args:
        count: Number of personas to generate
        
    Returns:
        List of PersonaConfig objects
    """
    styles = ["Critical", "Neutral", "Supportive"]
    personas = []
    
    for i in range(count):
        style = styles[i % len(styles)]
        personas.append(PersonaConfig(
            name=f"Agent_{i+1}_{style}",
            description=f"A debate participant with a {style.lower()} approach to analysis",
            style=style
        ))
    
    return personas
