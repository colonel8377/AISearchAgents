"""
Core logic for Multi-Agent Debate System.

Implements agent factory and stability logic based on Adaptive Stability paper.
"""

import re
from typing import Dict, List, Callable, Awaitable, Optional, Tuple, Any, Literal
from uuid import UUID, uuid4
import json
import numpy as np
from scipy import stats

from .schemas import PersonaConfig, AgentMetadata
from ..utils.logger import get_logger
from ..config.settings import settings, ExecutionMode

logger = get_logger(__name__)


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


# Default debate settings
DEFAULT_MAX_ROUNDS = 10
DEFAULT_STABILITY_THRESHOLD = 0.05
DEFAULT_CONSECUTIVE_STABLE_ROUNDS = 2

# Valid persona styles for debate agents
VALID_PERSONA_STYLES = ["critical", "supportive", "neutral"]


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


class DebateStatistics:
    """
    Statistics tracker for debate sessions.
    
    Tracks vote distributions, convergence metrics, and round-by-round analysis.
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.round_stats: List[Dict[str, Any]] = []
        self.convergence_history: List[float] = []
        self.agent_vote_history: Dict[str, List[int]] = {}
    
    def add_round(
        self,
        round_number: int,
        votes: List[int],
        agent_ids: List[str],
        reasonings: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Add statistics for a new round.
        
        Returns:
            Dictionary with round statistics
        """
        # Calculate vote distribution
        vote_counts = {}
        for vote in votes:
            vote_counts[vote] = vote_counts.get(vote, 0) + 1
        
        # Calculate consensus level (0-1, higher = more agreement)
        if len(votes) > 0:
            majority_count = max(vote_counts.values())
            consensus_level = majority_count / len(votes)
        else:
            consensus_level = 0.0
        
        # Track per-agent votes
        for i, agent_id in enumerate(agent_ids):
            if agent_id not in self.agent_vote_history:
                self.agent_vote_history[agent_id] = []
            self.agent_vote_history[agent_id].append(votes[i] if i < len(votes) else 0)
        
        # Calculate vote variance
        vote_variance = float(np.var(votes)) if len(votes) > 0 else 0.0
        vote_mean = float(np.mean(votes)) if len(votes) > 0 else 0.0
        
        # Build round statistics
        round_stat = {
            "round_number": round_number,
            "votes": votes,
            "vote_distribution": vote_counts,
            "consensus_level": consensus_level,
            "vote_mean": vote_mean,
            "vote_variance": vote_variance,
            "num_agents": len(votes),
            "reasonings": reasonings or []
        }
        
        self.round_stats.append(round_stat)
        
        # Calculate convergence (KS statistic with previous round)
        if len(self.round_stats) >= 2:
            prev_votes = self.round_stats[-2]["votes"]
            try:
                ks_stat, _ = stats.ks_2samp(prev_votes, votes)
                self.convergence_history.append(float(ks_stat))
                round_stat["ks_statistic"] = float(ks_stat)
            except (ValueError, RuntimeWarning):
                round_stat["ks_statistic"] = None
        
        return round_stat
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive debate statistics summary.
        
        Returns:
            Dictionary with full statistics summary
        """
        if not self.round_stats:
            return {
                "total_rounds": 0,
                "converged": False,
                "message": "No rounds recorded"
            }
        
        # Get final votes
        final_round = self.round_stats[-1]
        final_votes = final_round["votes"]
        
        # Determine winner
        vote_counts = final_round["vote_distribution"]
        if vote_counts:
            winner = max(vote_counts.keys(), key=lambda k: vote_counts[k])
            winner_count = vote_counts[winner]
        else:
            winner = None
            winner_count = 0
        
        # Calculate convergence status
        converged, convergence_round = self._detect_convergence()
        
        # Agent agreement analysis
        agent_consistency = self._analyze_agent_consistency()
        
        return {
            "session_id": self.session_id,
            "total_rounds": len(self.round_stats),
            "converged": converged,
            "convergence_round": convergence_round,
            "final_consensus_level": final_round["consensus_level"],
            "final_vote_distribution": vote_counts,
            "winner": winner,
            "winner_vote_count": winner_count,
            "total_agents": final_round["num_agents"],
            "convergence_history": self.convergence_history,
            "agent_analysis": agent_consistency,
            "round_by_round": self.round_stats
        }
    
    def _detect_convergence(self) -> Tuple[bool, Optional[int]]:
        """
        Detect if the debate has converged based on KS statistic history.
        
        Convergence is detected when the last N consecutive rounds have
        KS statistics below the stability threshold.
        
        Returns:
            Tuple of (converged: bool, convergence_round: Optional[int])
        """
        if len(self.convergence_history) < DEFAULT_CONSECUTIVE_STABLE_ROUNDS:
            return False, None
        
        # Check if last N rounds were stable
        recent_ks = self.convergence_history[-DEFAULT_CONSECUTIVE_STABLE_ROUNDS:]
        if not all(ks < DEFAULT_STABILITY_THRESHOLD for ks in recent_ks):
            return False, None
        
        # Find first round where stability started
        convergence_round = None
        for i in range(len(self.convergence_history) - DEFAULT_CONSECUTIVE_STABLE_ROUNDS, -1, -1):
            if self.convergence_history[i] >= DEFAULT_STABILITY_THRESHOLD:
                convergence_round = i + DEFAULT_CONSECUTIVE_STABLE_ROUNDS + 1
                break
        
        if convergence_round is None:
            convergence_round = DEFAULT_CONSECUTIVE_STABLE_ROUNDS + 1
        
        return True, convergence_round
    
    def _analyze_agent_consistency(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze how consistent each agent was throughout the debate.
        
        Returns:
            Dictionary mapping agent_id to consistency metrics
        """
        agent_consistency = {}
        for agent_id, votes in self.agent_vote_history.items():
            if len(votes) >= 2:
                # Calculate how consistent the agent was (lower = more consistent)
                variance = float(np.var(votes))
                agent_consistency[agent_id] = {
                    "vote_variance": variance,
                    "initial_vote": votes[0],
                    "final_vote": votes[-1],
                    "changed_position": votes[0] != votes[-1]
                }
        return agent_consistency
        
        return {
            "session_id": self.session_id,
            "total_rounds": len(self.round_stats),
            "converged": converged,
            "convergence_round": convergence_round,
            "final_consensus_level": final_round["consensus_level"],
            "final_vote_distribution": vote_counts,
            "winner": winner,
            "winner_vote_count": winner_count,
            "total_agents": final_round["num_agents"],
            "convergence_history": self.convergence_history,
            "agent_analysis": agent_consistency,
            "round_by_round": self.round_stats
        }


class PersonaGenerator:
    """
    Generator for debate personas with 3 execution modes.
    
    Modes:
    - chain_online: LLM does all the reasoning and persona generation
    - chain_local: We decompose the task into subtasks locally
    - no_chain: Simple prompt-based generation without chain reasoning
    """
    
    def __init__(self, llm_caller: Optional[Callable[[str, str], Awaitable[str]]] = None):
        self._llm_caller = llm_caller
    
    def set_llm_caller(self, llm_caller: Callable[[str, str], Awaitable[str]]) -> None:
        """Set the LLM caller for persona generation."""
        self._llm_caller = llm_caller
    
    async def generate_personas(
        self,
        topic: str,
        context: str = "",
        num_agents: int = 3,
        execution_mode: ExecutionMode = "chain_local"
    ) -> List[PersonaConfig]:
        """
        Generate personas based on topic and context.
        
        Args:
            topic: Debate topic
            context: Additional context for persona generation
            num_agents: Number of agents to create
            execution_mode: Generation mode (chain_online, chain_local, no_chain)
        
        Returns:
            List of PersonaConfig objects with appropriate personas
        """
        logger.info(f"Generating {num_agents} personas for topic '{topic[:50]}...' with mode={execution_mode}")
        
        if not self._llm_caller:
            logger.warning("No LLM caller configured, using default personas")
            return generate_default_personas(num_agents)
        
        try:
            if execution_mode == "no_chain":
                return await self._generate_no_chain(topic, context, num_agents)
            elif execution_mode == "chain_online":
                return await self._generate_chain_online(topic, context, num_agents)
            else:  # chain_local
                return await self._generate_chain_local(topic, context, num_agents)
        except Exception as e:
            logger.error(f"Persona generation failed: {e}, falling back to defaults")
            return generate_default_personas(num_agents)
    
    async def _generate_no_chain(
        self,
        topic: str,
        context: str,
        num_agents: int
    ) -> List[PersonaConfig]:
        """
        No chain mode: Simple prompt-based generation.
        """
        prompt = f"""Generate {num_agents} diverse debate personas for the following topic.

Topic: {topic}
{f'Context: {context}' if context else ''}

For each persona, provide:
- name: A descriptive name (e.g., "Dr. Economics Expert")
- description: Brief background and expertise
- style: One of "Critical", "Supportive", or "Neutral"

Ensure personas represent different perspectives on the topic.
Return as JSON array:
[{{"name": "...", "description": "...", "style": "..."}}]"""

        system_prompt = "You are an expert at designing diverse debate participants."
        response = await self._llm_caller(system_prompt, prompt)
        
        return self._parse_personas_response(response, num_agents)
    
    async def _generate_chain_online(
        self,
        topic: str,
        context: str,
        num_agents: int
    ) -> List[PersonaConfig]:
        """
        Chain online mode: LLM does all reasoning and generation.
        """
        prompt = f"""You are designing a multi-agent debate system. Think through this step by step:

Topic: {topic}
{f'Context: {context}' if context else ''}
Number of agents needed: {num_agents}

Step 1: Analyze the topic and identify key perspectives
Step 2: Determine what expertise would be valuable
Step 3: Design personas that will create productive debate (not just agreeing)
Step 4: Ensure diversity in viewpoints and styles

After your analysis, provide exactly {num_agents} personas in this JSON format:
[{{"name": "...", "description": "...", "style": "Critical|Supportive|Neutral"}}]

Think through each step before providing the final JSON."""

        system_prompt = "You are an expert debate designer who thinks through problems systematically."
        response = await self._llm_caller(system_prompt, prompt)
        
        return self._parse_personas_response(response, num_agents)
    
    async def _generate_chain_local(
        self,
        topic: str,
        context: str,
        num_agents: int
    ) -> List[PersonaConfig]:
        """
        Chain local mode: We decompose the task into subtasks.
        """
        # Subtask 1: Analyze the topic
        logger.debug("Subtask 1: Analyzing topic")
        topic_analysis = await self._analyze_topic(topic, context)
        
        # Subtask 2: Identify key perspectives
        logger.debug("Subtask 2: Identifying perspectives")
        perspectives = await self._identify_perspectives(topic, topic_analysis, num_agents)
        
        # Subtask 3: Generate persona details
        logger.debug("Subtask 3: Generating personas")
        personas = await self._generate_persona_details(topic, perspectives, num_agents)
        
        return personas
    
    async def _analyze_topic(self, topic: str, context: str) -> str:
        """Subtask: Analyze the debate topic."""
        prompt = f"""Analyze this debate topic and identify:
1. Key stakeholders
2. Main arguments for and against
3. Areas of potential controversy
4. Required expertise for meaningful debate

Topic: {topic}
{f'Context: {context}' if context else ''}

Provide a brief analysis."""

        system_prompt = "You are a debate topic analyst."
        return await self._llm_caller(system_prompt, prompt)
    
    async def _identify_perspectives(self, topic: str, analysis: str, num_agents: int) -> str:
        """Subtask: Identify key perspectives needed."""
        prompt = f"""Based on this topic and analysis, identify {num_agents} distinct perspectives that should be represented in a debate.

Topic: {topic}
Analysis: {analysis}

List {num_agents} perspectives with their likely stance (supportive/critical/neutral)."""

        system_prompt = "You are an expert at identifying diverse viewpoints."
        return await self._llm_caller(system_prompt, prompt)
    
    async def _generate_persona_details(
        self,
        topic: str,
        perspectives: str,
        num_agents: int
    ) -> List[PersonaConfig]:
        """Subtask: Generate detailed persona configurations."""
        prompt = f"""Create {num_agents} debate personas based on these perspectives:

Topic: {topic}
Perspectives: {perspectives}

Generate exactly {num_agents} personas in JSON format:
[{{"name": "...", "description": "...", "style": "Critical|Supportive|Neutral"}}]

Make sure each persona has:
- A meaningful name reflecting their expertise
- A clear description of their background
- An appropriate debate style"""

        system_prompt = "You are a persona designer for debate systems."
        response = await self._llm_caller(system_prompt, prompt)
        
        return self._parse_personas_response(response, num_agents)
    
    def _extract_json_array(self, text: str) -> Optional[str]:
        """
        Extract JSON array from text using multiple strategies.
        
        Handles cases where JSON may be embedded in markdown code blocks
        or surrounded by other text.
        
        Args:
            text: Text that may contain a JSON array
        
        Returns:
            JSON string or None if not found
        """
        # Strategy 1: Try to find JSON in markdown code blocks
        code_block_pattern = r'```(?:json)?\s*(\[[\s\S]*?\])\s*```'
        match = re.search(code_block_pattern, text)
        if match:
            return match.group(1)
        
        # Strategy 2: Find balanced brackets
        # Track bracket depth to handle nested arrays/objects
        start_idx = text.find('[')
        if start_idx == -1:
            return None
        
        depth = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text[start_idx:], start_idx):
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if in_string:
                continue
            
            if char == '[':
                depth += 1
            elif char == ']':
                depth -= 1
                if depth == 0:
                    return text[start_idx:i + 1]
        
        # Fallback: simple extraction (less reliable)
        end_idx = text.rfind(']')
        if end_idx > start_idx:
            return text[start_idx:end_idx + 1]
        
        return None
    
    def _parse_personas_response(self, response: str, expected_count: int) -> List[PersonaConfig]:
        """
        Parse LLM response into PersonaConfig objects.
        
        Uses robust JSON extraction to handle various response formats.
        
        Args:
            response: LLM response text
            expected_count: Expected number of personas
        
        Returns:
            List of PersonaConfig objects
        """
        try:
            # Try to extract JSON array from response
            json_str = self._extract_json_array(response)
            
            if json_str:
                personas_data = json.loads(json_str)
                
                personas = []
                for data in personas_data[:expected_count]:
                    # Validate and normalize style using the constant
                    style = data.get("style", "Neutral")
                    if style.lower() not in VALID_PERSONA_STYLES:
                        style = "Neutral"
                    else:
                        style = style.capitalize()
                    
                    personas.append(PersonaConfig(
                        name=data.get("name", f"Agent_{len(personas)+1}"),
                        description=data.get("description", "A debate participant"),
                        style=style
                    ))
                
                # Fill remaining with defaults if needed
                default_styles = ["Critical", "Neutral", "Supportive"]
                while len(personas) < expected_count:
                    personas.append(PersonaConfig(
                        name=f"Agent_{len(personas)+1}",
                        description="A debate participant",
                        style=default_styles[len(personas) % len(default_styles)]
                    ))
                
                return personas
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse personas from LLM response: {e}")
        
        # Fallback to defaults
        return generate_default_personas(expected_count)


class DebateService:
    """
    Service for managing multi-agent debate sessions.
    
    Handles agent creation, session management, stability checking,
    and comprehensive debate statistics.
    """
    
    def __init__(self, max_rounds: int = DEFAULT_MAX_ROUNDS):
        """Initialize the debate service."""
        self._sessions: Dict[str, Dict] = {}
        self._statistics: Dict[str, DebateStatistics] = {}
        self._llm_caller: Optional[Callable[[str, str], Awaitable[str]]] = None
        self._persona_generator = PersonaGenerator()
        self._max_rounds = max_rounds
        logger.info(f"DebateService initialized with max_rounds={max_rounds}")
    
    def set_llm_caller(self, llm_caller: Callable[[str, str], Awaitable[str]]) -> None:
        """
        Set the LLM caller function for dependency injection.
        
        Args:
            llm_caller: Async function that takes (system_prompt, user_message) and returns response
        """
        self._llm_caller = llm_caller
        self._persona_generator.set_llm_caller(llm_caller)
    
    async def generate_personas_for_topic(
        self,
        topic: str,
        context: str = "",
        num_agents: int = 3,
        execution_mode: ExecutionMode = "chain_local"
    ) -> List[PersonaConfig]:
        """
        Generate appropriate personas for a debate topic.
        
        Uses the configured execution mode to generate personas that are
        tailored to the specific topic and context.
        
        Args:
            topic: Debate topic
            context: Additional context
            num_agents: Number of personas to generate
            execution_mode: chain_online, chain_local, or no_chain
        
        Returns:
            List of PersonaConfig objects
        """
        return await self._persona_generator.generate_personas(
            topic=topic,
            context=context,
            num_agents=num_agents,
            execution_mode=execution_mode
        )
    
    def create_session(
        self,
        topic: str,
        personas: List[PersonaConfig],
        max_rounds: Optional[int] = None
    ) -> Tuple[str, List[AgentMetadata]]:
        """
        Create a new debate session with agents.
        
        Args:
            topic: Topic of the debate
            personas: List of persona configurations
            max_rounds: Maximum rounds for this session (optional)
            
        Returns:
            Tuple of (session_id, list of agent metadata)
        """
        session_id = str(uuid4())
        logger.info(f"Creating debate session: session_id={session_id}, topic='{topic}', num_personas={len(personas)}")
        
        agents = []
        
        for i, persona in enumerate(personas):
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
            logger.debug(f"Agent created for debate: agent_id={agent_id}, role={persona.name}, style={persona.style}")
        
        # Store session data
        self._sessions[session_id] = {
            "topic": topic,
            "agents": {str(agent.agent_id): agent for agent in agents},
            "vote_history": [],  # List of rounds, each round is a list of votes
            "reasoning_history": [],  # List of rounds, each round is a list of reasonings
            "max_rounds": max_rounds or self._max_rounds,
            "current_round": 0,
            "personas": personas
        }
        
        # Initialize statistics tracker
        self._statistics[session_id] = DebateStatistics(session_id)
        
        logger.info(f"Debate session created successfully: session_id={session_id}, num_agents={len(agents)}")
        
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
    
    def add_vote_round(
        self,
        session_id: str,
        votes: List[int],
        reasonings: Optional[List[str]] = None,
        agent_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Add a round of votes to session history with statistics.
        
        Args:
            session_id: Session identifier
            votes: List of integer votes for this round
            reasonings: Optional list of reasoning strings
            agent_ids: Optional list of agent IDs (for tracking)
            
        Returns:
            Dictionary with round statistics and debate status
        """
        session = self._sessions.get(session_id)
        if not session:
            logger.warning(f"Attempted to add votes to non-existent session: {session_id}")
            return {"error": f"Session '{session_id}' not found"}
        
        # Ensure votes are numeric (convert if needed)
        numeric_votes = []
        for vote in votes:
            try:
                numeric_votes.append(int(vote) if not isinstance(vote, int) else vote)
            except (ValueError, TypeError):
                # If conversion fails, use 0 as default
                logger.warning(f"Invalid vote value: {vote}, using 0 as default")
                numeric_votes.append(0)
        
        session["vote_history"].append(numeric_votes)
        session["reasoning_history"].append(reasonings or [])
        session["current_round"] += 1
        
        logger.info(f"Vote round added: session_id={session_id}, round={session['current_round']}, votes={numeric_votes}")
        
        # Update statistics
        stats = self._statistics.get(session_id)
        if stats:
            # Use provided agent_ids or generate from session
            if not agent_ids:
                agent_ids = list(session["agents"].keys())
            
            round_stats = stats.add_round(
                round_number=session["current_round"],
                votes=numeric_votes,
                agent_ids=agent_ids,
                reasonings=reasonings
            )
        else:
            round_stats = {"round_number": session["current_round"], "votes": numeric_votes}
        
        # Check if max rounds reached
        max_rounds_reached = session["current_round"] >= session["max_rounds"]
        
        return {
            "round_added": True,
            "current_round": session["current_round"],
            "max_rounds": session["max_rounds"],
            "max_rounds_reached": max_rounds_reached,
            "round_stats": round_stats
        }
    
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
            logger.warning(f"Stability check on non-existent session: {session_id}")
            return False
        
        vote_history = session["vote_history"]
        
        # Need at least 3 rounds to check for 2 consecutive stable rounds
        if len(vote_history) < 3:
            logger.debug(f"Insufficient rounds for stability check: {len(vote_history)} < 3")
            return False
        
        # Check last two transitions
        try:
            # Compare round n-2 vs n-1
            ks_stat_1, _ = stats.ks_2samp(vote_history[-3], vote_history[-2])
            
            # Compare round n-1 vs n
            ks_stat_2, _ = stats.ks_2samp(vote_history[-2], vote_history[-1])
            
            logger.debug(f"KS statistics: ks_stat_1={ks_stat_1:.4f}, ks_stat_2={ks_stat_2:.4f}")
            
            # If both transitions show small difference (< 0.05), we consider it stable
            is_stable = bool(ks_stat_1 < DEFAULT_STABILITY_THRESHOLD and ks_stat_2 < DEFAULT_STABILITY_THRESHOLD)
            logger.info(f"Stability check result: session_id={session_id}, stable={is_stable}")
            return is_stable
        except (ValueError, RuntimeWarning) as e:
            # If KS test fails (e.g., empty arrays, invalid data), not stable
            logger.warning(f"KS test failed for session {session_id}: {e}")
            return False
    
    def get_debate_statistics(self, session_id: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a debate session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Dictionary with full debate statistics
        """
        session = self._sessions.get(session_id)
        if not session:
            return {"error": f"Session '{session_id}' not found"}
        
        stats = self._statistics.get(session_id)
        if not stats:
            return {
                "session_id": session_id,
                "total_rounds": len(session["vote_history"]),
                "vote_history": session["vote_history"],
                "message": "No detailed statistics available"
            }
        
        summary = stats.get_summary()
        
        # Add session metadata
        summary["topic"] = session["topic"]
        summary["max_rounds"] = session["max_rounds"]
        summary["max_rounds_reached"] = session["current_round"] >= session["max_rounds"]
        
        return summary
    
    def should_continue_debate(self, session_id: str) -> Dict[str, Any]:
        """
        Determine if the debate should continue.
        
        Returns information about whether to continue and why.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Dictionary with continue status and reason
        """
        session = self._sessions.get(session_id)
        if not session:
            return {
                "continue": False,
                "reason": "Session not found"
            }
        
        current_round = session["current_round"]
        max_rounds = session["max_rounds"]
        
        # Check max rounds
        if current_round >= max_rounds:
            return {
                "continue": False,
                "reason": "Maximum rounds reached",
                "current_round": current_round,
                "max_rounds": max_rounds
            }
        
        # Check stability
        if self.calculate_stability(session_id):
            return {
                "continue": False,
                "reason": "Debate has converged (stable)",
                "current_round": current_round
            }
        
        return {
            "continue": True,
            "reason": "Debate ongoing",
            "current_round": current_round,
            "rounds_remaining": max_rounds - current_round
        }


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
