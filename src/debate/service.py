"""
Core logic for Multi-Agent Debate System.

Implements agent factory and stability logic based on Adaptive Stability paper.
"""

import json
import re
from typing import Dict, List, Callable, Awaitable, Optional, Tuple, Any
from uuid import UUID, uuid4

import numpy as np
from scipy import stats

from .schemas import PersonaConfig, AgentMetadata
from ..config.settings import ExecutionMode, settings
from ..utils.logger import get_logger
from ..utils.agent_cache import llm_cached
from ..storage import get_database

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
        persona: PersonaConfig with style information and optional corpus

    Returns:
        Specialized system prompt string
    """
    base_prompt = f"You are {persona.name}, a debate participant with the following characteristics: {persona.description}."

    # Add persona-specific background knowledge if corpus is provided
    if persona.corpus and len(persona.corpus) > 0:
        corpus_text = "\n\nYour background knowledge includes:\n" + "\n".join(f"- {statement}" for statement in persona.corpus[:5])
        base_prompt += corpus_text

    style_lower = persona.style.lower() if persona.style else ""

    if "skeptic" in style_lower or "critical" in style_lower:
        return base_prompt + "\n\nYour role is to critically evaluate proposals, identify potential flaws, and ensure rigorous analysis. You ask tough questions and demand evidence for claims."
    elif "support" in style_lower or "optimist" in style_lower or "agreeable" in style_lower:
        return base_prompt + "\n\nYour role is to identify strengths in proposals, build on positive aspects, and support constructive solutions. You focus on possibilities and potential."
    else:  # Neutral or other
        return base_prompt + "\n\nYour role is to provide balanced analysis, weighing both pros and cons objectively. You seek middle ground and comprehensive understanding."


def get_few_shot_example(persona: PersonaConfig) -> str:
    """
    Get the appropriate few-shot example based on persona style and corpus.

    Args:
        persona: PersonaConfig with style information and optional corpus

    Returns:
        Few-shot example string
    """
    # If persona has corpus, try to generate personalized examples
    if persona.corpus and len(persona.corpus) > 0:
        # Use corpus statements as inspiration for few-shot examples
        # Extract statements that might show voting/reasoning patterns
        reasoning_examples = []
        for statement in persona.corpus[:3]:  # Use up to 3 examples
            if len(statement.split()) > 10:  # Only use substantial statements
                # Convert statement to a voting-style example
                style_lower = persona.style.lower() if persona.style else ""
                if "skeptic" in style_lower or "critical" in style_lower:
                    verdict = 0  # Critical style tends toward disagreement
                elif "support" in style_lower or "optimist" in style_lower:
                    verdict = 2  # Supportive style tends toward agreement
                else:
                    verdict = 1  # Neutral style

                reasoning_examples.append(f"""
Example based on your previous statements:
{{
    "verdict": {verdict},
    "reasoning": "{statement[:200]}..."
}}""")

        if reasoning_examples:
            return "\n".join(reasoning_examples)

    # Fallback to default style-based examples
    style_lower = persona.style.lower() if persona.style else ""

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
        execution_mode: ExecutionMode = "chain_local",
        corpus: Optional[List[str]] = None
    ) -> Tuple[List[PersonaConfig], Optional[str]]:
        """
        Generate personas based on topic and context, with optional corpus for style detection.

        Args:
            topic: Debate topic
            context: Additional context for persona generation
            num_agents: Number of agents to create
            execution_mode: Generation mode (chain_online, chain_local, no_chain)
            corpus: Optional user history statements for style detection and few-shot examples

        Returns:
            Tuple of (List of PersonaConfig objects, Optional detected user style)
        """
        logger.info(f"Generating {num_agents} personas for topic '{topic[:50]}...' with mode={execution_mode}")

        # Detect user style from corpus if provided
        detected_style = None
        if corpus and len(corpus) > 0:
            detected_style = await self._detect_user_style(corpus)
            logger.info(f"Detected user style: {detected_style}")

        if not self._llm_caller:
            logger.warning("No LLM caller configured, using default personas")
            return generate_default_personas(num_agents), detected_style

        try:
            if execution_mode == "no_chain":
                personas = await self._generate_no_chain(topic, context, num_agents, corpus)
            elif execution_mode == "chain_online":
                personas = await self._generate_chain_online(topic, context, num_agents, corpus)
            else:  # chain_local
                personas = await self._generate_chain_local(topic, context, num_agents, corpus)

            # Apply style detection to personas if style was detected
            if detected_style:
                personas = self._adjust_personas_for_user_style(personas, detected_style)

            return personas, detected_style

        except Exception as e:
            logger.error(f"Persona generation failed: {e}, falling back to defaults")
            return generate_default_personas(num_agents), detected_style
    
    async def _generate_no_chain(
        self,
        topic: str,
        context: str,
        num_agents: int,
        corpus: Optional[List[str]] = None
    ) -> List[PersonaConfig]:
        """
        No chain mode: Simple prompt-based generation.
        """
        corpus_text = ""
        if corpus and len(corpus) > 0:
            corpus_text = f"""

User's previous statements (use as examples for debate style):
{chr(10).join(f"- {statement}" for statement in corpus[:5])}"""

        prompt = f"""Generate {num_agents} diverse debate personas for the following topic.

Topic: {topic}
{f'Context: {context}' if context else ''}{corpus_text}

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
        num_agents: int,
        corpus: Optional[List[str]] = None
    ) -> List[PersonaConfig]:
        """
        Chain online mode: LLM does all reasoning and generation.
        """
        corpus_text = ""
        if corpus and len(corpus) > 0:
            corpus_text = f"""
User's previous statements (consider their debate style):
{chr(10).join(f"- {statement}" for statement in corpus[:5])}"""

        prompt = f"""You are designing a multi-agent debate system. Think through this step by step:

Topic: {topic}
{f'Context: {context}' if context else ''}{corpus_text}
Number of agents needed: {num_agents}

Step 1: Analyze the topic and identify key perspectives
Step 2: Determine what expertise would be valuable
Step 3: Design personas that will create productive debate (not just agreeing)
Step 4: Consider user's debate style and ensure complementary perspectives
Step 5: Ensure diversity in viewpoints and styles

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
        num_agents: int,
        corpus: Optional[List[str]] = None
    ) -> List[PersonaConfig]:
        """
        Chain local mode: We decompose the task into subtasks.
        """
        # Subtask 1: Analyze the topic
        logger.debug("Subtask 1: Analyzing topic")
        topic_analysis = await self._analyze_topic(topic, context, corpus)

        # Subtask 2: Identify key perspectives
        logger.debug("Subtask 2: Identifying perspectives")
        perspectives = await self._identify_perspectives(topic, topic_analysis, num_agents, corpus)

        # Subtask 3: Generate persona details
        logger.debug("Subtask 3: Generating personas")
        personas = await self._generate_persona_details(topic, perspectives, num_agents, corpus)
        
        return personas
    
    async def _analyze_topic(self, topic: str, context: str, corpus: Optional[List[str]] = None) -> str:
        """Subtask: Analyze the debate topic."""
        corpus_text = ""
        if corpus and len(corpus) > 0:
            corpus_text = f"""

User's previous debate statements (consider their perspective):
{chr(10).join(f"- {statement}" for statement in corpus[:3])}"""

        prompt = f"""Analyze this debate topic and identify:
1. Key stakeholders
2. Main arguments for and against
3. Areas of potential controversy
4. Required expertise for meaningful debate

Topic: {topic}
{f'Context: {context}' if context else ''}{corpus_text}

Provide a brief analysis."""

        system_prompt = "You are a debate topic analyst."
        return await self._llm_caller(system_prompt, prompt)
    
    async def _identify_perspectives(self, topic: str, analysis: str, num_agents: int, corpus: Optional[List[str]] = None) -> str:
        """Subtask: Identify key perspectives needed."""
        corpus_text = ""
        if corpus and len(corpus) > 0:
            corpus_text = f"""

User's previous statements (ensure perspectives complement their style):
{chr(10).join(f"- {statement}" for statement in corpus[:3])}"""

        prompt = f"""Based on this topic and analysis, identify {num_agents} distinct perspectives that should be represented in a debate.

Topic: {topic}
Analysis: {analysis}{corpus_text}

List {num_agents} perspectives with their likely stance (supportive/critical/neutral)."""

        system_prompt = "You are an expert at identifying diverse viewpoints."
        return await self._llm_caller(system_prompt, prompt)
    
    async def _generate_persona_details(
        self,
        topic: str,
        perspectives: str,
        num_agents: int,
        corpus: Optional[List[str]] = None
    ) -> List[PersonaConfig]:
        """Subtask: Generate detailed persona configurations."""
        corpus_text = ""
        if corpus and len(corpus) > 0:
            corpus_text = f"""

User's previous statements (use as examples for debate style):
{chr(10).join(f"- {statement}" for statement in corpus[:3])}"""

        prompt = f"""Create {num_agents} debate personas based on these perspectives:

Topic: {topic}
Perspectives: {perspectives}{corpus_text}

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

    async def _detect_user_style(self, corpus: List[str]) -> Optional[str]:
        """
        Detect user's debate style from their statement corpus.

        Args:
            corpus: List of user's historical statements

        Returns:
            Detected style ("Critical", "Supportive", "Neutral") or None if detection fails
        """
        if not corpus or len(corpus) < 2:
            return None

        try:
            # Sample up to 10 statements to avoid token limits
            sample_corpus = corpus[:10]

            prompt = f"""Analyze the following user statements and determine their debate style.

User statements:
{chr(10).join(f"- {statement}" for statement in sample_corpus)}

Based on these statements, classify the user's debate style as one of:
- Critical: Focuses on identifying flaws, risks, and potential problems
- Supportive: Emphasizes benefits, opportunities, and positive aspects
- Neutral: Balances pros and cons objectively, seeks middle ground

Return only the style name (Critical/Supportive/Neutral) or "Neutral" if unclear."""

            system_prompt = "You are an expert at analyzing debate styles from text."
            response = await self._llm_caller(system_prompt, prompt)

            response_clean = response.strip().lower()
            if "critical" in response_clean:
                return "Critical"
            elif "supportive" in response_clean:
                return "Supportive"
            elif "neutral" in response_clean:
                return "Neutral"
            else:
                return "Neutral"  # Default fallback

        except Exception as e:
            logger.warning(f"Failed to detect user style: {e}")
            return None

    def _adjust_personas_for_user_style(self, personas: List[PersonaConfig], user_style: str) -> List[PersonaConfig]:
        """
        Adjust personas to complement the detected user style.

        Args:
            personas: Original list of personas
            user_style: Detected user style

        Returns:
            Adjusted list of personas
        """
        if user_style == "Critical":
            # If user is critical, add more supportive and neutral personas
            style_preferences = ["Supportive", "Neutral", "Supportive"]
        elif user_style == "Supportive":
            # If user is supportive, add more critical and neutral personas
            style_preferences = ["Critical", "Neutral", "Critical"]
        else:  # Neutral
            # If user is neutral, keep balanced mix
            style_preferences = ["Critical", "Supportive", "Neutral"]

        adjusted_personas = []
        for i, persona in enumerate(personas):
            if persona.style is None:
                # Auto-assign style if not set
                preferred_style = style_preferences[i % len(style_preferences)]
                adjusted_persona = persona.model_copy()
                adjusted_persona.style = preferred_style
                adjusted_personas.append(adjusted_persona)
            else:
                adjusted_personas.append(persona)

        return adjusted_personas


class DebateService:
    """
    Service for managing multi-agent debate sessions.
    
    Handles agent creation, session management, stability checking,
    and comprehensive debate statistics.
    """
    
    def __init__(self, max_rounds: int = DEFAULT_MAX_ROUNDS):
        """Initialize the debate service."""
        # Optional cache for performance (not primary storage)
        self._sessions_cache: Dict[str, Dict] = {}
        self._statistics: Dict[str, DebateStatistics] = {}
        self._llm_caller: Optional[Callable[[str, str], Awaitable[str]]] = None
        self._persona_generator = PersonaGenerator()
        self._max_rounds = max_rounds

        # Initialize storage connection if persistence is enabled
        self._database = get_database() if settings.enable_persistence else None

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
        execution_mode: ExecutionMode = "chain_local",
        corpus: Optional[List[str]] = None
    ) -> Tuple[List[PersonaConfig], Optional[str]]:
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
            execution_mode=execution_mode,
            corpus=corpus
        )
    
    def create_session(
        self,
        topic: str,
        personas: List[PersonaConfig],
        max_rounds: Optional[int] = None,
        corpus: Optional[List[str]] = None,
        detected_user_style: Optional[str] = None
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
                few_shot_example=few_shot_example,
                corpus=persona.corpus
            )
            agents.append(agent_metadata)
            logger.debug(f"Agent created for debate: agent_id={agent_id}, role={persona.name}, style={persona.style}")
        
        # Prepare session data
        session_data = {
            "session_id": session_id,
            "topic": topic,
            "agents": {str(agent.agent_id): agent for agent in agents},
            "vote_history": [],  # List of rounds, each round is a list of votes
            "reasoning_history": [],  # List of rounds, each round is a list of reasonings
            "max_rounds": max_rounds or self._max_rounds,
            "current_round": 0,
            "personas": personas,
            "statistics": {},  # Will be populated by DebateStatistics
            "user_corpus": corpus or [],
            "detected_user_style": detected_user_style,
            "debate_history": [],  # Complete history by rounds
            "current_phase": None,  # "reasoning" or "voting"
            "current_round_reasonings": []  # Current round's reasoning data
        }

        # Save to database if persistence is enabled
        if self._database:
            success = self._database.save_debate_session(session_data)
            if not success:
                logger.warning(f"Failed to persist debate session {session_id}, continuing without persistence")

        # Store in cache for performance
        self._sessions_cache[session_id] = session_data

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
        # Check cache first for performance
        if session_id in self._sessions_cache:
            return self._sessions_cache[session_id]

        # Load from database if persistence is enabled
        if self._database:
            session_data = self._database.load_debate_session(session_id)
            if session_data:
                # Cache for future use
                self._sessions_cache[session_id] = session_data
                return session_data

        return None
    
    def get_agent_metadata(self, session_id: str, agent_id: UUID) -> Optional[AgentMetadata]:
        """
        Get agent metadata from a session.

        Args:
            session_id: Session identifier
            agent_id: Agent identifier

        Returns:
            AgentMetadata or None if not found
        """
        session = self.get_session(session_id)
        if not session:
            return None
        return session["agents"].get(str(agent_id))
    
    @llm_cached(ttl=3600)  # Cache LLM responses for 1 hour
    async def call_llm(self, system_prompt: str, user_message: str) -> str:
        """
        Call the LLM with caching to reduce token consumption.

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
        session = self.get_session(session_id)
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

        # Save updated session to database
        if self._database:
            success = self._database.save_debate_session(session)
            if not success:
                logger.warning(f"Failed to persist updated session {session_id}")

        # Update cache
        self._sessions_cache[session_id] = session

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
        session = self.get_session(session_id)
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
        session = self.get_session(session_id)
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
        session = self.get_session(session_id)
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

    async def execute_round_reasoning(self, session_id: str) -> Dict[str, Any]:
        """
        Execute a reasoning round where all agents provide statements and reasoning.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with round reasoning results
        """
        session = self.get_session(session_id)
        if not session:
            return {"error": f"Session '{session_id}' not found"}

        # Get previous debate context (all previous rounds)
        debate_context = self._build_debate_context(session)

        # Get all agents
        agents = list(session["agents"].values())

        # Execute reasoning for all agents in parallel
        import asyncio
        import time

        reasoning_tasks = []
        for agent in agents:
            task = self._execute_agent_reasoning(agent, session["topic"], debate_context)
            reasoning_tasks.append(task)

        logger.info(f"Executing parallel reasoning for {len(agents)} agents in session {session_id}")
        reasoning_results = await asyncio.gather(*reasoning_tasks, return_exceptions=True)

        # Process results
        round_reasonings = []
        total_word_count = 0
        timestamp = time.time()

        for i, result in enumerate(reasoning_results):
            agent = agents[i]
            if isinstance(result, Exception):
                logger.error(f"Agent {agent.agent_id} reasoning failed: {result}")
                # Create fallback reasoning
                reasoning = {
                    "agent_id": agent.agent_id,
                    "statement": "I need more time to formulate my position on this topic.",
                    "reasoning": f"Due to technical issues, I cannot provide a detailed analysis at this time. My general stance on {session['topic']} requires further consideration.",
                    "timestamp": timestamp,
                    "word_count": 25
                }
            else:
                reasoning = result
                reasoning["agent_id"] = agent.agent_id
                reasoning["timestamp"] = timestamp
                reasoning["word_count"] = len(reasoning["statement"].split()) + len(reasoning["reasoning"].split())

            round_reasonings.append(reasoning)
            total_word_count += reasoning["word_count"]

        # Update session
        session["current_round_reasonings"] = round_reasonings
        session["current_phase"] = "reasoning"

        # Add to debate history
        round_data = {
            "round_number": session["current_round"] + 1,
            "phase": "reasoning",
            "timestamp": timestamp,
            "reasonings": round_reasonings,
            "total_word_count": total_word_count
        }
        session["debate_history"].append(round_data)

        # Save to database
        if self._database:
            success = self._database.save_debate_session(session)
            if not success:
                logger.warning(f"Failed to persist reasoning round for session {session_id}")

        # Update cache
        self._sessions_cache[session_id] = session

        logger.info(f"Completed reasoning round for session {session_id}, {len(round_reasonings)} reasonings collected")

        return {
            "round_number": session["current_round"] + 1,
            "reasonings": round_reasonings,
            "timestamp": timestamp,
            "total_word_count": total_word_count,
            "phase": "reasoning_completed"
        }

    async def execute_round_voting(self, session_id: str) -> Dict[str, Any]:
        """
        Execute a voting round where all agents vote based on the current round's reasoning.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with round voting results
        """
        session = self.get_session(session_id)
        if not session:
            return {"error": f"Session '{session_id}' not found"}

        if session.get("current_phase") != "reasoning":
            return {"error": "Cannot execute voting round: reasoning phase not completed"}

        # Get current round reasonings
        current_reasonings = session.get("current_round_reasonings", [])
        if not current_reasonings:
            return {"error": "No reasoning data available for voting"}

        # Get all agents
        agents = list(session["agents"].values())

        # Build voting context: history + current round statements (current round has higher priority)
        debate_history = self._build_debate_context(session)
        current_round_statements = self._build_voting_context(current_reasonings, session)
        voting_context = f"{debate_history}\n\n=== CURRENT ROUND STATEMENTS (HIGHEST PRIORITY) ===\n{current_round_statements}"

        # Execute voting for all agents in parallel
        import asyncio
        import time

        voting_tasks = []
        for agent in agents:
            task = self._execute_agent_voting(agent, session["topic"], voting_context)
            voting_tasks.append(task)

        logger.info(f"Executing parallel voting for {len(agents)} agents in session {session_id}")
        voting_results = await asyncio.gather(*voting_tasks, return_exceptions=True)

        # Process results
        round_votes = []
        total_word_count = 0
        timestamp = time.time()

        for i, result in enumerate(voting_results):
            agent = agents[i]
            if isinstance(result, Exception):
                logger.error(f"Agent {agent.agent_id} voting failed: {result}")
                # Create fallback vote
                vote = {
                    "agent_id": agent.agent_id,
                    "verdict": 1,  # Neutral fallback
                    "reasoning": f"Due to technical issues, I cannot provide a detailed vote at this time. My assessment of the current round's discussion requires further consideration.",
                    "timestamp": timestamp,
                    "word_count": 30
                }
            else:
                vote = result
                vote["agent_id"] = agent.agent_id
                vote["timestamp"] = timestamp
                vote["word_count"] = len(vote["reasoning"].split())

            round_votes.append(vote)
            total_word_count += vote["word_count"]

        # Extract votes for statistics
        votes = [vote["verdict"] for vote in round_votes]
        reasonings = [vote["reasoning"] for vote in round_votes]

        # Update session
        session["current_round"] += 1
        session["current_phase"] = "voting_completed"
        session["current_round_reasonings"] = []  # Clear for next round

        # Add to debate history
        round_data = {
            "round_number": session["current_round"],
            "phase": "voting",
            "timestamp": timestamp,
            "votes": round_votes,
            "total_word_count": total_word_count,
            "statistics": {}  # Will be populated below
        }
        session["debate_history"].append(round_data)

        # Update statistics
        if self._statistics.get(session_id):
            round_stats = self._statistics[session_id].add_round(
                round_number=session["current_round"],
                votes=votes,
                agent_ids=[str(vote["agent_id"]) for vote in round_votes],
                reasonings=reasonings
            )
            round_data["statistics"] = round_stats

        # Save to database
        if self._database:
            success = self._database.save_debate_session(session)
            if not success:
                logger.warning(f"Failed to persist voting round for session {session_id}")

        # Update cache
        self._sessions_cache[session_id] = session

        logger.info(f"Completed voting round {session['current_round']} for session {session_id}")

        return {
            "round_number": session["current_round"],
            "votes": round_votes,
            "timestamp": timestamp,
            "total_word_count": total_word_count,
            "statistics": round_data["statistics"],
            "phase": "voting_completed"
        }

    async def _execute_agent_reasoning(self, agent: AgentMetadata, topic: str, debate_context: str) -> Dict[str, Any]:
        """
        Execute reasoning for a single agent.

        Args:
            agent: Agent metadata
            topic: Debate topic
            debate_context: Context from previous rounds

        Returns:
            Dictionary with statement and reasoning
        """
        prompt = f"""Topic: {topic}

Previous debate context (you can only see statements from previous rounds):
{debate_context}

As {agent.role_name}, provide your position statement and detailed reasoning.

IMPORTANT: You can see statements from all previous rounds of discussion, but you CANNOT see any statements or reasoning from other agents in the current round. You must form your position independently based only on the previous debate history.

Respond in JSON format:
{{
    "statement": "Your clear position statement (1-2 sentences)",
    "reasoning": "Your detailed reasoning and analysis (3-5 sentences)"
}}"""

        try:
            response = await self.call_llm(agent.system_prompt, prompt)

            # Parse JSON response
            import json
            parsed = json.loads(response)

            return {
                "statement": parsed.get("statement", "I need more time to formulate my position."),
                "reasoning": parsed.get("reasoning", "My analysis requires further consideration.")
            }

        except Exception as e:
            logger.warning(f"Failed to parse reasoning response from agent {agent.agent_id}: {e}")
            return {
                "statement": "I need more time to formulate my position.",
                "reasoning": f"Due to parsing issues, I cannot provide detailed reasoning at this time. My general thoughts on {topic} require further development."
            }

    async def _execute_agent_voting(self, agent: AgentMetadata, topic: str, voting_context: str) -> Dict[str, Any]:
        """
        Execute voting for a single agent.

        Args:
            agent: Agent metadata
            topic: Debate topic
            voting_context: Context from current round reasonings

        Returns:
            Dictionary with verdict and reasoning
        """
        prompt = f"""Topic: {topic}

Debate history and current round statements (you can only see statements, not detailed reasoning):
{voting_context}

As {agent.role_name}, review all the statements above and provide your vote. Consider the strengths and weaknesses of each position.

IMPORTANT NOTES:
- Current round statements have HIGHEST PRIORITY - they represent the latest positions
- You can see statements from all previous rounds and the current round, but you CANNOT see any detailed reasoning or analysis from other agents
- You must evaluate based only on the statements provided, with emphasis on the most recent positions

Vote on a scale of 0-2:
- 0: Strongly disagree with the collective position
- 1: Neutral/mixed feelings
- 2: Strongly agree with the collective position

Respond in JSON format:
{{
    "verdict": 1,
    "reasoning": "Your detailed reasoning for this vote (2-4 sentences)"
}}

{agent.few_shot_example}"""

        try:
            response = await self.call_llm(agent.system_prompt, prompt)

            # Parse JSON response
            import json
            parsed = json.loads(response)

            return {
                "verdict": int(parsed.get("verdict", 1)),
                "reasoning": parsed.get("reasoning", "My vote reflects careful consideration of all positions.")
            }

        except Exception as e:
            logger.warning(f"Failed to parse voting response from agent {agent.agent_id}: {e}")
            return {
                "verdict": 1,
                "reasoning": f"Due to parsing issues, I provide a neutral vote. The discussion on {topic} presents valid points from multiple perspectives."
            }

    def _build_debate_context(self, session: Dict[str, Any]) -> str:
        """
        Build debate context from all previous rounds.

        Args:
            session: Session data

        Returns:
            Formatted context string
        """
        debate_history = session.get("debate_history", [])
        if not debate_history:
            return "This is the first round of debate. No previous context available."

        context_parts = []
        for round_data in debate_history:
            round_num = round_data["round_number"]
            phase = round_data["phase"]

            if phase == "reasoning":
                context_parts.append(f"\n--- Round {round_num} Reasoning Phase ---")
                for reasoning in round_data["reasonings"]:
                    agent_name = "Unknown Agent"
                    # Find agent name
                    agents = session.get("agents", {})
                    if str(reasoning["agent_id"]) in agents:
                        agent_name = agents[str(reasoning["agent_id"])].role_name

                    # Agent只能看到其他agent的statement，不能看到reasoning
                    context_parts.append(f"{agent_name}: {reasoning['statement']}")

            elif phase == "voting":
                context_parts.append(f"\n--- Round {round_num} Voting Phase ---")
                for vote in round_data["votes"]:
                    agent_name = "Unknown Agent"
                    agents = session.get("agents", {})
                    if str(vote["agent_id"]) in agents:
                        agent_name = agents[str(vote["agent_id"])].role_name

                    verdict_text = {0: "disagrees", 1: "is neutral", 2: "agrees"}.get(vote["verdict"], "has mixed feelings")
                    context_parts.append(f"{agent_name} {verdict_text}: {vote['reasoning'][:150]}...")

        return "\n".join(context_parts)

    def _build_voting_context(self, current_reasonings: List[Dict[str, Any]], session: Optional[Dict] = None) -> str:
        """
        Build voting context from current round reasonings.

        Args:
            current_reasonings: Current round's reasoning data
            session: Session data for agent name lookup (optional)

        Returns:
            Formatted context string for voting
        """
        if not current_reasonings:
            return "No statements available for voting."

        context_parts = []
        for reasoning in current_reasonings:
            agent_id = reasoning["agent_id"]

            # Try to get real agent name from session
            agent_name = f"Agent_{str(agent_id)[:8]}"  # Fallback name
            if session and "agents" in session:
                agents = session.get("agents", {})
                if str(agent_id) in agents:
                    agent_name = agents[str(agent_id)].role_name

            context_parts.append(f"\n--- {agent_name} ---")
            # Agent只能看到其他agent的statement，不能看到reasoning
            context_parts.append(f"Statement: {reasoning['statement']}")

        return "\n".join(context_parts)


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
