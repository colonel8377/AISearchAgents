"""
Tests for Multi-Agent Debate System.

Tests schemas, service logic, and API endpoints.
"""

import pytest
from uuid import UUID
from src.debate.schemas import PersonaConfig, AgentMetadata, InitRequest, InteractRequest, VoteResponse
from src.debate.service import (
    DebateService,
    generate_system_prompt,
    get_few_shot_example,
    generate_default_personas,
    FEW_SHOT_EXAMPLES
)


class TestSchemas:
    """Test Pydantic schemas."""
    
    def test_persona_config(self):
        """Test PersonaConfig schema."""
        persona = PersonaConfig(
            name="Skeptic",
            description="A critical thinker",
            style="Critical"
        )
        assert persona.name == "Skeptic"
        assert persona.style == "Critical"
    
    def test_agent_metadata(self):
        """Test AgentMetadata schema."""
        from uuid import uuid4
        agent_id = uuid4()
        
        metadata = AgentMetadata(
            agent_id=agent_id,
            role_name="Skeptic",
            system_prompt="You are a skeptic",
            few_shot_example="Example output"
        )
        assert metadata.agent_id == agent_id
        assert metadata.role_name == "Skeptic"
    
    def test_init_request(self):
        """Test InitRequest schema."""
        request = InitRequest(
            topic="Climate Change",
            custom_personas=[],
            auto_agent_count=3
        )
        assert request.topic == "Climate Change"
        assert request.auto_agent_count == 3
    
    def test_vote_response(self):
        """Test VoteResponse schema."""
        from uuid import uuid4
        agent_id = uuid4()
        
        response = VoteResponse(
            agent_id=agent_id,
            verdict=1,
            reasoning="This is my reasoning"
        )
        assert response.verdict == 1
        assert "reasoning" in response.reasoning


class TestServiceHelpers:
    """Test helper functions in service module."""
    
    def test_generate_system_prompt_skeptic(self):
        """Test system prompt generation for skeptical persona."""
        persona = PersonaConfig(
            name="Dr. Skeptic",
            description="Critical analyst",
            style="Skeptical"
        )
        prompt = generate_system_prompt(persona)
        
        assert "Dr. Skeptic" in prompt
        assert "Critical analyst" in prompt
        assert "critically evaluate" in prompt.lower()
    
    def test_generate_system_prompt_supportive(self):
        """Test system prompt generation for supportive persona."""
        persona = PersonaConfig(
            name="Ms. Optimist",
            description="Positive thinker",
            style="Supportive"
        )
        prompt = generate_system_prompt(persona)
        
        assert "Ms. Optimist" in prompt
        assert "Positive thinker" in prompt
        assert "strengths" in prompt.lower() or "support" in prompt.lower()
    
    def test_generate_system_prompt_neutral(self):
        """Test system prompt generation for neutral persona."""
        persona = PersonaConfig(
            name="Judge",
            description="Balanced analyst",
            style="Neutral"
        )
        prompt = generate_system_prompt(persona)
        
        assert "Judge" in prompt
        assert "balanced" in prompt.lower()
    
    def test_get_few_shot_example_critical(self):
        """Test few-shot example for critical style."""
        persona = PersonaConfig(
            name="Critic",
            description="Critical thinker",
            style="Critical"
        )
        example = get_few_shot_example(persona)
        
        assert "verdict" in example
        assert "reasoning" in example
        assert example == FEW_SHOT_EXAMPLES["Critical"]
    
    def test_get_few_shot_example_supportive(self):
        """Test few-shot example for supportive style."""
        persona = PersonaConfig(
            name="Supporter",
            description="Supportive analyst",
            style="Supportive"
        )
        example = get_few_shot_example(persona)
        
        assert example == FEW_SHOT_EXAMPLES["Supportive"]
    
    def test_generate_default_personas(self):
        """Test auto-generation of default personas."""
        personas = generate_default_personas(5)
        
        assert len(personas) == 5
        assert all(isinstance(p, PersonaConfig) for p in personas)
        
        # Check that different styles are used
        styles = [p.style for p in personas]
        assert "Critical" in styles
        assert "Neutral" in styles
        assert "Supportive" in styles


class TestDebateService:
    """Test DebateService class."""
    
    def test_create_session(self):
        """Test creating a debate session."""
        service = DebateService()
        
        personas = [
            PersonaConfig(name="Agent1", description="First agent", style="Critical"),
            PersonaConfig(name="Agent2", description="Second agent", style="Supportive")
        ]
        
        session_id, agents = service.create_session("Test Topic", personas)
        
        assert session_id is not None
        assert len(agents) == 2
        assert all(isinstance(a, AgentMetadata) for a in agents)
        assert agents[0].role_name == "Agent1"
        assert agents[1].role_name == "Agent2"
    
    def test_get_session(self):
        """Test retrieving session data."""
        service = DebateService()
        
        personas = [PersonaConfig(name="Agent1", description="Test", style="Neutral")]
        session_id, _ = service.create_session("Test", personas)
        
        session = service.get_session(session_id)
        
        assert session is not None
        assert session["topic"] == "Test"
        assert "agents" in session
        assert "vote_history" in session
    
    def test_get_agent_metadata(self):
        """Test retrieving agent metadata."""
        service = DebateService()
        
        personas = [PersonaConfig(name="Agent1", description="Test", style="Neutral")]
        session_id, agents = service.create_session("Test", personas)
        
        agent_id = agents[0].agent_id
        metadata = service.get_agent_metadata(session_id, agent_id)
        
        assert metadata is not None
        assert metadata.agent_id == agent_id
        assert metadata.role_name == "Agent1"
    
    def test_add_vote_round(self):
        """Test adding vote rounds."""
        service = DebateService()
        
        personas = [PersonaConfig(name="A1", description="T", style="Neutral")]
        session_id, _ = service.create_session("Test", personas)
        
        result = service.add_vote_round(session_id, [1, 2, 1])
        assert result is True
        
        session = service.get_session(session_id)
        assert len(session["vote_history"]) == 1
        assert session["vote_history"][0] == [1, 2, 1]
    
    def test_calculate_stability_not_enough_rounds(self):
        """Test stability check with insufficient rounds."""
        service = DebateService()
        
        personas = [PersonaConfig(name="A1", description="T", style="Neutral")]
        session_id, _ = service.create_session("Test", personas)
        
        service.add_vote_round(session_id, [1, 1, 1])
        service.add_vote_round(session_id, [1, 1, 1])
        
        # Need at least 3 rounds
        assert service.calculate_stability(session_id) is False
    
    def test_calculate_stability_stable(self):
        """Test stability check when votes are stable."""
        service = DebateService()
        
        personas = [PersonaConfig(name="A1", description="T", style="Neutral")]
        session_id, _ = service.create_session("Test", personas)
        
        # Add three rounds with identical or very similar distributions
        service.add_vote_round(session_id, [1, 1, 2, 2])
        service.add_vote_round(session_id, [1, 1, 2, 2])
        service.add_vote_round(session_id, [1, 1, 2, 2])
        
        # Should be stable (identical distributions)
        assert service.calculate_stability(session_id) == True
    
    def test_calculate_stability_unstable(self):
        """Test stability check when votes are changing."""
        service = DebateService()
        
        personas = [PersonaConfig(name="A1", description="T", style="Neutral")]
        session_id, _ = service.create_session("Test", personas)
        
        # Add three rounds with different distributions
        service.add_vote_round(session_id, [1, 1, 1, 1])
        service.add_vote_round(session_id, [2, 2, 2, 2])
        service.add_vote_round(session_id, [3, 3, 3, 3])
        
        # Should be unstable (changing distributions)
        assert service.calculate_stability(session_id) == False
    
    @pytest.mark.asyncio
    async def test_call_llm_placeholder(self):
        """Test LLM call with placeholder."""
        service = DebateService()
        
        response = await service.call_llm("System prompt", "User message")
        
        # Should return placeholder response
        assert "placeholder" in response.lower() or "verdict" in response.lower()
    
    @pytest.mark.asyncio
    async def test_call_llm_with_custom_caller(self):
        """Test LLM call with custom caller."""
        service = DebateService()
        
        async def mock_llm_caller(system_prompt, user_message):
            return f"Response to: {user_message[:20]}"
        
        service.set_llm_caller(mock_llm_caller)
        
        response = await service.call_llm("System", "Test message")
        assert "Test message" in response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
