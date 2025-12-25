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

    def test_persona_config_with_corpus(self):
        """Test PersonaConfig schema with corpus."""
        corpus = ["I always question assumptions", "Evidence must be strong"]
        persona = PersonaConfig(
            name="Skeptic",
            description="A critical thinker",
            style="Critical",
            corpus=corpus
        )
        assert persona.corpus == corpus
        assert len(persona.corpus) == 2
    
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

    def test_agent_metadata_with_corpus(self):
        """Test AgentMetadata schema with corpus."""
        from uuid import uuid4
        agent_id = uuid4()
        corpus = ["Historical statement 1", "Historical statement 2"]

        metadata = AgentMetadata(
            agent_id=agent_id,
            role_name="Skeptic",
            system_prompt="You are a skeptic",
            few_shot_example="Example output",
            corpus=corpus
        )
        assert metadata.corpus == corpus
        assert len(metadata.corpus) == 2
    
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

    def test_generate_system_prompt_with_corpus(self):
        """Test system prompt generation with corpus background knowledge."""
        corpus = ["I have extensive experience in economics", "I believe in evidence-based policy making"]
        persona = PersonaConfig(
            name="Economist",
            description="Economic analyst",
            style="Critical",
            corpus=corpus
        )
        prompt = generate_system_prompt(persona)

        assert "Economist" in prompt
        assert "Economic analyst" in prompt
        assert "background knowledge includes" in prompt.lower()
        assert "economics" in prompt.lower()
        assert "evidence-based" in prompt.lower()

    def test_get_few_shot_example_with_corpus_critical(self):
        """Test few-shot example generation from corpus for critical persona."""
        corpus = ["I always demand strong evidence for claims", "Questioning assumptions is crucial for good analysis"]
        persona = PersonaConfig(
            name="Critic",
            description="Critical thinker",
            style="Critical",
            corpus=corpus
        )
        example = get_few_shot_example(persona)

        # Should generate personalized examples from corpus
        assert "verdict" in example
        assert "reasoning" in example
        assert "evidence" in example.lower() or "questioning" in example.lower()

    def test_get_few_shot_example_with_corpus_supportive(self):
        """Test few-shot example generation from corpus for supportive persona."""
        corpus = ["Positive changes are always possible", "I focus on opportunities and solutions"]
        persona = PersonaConfig(
            name="Optimist",
            description="Positive thinker",
            style="Supportive",
            corpus=corpus
        )
        example = get_few_shot_example(persona)

        # Should generate personalized examples from corpus
        assert "verdict" in example
        assert "reasoning" in example
        assert "positive" in example.lower() or "opportunities" in example.lower()
    
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

    def test_create_session_with_persona_corpus(self):
        """Test creating a debate session with personas that have individual corpora."""
        service = DebateService()

        personas = [
            PersonaConfig(
                name="Economist",
                description="Economic expert",
                style="Critical",
                corpus=["I specialize in fiscal policy", "Evidence-based economics is crucial"]
            ),
            PersonaConfig(
                name="Sociologist",
                description="Social scientist",
                style="Neutral",
                corpus=["Social impact assessment is my expertise", "Human behavior drives policy outcomes"]
            )
        ]

        session_id, agents = service.create_session("Economic Policy Debate", personas)

        assert session_id is not None
        assert len(agents) == 2
        assert all(isinstance(a, AgentMetadata) for a in agents)

        # Check first agent
        agent1 = agents[0]
        assert agent1.role_name == "Economist"
        assert agent1.corpus == ["I specialize in fiscal policy", "Evidence-based economics is crucial"]
        assert "fiscal policy" in agent1.system_prompt
        assert "evidence-based economics" in agent1.system_prompt

        # Check second agent
        agent2 = agents[1]
        assert agent2.role_name == "Sociologist"
        assert agent2.corpus == ["Social impact assessment is my expertise", "Human behavior drives policy outcomes"]
        assert "social impact" in agent2.system_prompt.lower()
        assert "human behavior" in agent2.system_prompt.lower()
    
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
        # Result is now a dictionary with round_added field
        assert result.get("round_added") is True
        
        session = service.get_session(session_id)
        assert len(session["vote_history"]) == 1
        assert session["vote_history"][0] == [1, 2, 1]
    
    def test_add_vote_round_with_conversion(self):
        """Test adding vote rounds with mixed types (string votes)."""
        service = DebateService()
        
        personas = [PersonaConfig(name="A1", description="T", style="Neutral")]
        session_id, _ = service.create_session("Test", personas)
        
        # Test with mixed types - strings that can be converted to int
        result = service.add_vote_round(session_id, ["1", 2, "3"])
        # Result is now a dictionary with round_added field
        assert result.get("round_added") is True
        
        session = service.get_session(session_id)
        assert session["vote_history"][0] == [1, 2, 3]
        
        # Test with invalid values - should default to 0
        result = service.add_vote_round(session_id, ["invalid", 2, None])
        assert result.get("round_added") is True
        assert session["vote_history"][1] == [0, 2, 0]
    
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
        assert service.calculate_stability(session_id) is True
    
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
        assert service.calculate_stability(session_id) is False
    
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


def test_corpus_functionality():
    """Manual test for new corpus functionality."""
    from src.debate.schemas import PersonaConfig, AgentMetadata
    from src.debate.service import DebateService, generate_system_prompt, get_few_shot_example

    print("Testing PersonaConfig with corpus...")
    corpus = ["I specialize in economics", "Evidence-based analysis is crucial"]
    persona = PersonaConfig(
        name="Economist",
        description="Economic expert",
        style="Critical",
        corpus=corpus
    )
    assert persona.corpus == corpus
    print("✓ PersonaConfig corpus field works")

    print("\nTesting AgentMetadata with corpus...")
    metadata = AgentMetadata(
        agent_id="test-uuid",
        role_name="Economist",
        system_prompt="Test prompt",
        few_shot_example="Test example",
        corpus=corpus
    )
    assert metadata.corpus == corpus
    print("✓ AgentMetadata corpus field works")

    print("\nTesting generate_system_prompt with corpus...")
    prompt = generate_system_prompt(persona)
    assert "background knowledge includes" in prompt.lower()
    assert "economics" in prompt.lower()
    print("✓ System prompt includes corpus knowledge")

    print("\nTesting get_few_shot_example with corpus...")
    example = get_few_shot_example(persona)
    assert "verdict" in example
    assert "reasoning" in example
    print("✓ Few-shot example generated from corpus")

    print("\nTesting DebateService with persona corpus...")
    service = DebateService()
    personas = [
        PersonaConfig(
            name="Economist",
            description="Economic expert",
            style="Critical",
            corpus=["I focus on fiscal policy", "GDP growth is important"]
        ),
        PersonaConfig(
            name="Sociologist",
            description="Social scientist",
            style="Neutral",
            corpus=["Social equity matters", "Community impact is key"]
        )
    ]

    session_id, agents = service.create_session("Test Debate", personas)
    assert len(agents) == 2
    assert agents[0].corpus == ["I focus on fiscal policy", "GDP growth is important"]
    assert agents[1].corpus == ["Social equity matters", "Community impact is key"]

    # Check that system prompts include corpus
    assert "fiscal policy" in agents[0].system_prompt
    assert "social equity" in agents[1].system_prompt.lower()
    print("✓ DebateService correctly handles persona corpora")

    print("\n🎉 All corpus functionality tests passed!")


if __name__ == "__main__":
    test_corpus_functionality()
