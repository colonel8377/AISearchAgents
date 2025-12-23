"""
Tests for Claim Atomizer Agent.

These tests verify the functionality of the ClaimAtomizerAgent
without requiring external LLM API access.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.agents.claim_atomizer.agent import (
    ClaimAtomizerAgent,
    ClaimAtomizationResult,
    AtomicClaim
)


class TestClaimAtomizerAgent:
    """Tests for ClaimAtomizerAgent."""

    @pytest.fixture
    def agent(self):
        """Create a ClaimAtomizerAgent instance for testing."""
        with patch('src.agents.claim_atomizer.agent.llm_manager'):
            agent = ClaimAtomizerAgent(
                model_name="test-model",
                api_key="test-key",
                temperature=0.1
            )
        return agent

    def test_agent_initialization(self, agent):
        """Test agent initialization with default parameters."""
        assert agent.execution_mode == "chain_local"
        assert isinstance(agent.llm, MagicMock)  # Mocked LLM

    @patch('src.agents.claim_atomizer.agent.ChatOpenAI')
    def test_atomize_claims_with_llm(self, mock_chat_class, agent):
        """Test claim atomization with mocked LLM."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "- CLAIM_1: The company reported revenue.\n- CLAIM_2: Revenue increased by 15%."
        mock_llm.invoke.return_value = mock_response
        mock_chat_class.return_value = mock_llm

        claims = agent._atomize_claims_with_llm("test text")

        assert len(claims) == 2
        assert claims[0]['id'] == '1'
        assert claims[0]['text'] == 'The company reported revenue.'
        assert claims[0]['confidence'] == 0.8

    @patch('src.agents.claim_atomizer.agent.ChatOpenAI')
    def test_atomize_claims_with_cot(self, mock_chat_class, agent):
        """Test claim atomization with Chain of Thought."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "- CLAIM_1: The company reported revenue.\n- CLAIM_2: Revenue increased by 15%."
        mock_llm.invoke.return_value = mock_response
        mock_chat_class.return_value = mock_llm

        claims = agent._atomize_claims_with_llm("test text", use_cot=True)

        assert len(claims) == 2
        mock_llm.invoke.assert_called_once()

    def test_atomize_text_basic(self, agent):
        """Test basic text atomization."""
        with patch.object(agent, '_atomize_claims_with_llm') as mock_atomize:
            mock_atomize.return_value = [
                {"id": "1", "text": "Claim 1", "original_sentence": "test text", "confidence": 0.8}
            ]

            result = agent.atomize_text("test text", use_cot=False)

            assert isinstance(result, ClaimAtomizationResult)
            assert len(result.atomic_claims) == 1
            assert result.atomic_claims[0].text == "Claim 1"
            assert result.execution_mode == "no_chain"

    def test_atomize_text_with_cot(self, agent):
        """Test text atomization with CoT enabled."""
        with patch.object(agent, '_atomize_claims_with_llm') as mock_atomize:
            mock_atomize.return_value = [
                {"id": "1", "text": "Claim 1", "original_sentence": "test text", "confidence": 0.8}
            ]

            result = agent.atomize_text("test text", use_cot=True)

            assert result.execution_mode == "chain_online"
            mock_atomize.assert_called_once_with("test text", use_cot=True, custom_few_shots=None)

    def test_atomize_text_with_custom_few_shots(self, agent):
        """Test text atomization with custom few-shot examples."""
        custom_shots = "Custom example here"

        with patch.object(agent, '_atomize_claims_with_llm') as mock_atomize:
            mock_atomize.return_value = [
                {"id": "1", "text": "Claim 1", "original_sentence": "test text", "confidence": 0.8}
            ]

            result = agent.atomize_text("test text", custom_few_shots=custom_shots)

            mock_atomize.assert_called_once_with("test text", use_cot=False, custom_few_shots=custom_shots)

    def test_atomic_claim_creation(self, agent):
        """Test AtomicClaim object creation."""
        claim_data = {
            "id": "1",
            "text": "Test claim",
            "original_sentence": "Original sentence",
            "confidence": 0.9
        }

        with patch.object(agent, '_atomize_claims_with_llm') as mock_atomize:
            mock_atomize.return_value = [claim_data]

            result = agent.atomize_text("test")

            claim = result.atomic_claims[0]
            assert isinstance(claim, AtomicClaim)
            assert claim.id == "1"
            assert claim.text == "Test claim"
            assert claim.confidence == 0.9

    def test_get_default_few_shots(self):
        """Test getting default few-shot examples."""
        few_shots = ClaimAtomizerAgent.get_default_few_shots()

        assert isinstance(few_shots, str)
        assert "Example 1:" in few_shots
        assert "CLAIM_1:" in few_shots
        assert "CLAIM_2:" in few_shots

    def test_reset_agent(self, agent):
        """Test agent reset functionality."""
        # Reset should not raise any errors
        agent.reset()
        # Agent should still be functional after reset
        assert agent.execution_mode == "chain_local"

    @patch('src.agents.claim_atomizer.agent.ChatOpenAI')
    def test_llm_error_handling(self, mock_chat_class, agent):
        """Test handling of LLM errors."""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM Error")
        mock_chat_class.return_value = mock_llm

        claims = agent._atomize_claims_with_llm("test text")

        # Should return empty list on error
        assert claims == []

    def test_empty_text_handling(self, agent):
        """Test handling of empty input text."""
        with patch.object(agent, '_atomize_claims_with_llm') as mock_atomize:
            mock_atomize.return_value = []

            result = agent.atomize_text("")

            assert isinstance(result, ClaimAtomizationResult)
            assert len(result.atomic_claims) == 0
