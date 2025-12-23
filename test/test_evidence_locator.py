"""
Tests for Evidence Locator Agent.

These tests verify the functionality of the EvidenceLocatorAgent
without requiring external LLM API access.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.agents.evidence_locator.agent import (
    EvidenceLocatorAgent,
    EvidenceLocationResult,
    ClaimEvidence,
    EvidenceQuote
)


class TestEvidenceLocatorAgent:
    """Tests for EvidenceLocatorAgent."""

    @pytest.fixture
    def agent(self):
        """Create an EvidenceLocatorAgent instance for testing."""
        with patch('src.agents.evidence_locator.agent.llm_manager'):
            agent = EvidenceLocatorAgent(
                model_name="test-model",
                api_key="test-key",
                temperature=0.1
            )
        return agent

    def test_agent_initialization(self, agent):
        """Test agent initialization with default parameters."""
        assert isinstance(agent.llm, MagicMock)  # Mocked LLM

    def test_split_into_paragraphs(self, agent):
        """Test splitting text into paragraphs."""
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."

        paragraphs = agent._split_into_paragraphs(text)

        assert len(paragraphs) == 3
        assert paragraphs[0][0] == "Paragraph 1."
        assert paragraphs[1][0] == "Paragraph 2."
        assert paragraphs[2][0] == "Paragraph 3."

    def test_verify_quote_in_text_exact_match(self, agent):
        """Test quote verification with exact match."""
        main_body = "This is a test sentence with exact content."
        quote = "test sentence"

        is_valid, reason = agent._verify_quote_in_text(quote, main_body)

        assert is_valid == True
        assert "Exact match found" in reason

    def test_verify_quote_in_text_fuzzy_match(self, agent):
        """Test quote verification with fuzzy match (punctuation differences)."""
        main_body = "This is a test sentence, with punctuation!"
        quote = "This is a test sentence with punctuation"

        is_valid, reason = agent._verify_quote_in_text(quote, main_body)

        assert is_valid == True
        assert "Fuzzy match found" in reason

    def test_verify_quote_in_text_no_match(self, agent):
        """Test quote verification with no match."""
        main_body = "This is different content."
        quote = "nonexistent quote"

        is_valid, reason = agent._verify_quote_in_text(quote, main_body)

        assert is_valid == False
        assert "Insufficient match" in reason

    def test_verify_quote_in_text_partial_match(self, agent):
        """Test quote verification with high partial match."""
        main_body = "The company reported revenue growth quarter over quarter."
        quote = "company reported revenue growth"

        is_valid, reason = agent._verify_quote_in_text(quote, main_body)

        assert is_valid == True
        assert "High partial match" in reason

    def test_verify_quote_empty_input(self, agent):
        """Test quote verification with empty input."""
        is_valid, reason = agent._verify_quote_in_text("", "some content")

        assert is_valid == False
        assert "Empty quote" in reason

    @patch('src.agents.evidence_locator.agent.ChatOpenAI')
    def test_locate_evidence_with_llm(self, mock_chat_class, agent):
        """Test evidence location with mocked LLM."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = """- FOR CLAIM_1:
  - QUOTE: "The company reported revenue"
  - LOCATION: Paragraph 1
- FOR CLAIM_2:
  - QUOTE: "Revenue increased by 15%"
  - LOCATION: Paragraph 2"""
        mock_llm.invoke.return_value = mock_response
        mock_chat_class.return_value = mock_llm

        claims = [
            {"id": "1", "text": "Company reported revenue"},
            {"id": "2", "text": "Revenue increased"}
        ]
        main_body = "The company reported revenue of $100M. Revenue increased by 15% from last year."

        results = agent._find_evidence_with_llm(claims, main_body)

        assert len(results) == 2
        assert results[0]["claim_id"] == "1"
        assert "evidence_found" in results[0]
        assert "quotes" in results[0]

    def test_locate_evidence_without_llm(self, agent):
        """Test evidence location using string matching only."""
        claims = [
            {"id": "1", "text": "company revenue"},
            {"id": "2", "text": "market share"}
        ]
        main_body = "The company reported strong revenue growth. Market conditions improved significantly."

        result = agent.locate_evidence(claims, main_body, use_llm=False)

        assert isinstance(result, EvidenceLocationResult)
        assert len(result.claim_evidences) == 2
        assert result.main_body_text == main_body

    def test_locate_evidence_with_verification(self, agent):
        """Test evidence location with quote verification."""
        claims = [{"id": "1", "text": "company revenue"}]

        # Mock LLM to return a quote that exists
        with patch.object(agent, '_find_evidence_with_llm') as mock_llm:
            mock_llm.return_value = [{
                "claim_id": "1",
                "claim_text": "company revenue",
                "evidence_found": True,
                "quotes": [EvidenceQuote(
                    text="company revenue",
                    location="Paragraph 1",
                    start_pos=4,
                    end_pos=19
                )],
                "reasoning": "Found in text"
            }]

            main_body = "The company revenue increased significantly."
            result = agent.locate_evidence(claims, main_body, use_llm=True)

            assert len(result.claim_evidences) == 1
            assert result.claim_evidences[0].evidence_found == True
            assert len(result.claim_evidences[0].quotes) == 1

    def test_locate_evidence_failed_verification(self, agent):
        """Test evidence location when quote verification fails."""
        claims = [{"id": "1", "text": "nonexistent content"}]

        # Mock LLM to return a quote that doesn't exist
        with patch.object(agent, '_find_evidence_with_llm') as mock_llm:
            mock_llm.return_value = [{
                "claim_id": "1",
                "claim_text": "nonexistent content",
                "evidence_found": True,
                "quotes": [EvidenceQuote(
                    text="this quote does not exist",
                    location="Paragraph 1",
                    start_pos=0,
                    end_pos=25
                )],
                "reasoning": "LLM found it"
            }]

            main_body = "Different content here."
            result = agent.locate_evidence(claims, main_body, use_llm=True)

            assert len(result.claim_evidences) == 1
            # Should be marked as no evidence due to verification failure
            assert result.claim_evidences[0].evidence_found == False
            assert len(result.claim_evidences[0].quotes) == 0

    def test_evidence_quote_creation(self):
        """Test EvidenceQuote object creation."""
        quote = EvidenceQuote(
            text="Sample quote text",
            location="Paragraph 1",
            start_pos=10,
            end_pos=28
        )

        assert quote.text == "Sample quote text"
        assert quote.location == "Paragraph 1"
        assert quote.start_pos == 10
        assert quote.end_pos == 28

    def test_claim_evidence_creation(self):
        """Test ClaimEvidence object creation."""
        quotes = [EvidenceQuote(text="quote", location="para 1", start_pos=0, end_pos=5)]

        evidence = ClaimEvidence(
            claim_id="1",
            claim_text="Test claim",
            evidence_found=True,
            quotes=quotes,
            reasoning="Found in text"
        )

        assert evidence.claim_id == "1"
        assert evidence.evidence_found == True
        assert len(evidence.quotes) == 1
        assert evidence.reasoning == "Found in text"

    def test_reset_agent(self, agent):
        """Test agent reset functionality."""
        # Reset should not raise any errors
        agent.reset()
        # Agent should still be functional after reset
        assert agent.llm is not None

    @patch('src.agents.evidence_locator.agent.ChatOpenAI')
    def test_llm_error_handling(self, mock_chat_class, agent):
        """Test handling of LLM errors."""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM Error")
        mock_chat_class.return_value = mock_llm

        claims = [{"id": "1", "text": "test claim"}]
        results = agent._find_evidence_with_llm(claims, "test body")

        # Should return error results for all claims
        assert len(results) == 1
        assert results[0]["evidence_found"] == False
        assert "audit_failed" in results[0]["reasoning"]

    def test_empty_claims_handling(self, agent):
        """Test handling of empty claims list."""
        result = agent.locate_evidence([], "some content", use_llm=False)

        assert isinstance(result, EvidenceLocationResult)
        assert len(result.claim_evidences) == 0
