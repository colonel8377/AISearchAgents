"""
Tests for Synthesis Aggregator Agent.

These tests verify the functionality of the SynthesisAggregatorAgent
without requiring external LLM API access.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.agents.synthesis_aggregator.agent import (
    SynthesisAggregatorAgent,
    SynthesisAggregationResult,
    SynthesisReport
)


class TestSynthesisAggregatorAgent:
    """Tests for SynthesisAggregatorAgent."""

    @pytest.fixture
    def agent(self):
        """Create a SynthesisAggregatorAgent instance for testing."""
        with patch('src.agents.synthesis_aggregator.agent.llm_manager'):
            agent = SynthesisAggregatorAgent(
                model_name="test-model",
                api_key="test-key",
                temperature=0.1
            )
        return agent

    def test_agent_initialization(self, agent):
        """Test agent initialization with default parameters."""
        assert isinstance(agent.llm, MagicMock)  # Mocked LLM

    def test_create_synthesis_report_basic(self, agent):
        """Test basic synthesis report creation."""
        conflict_analyses = [
            {
                "claim_id": "1",
                "claim_text": "Claim 1",
                "evidence_quotes": ["evidence 1"],
                "verdict": "supported",
                "conflict_type": "N/A",
                "analysis": "Supported",
                "confidence": 0.9
            },
            {
                "claim_id": "2",
                "claim_text": "Claim 2",
                "evidence_quotes": ["evidence 2"],
                "verdict": "contradicted",
                "conflict_type": "numerical_discrepancy",
                "analysis": "Numbers don't match",
                "confidence": 0.8
            }
        ]

        report = agent._create_synthesis_report(conflict_analyses)

        assert isinstance(report, SynthesisReport)
        assert report.metrics["total_claims"] == 2
        assert report.metrics["supported_claims"] == 1
        assert report.metrics["conflicted_claims"] == 1
        assert "numerical_discrepancy" in report.metrics["conflict_types"]
        assert len(report.detailed_discrepancies) == 1

    def test_create_synthesis_report_all_supported(self, agent):
        """Test synthesis report when all claims are supported."""
        conflict_analyses = [
            {
                "claim_id": "1",
                "claim_text": "Claim 1",
                "evidence_quotes": ["evidence"],
                "verdict": "supported",
                "conflict_type": "N/A",
                "analysis": "Fully supported",
                "confidence": 0.95
            },
            {
                "claim_id": "2",
                "claim_text": "Claim 2",
                "evidence_quotes": ["evidence"],
                "verdict": "supported",
                "conflict_type": "N/A",
                "analysis": "Also supported",
                "confidence": 0.9
            }
        ]

        report = agent._create_synthesis_report(conflict_analyses)

        assert report.quality_assessment == "HIGH: No factual conflicts detected. Claims are well-supported by evidence."
        assert report.confidence_score == 0.9
        assert len(report.detailed_discrepancies) == 0

    def test_create_synthesis_report_high_conflicts(self, agent):
        """Test synthesis report with high number of conflicts."""
        conflict_analyses = [
            {
                "claim_id": str(i),
                "claim_text": f"Claim {i}",
                "evidence_quotes": ["evidence"],
                "verdict": "contradicted" if i % 2 == 0 else "supported",
                "conflict_type": "numerical_discrepancy" if i % 2 == 0 else "N/A",
                "analysis": f"Analysis {i}",
                "confidence": 0.8
            }
            for i in range(1, 11)  # 10 claims, 5 conflicted
        ]

        report = agent._create_synthesis_report(conflict_analyses)

        assert report.metrics["total_claims"] == 10
        assert report.metrics["conflicted_claims"] == 5
        assert report.quality_assessment == "CRITICAL: Major factual inconsistencies. Source contains substantial inaccuracies."
        assert report.confidence_score == 0.1
        assert len(report.detailed_discrepancies) == 5

    def test_create_synthesis_report_hallucination_rate(self, agent):
        """Test calculation of hallucination rate."""
        conflict_analyses = [
            {
                "claim_id": "1",
                "claim_text": "Claim 1",
                "evidence_quotes": ["evidence"],
                "verdict": "supported",
                "conflict_type": "N/A",
                "analysis": "Supported",
                "confidence": 0.9
            },
            {
                "claim_id": "2",
                "claim_text": "Claim 2",
                "evidence_quotes": [],
                "verdict": "neutral_missing",
                "conflict_type": "missing_evidence",
                "analysis": "No evidence",
                "confidence": 0.5
            },
            {
                "claim_id": "3",
                "claim_text": "Claim 3",
                "evidence_quotes": [],
                "verdict": "neutral_missing",
                "conflict_type": "missing_evidence",
                "analysis": "No evidence",
                "confidence": 0.5
            }
        ]

        report = agent._create_synthesis_report(conflict_analyses)

        # 2 out of 3 claims are neutral/missing, so hallucination rate should be 2/3 ≈ 0.667
        expected_rate = 2/3
        assert abs(report.metrics["hallucination_rate"] - expected_rate) < 0.001

    @patch('src.agents.synthesis_aggregator.agent.ChatOpenAI')
    def test_enhance_report_with_llm(self, mock_chat_class, agent):
        """Test report enhancement with LLM."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Enhanced research summary with detailed analysis and insights."
        mock_llm.invoke.return_value = mock_response
        mock_chat_class.return_value = mock_llm

        basic_report = SynthesisReport(
            research_summary="Basic summary",
            metrics={"total_claims": 5, "conflicted_claims": 2},
            detailed_discrepancies=[],
            quality_assessment="MEDIUM",
            confidence_score=0.7
        )

        conflict_analyses = [
            {"claim_id": "1", "verdict": "supported"},
            {"claim_id": "2", "verdict": "contradicted"}
        ]

        enhanced_report = agent._enhance_report_with_llm(basic_report, conflict_analyses)

        assert enhanced_report.research_summary == "Enhanced research summary with detailed analysis and insights."
        assert enhanced_report.metrics == basic_report.metrics  # Should preserve metrics
        mock_llm.invoke.assert_called_once()

    def test_aggregate_synthesis_basic(self, agent):
        """Test basic synthesis aggregation."""
        conflict_analyses = [
            {
                "claim_id": "1",
                "claim_text": "Test claim",
                "evidence_quotes": ["evidence"],
                "verdict": "supported",
                "conflict_type": "N/A",
                "analysis": "Supported",
                "confidence": 0.9
            }
        ]

        result = agent.aggregate_synthesis(conflict_analyses, use_llm_enhancement=False)

        assert isinstance(result, SynthesisAggregationResult)
        assert isinstance(result.synthesis_report, SynthesisReport)
        assert result.metadata["total_analyses"] == 1
        assert result.metadata["conflicted_claims"] == 0

    @patch('src.agents.synthesis_aggregator.agent.ChatOpenAI')
    def test_aggregate_synthesis_with_enhancement(self, mock_chat_class, agent):
        """Test synthesis aggregation with LLM enhancement."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "LLM-enhanced summary"
        mock_llm.invoke.return_value = mock_response
        mock_chat_class.return_value = mock_llm

        conflict_analyses = [
            {
                "claim_id": "1",
                "claim_text": "Test claim",
                "evidence_quotes": ["evidence"],
                "verdict": "supported",
                "conflict_type": "N/A",
                "analysis": "Supported",
                "confidence": 0.9
            }
        ]

        result = agent.aggregate_synthesis(conflict_analyses, use_llm_enhancement=True)

        assert result.synthesis_report.research_summary == "LLM-enhanced summary"
        assert result.metadata["use_llm_enhancement"] == True

    def test_reset_agent(self, agent):
        """Test agent reset functionality."""
        # Reset should not raise any errors
        agent.reset()
        # Agent should still be functional after reset
        assert agent.llm is not None

    @patch('src.agents.synthesis_aggregator.agent.ChatOpenAI')
    def test_llm_enhancement_error_handling(self, mock_chat_class, agent):
        """Test handling of LLM enhancement errors."""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM Error")
        mock_chat_class.return_value = mock_llm

        basic_report = SynthesisReport(
            research_summary="Basic summary",
            metrics={"total_claims": 1},
            detailed_discrepancies=[],
            quality_assessment="HIGH",
            confidence_score=0.9
        )

        conflict_analyses = [{"claim_id": "1", "verdict": "supported"}]

        # Should return basic report on LLM error
        enhanced_report = agent._enhance_report_with_llm(basic_report, conflict_analyses)

        assert enhanced_report == basic_report

    def test_empty_conflict_analyses_handling(self, agent):
        """Test handling of empty conflict analyses."""
        result = agent.aggregate_synthesis([])

        assert isinstance(result, SynthesisAggregationResult)
        assert result.synthesis_report.metrics["total_claims"] == 0

    def test_research_summary_generation(self, agent):
        """Test automatic research summary generation."""
        # Test with no conflicts
        analyses_no_conflicts = [
            {
                "claim_id": "1",
                "verdict": "supported",
                "claim_text": "Claim 1",
                "evidence_quotes": ["evidence"],
                "conflict_type": "N/A",
                "analysis": "Supported",
                "confidence": 0.9
            }
        ]

        report = agent._create_synthesis_report(analyses_no_conflicts)
        assert "No factual conflicts detected" in report.research_summary

        # Test with conflicts
        analyses_with_conflicts = [
            {
                "claim_id": "1",
                "verdict": "contradicted",
                "claim_text": "Claim 1",
                "evidence_quotes": ["evidence"],
                "conflict_type": "numerical_discrepancy",
                "analysis": "Conflict detected",
                "confidence": 0.8
            }
        ]

        report = agent._create_synthesis_report(analyses_with_conflicts)
        assert "conflicts" in report.research_summary.lower()
        assert "hallucination rate" in report.research_summary.lower()
