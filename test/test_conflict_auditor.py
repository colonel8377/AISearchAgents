"""
Tests for Conflict Auditor Agent.

These tests verify the functionality of the ConflictAuditorAgent
without requiring external LLM API access.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.agents.conflict_auditor.agent import (
    ConflictAuditorAgent,
    ConflictAuditResult,
    ConflictAnalysis,
    ConflictType
)


class TestConflictAuditorAgent:
    """Tests for ConflictAuditorAgent."""

    @pytest.fixture
    def agent(self):
        """Create a ConflictAuditorAgent instance for testing."""
        with patch('src.agents.conflict_auditor.agent.llm_manager'):
            agent = ConflictAuditorAgent(
                model_name="test-model",
                api_key="test-key",
                temperature=0.1
            )
        return agent

    def test_agent_initialization(self, agent):
        """Test agent initialization with default parameters."""
        assert agent.execution_mode == "chain_local"
        assert isinstance(agent.llm, MagicMock)  # Mocked LLM

    def test_conflict_type_enum(self):
        """Test ConflictType enum values."""
        assert ConflictType.SUPPORTED.value == "supported"
        assert ConflictType.CONTRADICTED.value == "contradicted"
        assert ConflictType.NEUTRAL_MISSING.value == "neutral_missing"

    @patch('src.agents.conflict_auditor.agent.ChatOpenAI')
    def test_audit_conflicts_with_llm(self, mock_chat_class, agent):
        """Test conflict auditing with mocked LLM."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = """- CLAIM_ID: 1
  - VERDICT: Supported
  - CONFLICT_TYPE: N/A
  - ANALYSIS: The evidence confirms the claim.
- CLAIM_ID: 2
  - VERDICT: Contradicted
  - CONFLICT_TYPE: Numerical discrepancy
  - ANALYSIS: Numbers don't match."""
        mock_llm.invoke.return_value = mock_response
        mock_chat_class.return_value = mock_llm

        claim_evidences = [
            {"claim_id": "1", "claim_text": "Claim 1", "evidence_quotes": ["evidence 1"]},
            {"claim_id": "2", "claim_text": "Claim 2", "evidence_quotes": ["evidence 2"]}
        ]

        results = agent._audit_conflicts_with_llm(claim_evidences)

        assert len(results) == 2
        assert results[0]["claim_id"] == "1"
        assert results[0]["verdict"] == ConflictType.SUPPORTED
        assert results[0]["conflict_type"] == "N/A"
        assert results[1]["verdict"] == ConflictType.CONTRADICTED
        assert results[1]["conflict_type"] == "Numerical discrepancy"

    @patch('src.agents.conflict_auditor.agent.ChatOpenAI')
    def test_audit_conflicts_with_cot(self, mock_chat_class, agent):
        """Test conflict auditing with Chain of Thought."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = """- CLAIM_ID: 1
  - VERDICT: Supported
  - CONFLICT_TYPE: N/A
  - ANALYSIS: Step-by-step reasoning shows support."""
        mock_llm.invoke.return_value = mock_response
        mock_chat_class.return_value = mock_llm

        claim_evidences = [{"claim_id": "1", "claim_text": "Claim 1", "evidence_quotes": ["evidence"]}]

        results = agent._audit_conflicts_with_llm(claim_evidences, use_cot=True)

        assert len(results) == 1
        mock_llm.invoke.assert_called_once()

    def test_audit_conflicts_basic(self, agent):
        """Test basic conflict auditing."""
        with patch.object(agent, '_audit_conflicts_with_llm') as mock_audit:
            mock_audit.return_value = [
                {
                    "claim_id": "1",
                    "claim_text": "Test claim",
                    "evidence_quotes": ["evidence"],
                    "verdict": ConflictType.SUPPORTED,
                    "conflict_type": "N/A",
                    "analysis": "Evidence supports claim",
                    "confidence": 0.9
                }
            ]

            claim_evidences = [{"claim_id": "1", "claim_text": "Test claim", "evidence_quotes": ["evidence"]}]

            result = agent.audit_conflicts(claim_evidences, use_cot=False)

            assert isinstance(result, ConflictAuditResult)
            assert len(result.conflict_analyses) == 1
            assert result.execution_mode == "no_chain"
            assert result.summary_stats["total_claims"] == 1
            assert result.summary_stats["supported"] == 1

    def test_audit_conflicts_with_statistics(self, agent):
        """Test conflict auditing with comprehensive statistics."""
        with patch.object(agent, '_audit_conflicts_with_llm') as mock_audit:
            mock_audit.return_value = [
                {
                    "claim_id": "1",
                    "claim_text": "Claim 1",
                    "evidence_quotes": ["evidence"],
                    "verdict": ConflictType.SUPPORTED,
                    "conflict_type": "N/A",
                    "analysis": "Supported",
                    "confidence": 0.9
                },
                {
                    "claim_id": "2",
                    "claim_text": "Claim 2",
                    "evidence_quotes": ["evidence"],
                    "verdict": ConflictType.CONTRADICTED,
                    "conflict_type": "numerical_discrepancy",
                    "analysis": "Numbers don't match",
                    "confidence": 0.8
                },
                {
                    "claim_id": "3",
                    "claim_text": "Claim 3",
                    "evidence_quotes": [],
                    "verdict": ConflictType.NEUTRAL_MISSING,
                    "conflict_type": "missing_evidence",
                    "analysis": "No evidence found",
                    "confidence": 0.5
                }
            ]

            claim_evidences = [
                {"claim_id": "1", "claim_text": "Claim 1", "evidence_quotes": ["evidence"]},
                {"claim_id": "2", "claim_text": "Claim 2", "evidence_quotes": ["evidence"]},
                {"claim_id": "3", "claim_text": "Claim 3", "evidence_quotes": []}
            ]

            result = agent.audit_conflicts(claim_evidences)

            # Check summary statistics
            assert result.summary_stats["total_claims"] == 3
            assert result.summary_stats["supported"] == 1
            assert result.summary_stats["contradicted"] == 1
            assert result.summary_stats["neutral_missing"] == 1
            assert "numerical_discrepancy" in result.summary_stats["conflict_types"]
            assert result.summary_stats["conflict_types"]["numerical_discrepancy"] == 1

    def test_conflict_analysis_creation(self, agent):
        """Test ConflictAnalysis object creation."""
        with patch.object(agent, '_audit_conflicts_with_llm') as mock_audit:
            mock_audit.return_value = [{
                "claim_id": "1",
                "claim_text": "Test claim",
                "evidence_quotes": ["evidence"],
                "verdict": ConflictType.CONTRADICTED,
                "conflict_type": "numerical_error",
                "analysis": "Detailed analysis",
                "confidence": 0.85
            }]

            claim_evidences = [{"claim_id": "1", "claim_text": "Test claim", "evidence_quotes": ["evidence"]}]

            result = agent.audit_conflicts(claim_evidences)

            analysis = result.conflict_analyses[0]
            assert isinstance(analysis, ConflictAnalysis)
            assert analysis.claim_id == "1"
            assert analysis.verdict == ConflictType.CONTRADICTED
            assert analysis.conflict_type == "numerical_error"
            assert analysis.confidence == 0.85

    def test_get_default_few_shots(self):
        """Test getting default few-shot examples."""
        few_shots = ConflictAuditorAgent.get_default_few_shots()

        assert isinstance(few_shots, str)
        assert "Example 1" in few_shots
        assert "CLAIM_ID:" in few_shots
        assert "VERDICT:" in few_shots
        assert "ANALYSIS:" in few_shots

    def test_reset_agent(self, agent):
        """Test agent reset functionality."""
        # Reset should not raise any errors
        agent.reset()
        # Agent should still be functional after reset
        assert agent.execution_mode == "chain_local"

    @patch('src.agents.conflict_auditor.agent.ChatOpenAI')
    def test_llm_error_handling(self, mock_chat_class, agent):
        """Test handling of LLM errors."""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM Error")
        mock_chat_class.return_value = mock_llm

        claim_evidences = [{"claim_id": "1", "claim_text": "claim", "evidence_quotes": ["evidence"]}]
        results = agent._audit_conflicts_with_llm(claim_evidences)

        # Should return error results for all claims
        assert len(results) == 1
        assert results[0]["verdict"] == ConflictType.NEUTRAL_MISSING
        assert "audit_failed" in results[0]["analysis"]

    def test_empty_claim_evidences_handling(self, agent):
        """Test handling of empty claim evidences list."""
        result = agent.audit_conflicts([])

        assert isinstance(result, ConflictAuditResult)
        assert len(result.conflict_analyses) == 0
        assert result.summary_stats["total_claims"] == 0

    def test_custom_few_shots_integration(self, agent):
        """Test integration of custom few-shot examples."""
        custom_shots = "Custom few-shot example"

        with patch.object(agent, '_audit_conflicts_with_llm') as mock_audit:
            mock_audit.return_value = [{
                "claim_id": "1",
                "claim_text": "Test claim",
                "evidence_quotes": ["evidence"],
                "verdict": ConflictType.SUPPORTED,
                "conflict_type": "N/A",
                "analysis": "Supported",
                "confidence": 0.9
            }]

            claim_evidences = [{"claim_id": "1", "claim_text": "Test claim", "evidence_quotes": ["evidence"]}]

            result = agent.audit_conflicts(claim_evidences, custom_few_shots=custom_shots)

            mock_audit.assert_called_once_with(claim_evidences, use_cot=False, custom_few_shots=custom_shots)
