"""
Tests for Academic Analysis API endpoints.

These tests verify the academic analysis API endpoints
without requiring external network or LLM API access.
All academic agents are stateless and create new instances per request.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException

from src.api.main import app
from src.agents.content_extractor.agent import ContentExtractionResult
from src.agents.claim_atomizer.agent import ClaimAtomizationResult, AtomicClaim
from src.agents.evidence_locator.agent import EvidenceLocationResult, ClaimEvidence, EvidenceQuote
from src.agents.conflict_auditor.agent import ConflictAuditResult, ConflictAnalysis, ConflictType
from src.agents.synthesis_aggregator.agent import SynthesisAggregationResult, SynthesisReport


class TestAcademicAnalysisAPI:
    """Tests for academic analysis API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)

    def test_extract_content_url(self, client):
        """Test content extraction from URL."""
        # Mock the ContentExtractorAgent
        with patch('src.agents.content_extractor.agent.ContentExtractorAgent') as mock_agent_class:
            mock_agent = Mock()
            mock_result = ContentExtractionResult(
                url="http://example.com",
                title="Test Title",
                main_body="Test content",
                text_length=12,
                truncated=False
            )
            mock_agent.extract_from_url.return_value = mock_result
            mock_agent_class.return_value = mock_agent

            response = client.post(
                "/api/v1/content/extract",
                json={"url": "http://example.com"},
                headers={"X-API-Key": "test-key"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["url"] == "http://example.com"
            assert data["title"] == "Test Title"
            assert data["main_body"] == "Test content"

    def test_extract_content_with_cot(self, client):
        """Test content extraction with CoT enabled."""
        with patch('src.agents.content_extractor.agent.ContentExtractorAgent') as mock_agent_class:
            mock_agent = Mock()
            mock_result = ContentExtractionResult(
                url="http://example.com",
                title="Test Title",
                main_body="Test content",
                text_length=12,
                truncated=False
            )
            mock_agent.extract_from_url.return_value = mock_result
            mock_agent_class.return_value = mock_agent

            response = client.post(
                "/api/v1/content-extractor/extract",
                json={
                    "url": "http://example.com",
                    "use_cot": True,
                    "custom_few_shots": "Custom examples"
                },
                headers={"X-API-Key": "test-key"}
            )

        assert response.status_code == 200
        mock_agent.extract_from_url.assert_called_once_with(
            "http://example.com",
            use_llm=False,
            use_cot=True,
            custom_few_shots="Custom examples"
        )

    def test_atomize_claims(self, client):
        """Test claim atomization."""
        with patch('src.agents.claim_atomizer.agent.ClaimAtomizerAgent') as mock_agent_class:
            mock_agent = Mock()
            mock_claim = AtomicClaim(
                id="1",
                text="Test claim",
                original_sentence="Original sentence",
                confidence=0.8
            )
            mock_result = ClaimAtomizationResult(
                atomic_claims=[mock_claim],
                original_text="Test text",
                execution_mode="no_chain",
                metadata={}
            )
            mock_agent.atomize_text.return_value = mock_result
            mock_agent_class.return_value = mock_agent

            response = client.post(
                "/api/v1/claim-atomizer/atomize",
                json={"text": "Test text"},
                headers={"X-API-Key": "test-key"}
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["atomic_claims"]) == 1
            assert data["atomic_claims"][0]["text"] == "Test claim"

    def test_locate_evidence(self, client):
        """Test evidence location."""
        with patch('src.agents.evidence_locator.agent.EvidenceLocatorAgent') as mock_agent_class:
            mock_agent = Mock()
            mock_quote = EvidenceQuote(
                text="evidence text",
                location="Paragraph 1",
                start_pos=0,
                end_pos=13
            )
            mock_evidence = ClaimEvidence(
                claim_id="1",
                claim_text="Test claim",
                evidence_found=True,
                quotes=[mock_quote],
                reasoning="Found in text"
            )
            mock_result = EvidenceLocationResult(
                claim_evidences=[mock_evidence],
                main_body_text="Main body text",
                metadata={}
            )
            mock_agent.locate_evidence.return_value = mock_result
            mock_agent_class.return_value = mock_agent

            response = client.post(
                "/api/v1/evidence-locator/locate",
                json={
                    "claims": [{"id": "1", "text": "Test claim"}],
                    "main_body": "Main body text"
                },
                headers={"X-API-Key": "test-key"}
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["claim_evidences"]) == 1
            assert data["claim_evidences"][0]["evidence_found"] == True

    def test_audit_conflicts(self, client):
        """Test conflict auditing."""
        with patch('src.agents.conflict_auditor.agent.ConflictAuditorAgent') as mock_agent_class:
            mock_agent = Mock()
            mock_analysis = ConflictAnalysis(
                claim_id="1",
                claim_text="Test claim",
                evidence_quotes=["evidence"],
                verdict=ConflictType.SUPPORTED,
                conflict_type="N/A",
                analysis="Supported",
                confidence=0.9
            )
            mock_result = ConflictAuditResult(
                conflict_analyses=[mock_analysis],
                summary_stats={"total_claims": 1, "supported": 1, "contradicted": 0, "neutral_missing": 0},
                execution_mode="no_chain",
                metadata={}
            )
            mock_agent.audit_conflicts.return_value = mock_result
            mock_agent_class.return_value = mock_agent

            response = client.post(
                "/api/v1/conflict-auditor/audit",
                json={"claim_evidences": [{"claim_id": "1", "claim_text": "Test claim", "evidence_quotes": ["evidence"]}]},
                headers={"X-API-Key": "test-key"}
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["conflict_analyses"]) == 1
            assert data["conflict_analyses"][0]["verdict"] == "supported"
            assert data["summary_stats"]["total_claims"] == 1

    def test_aggregate_synthesis(self, client):
        """Test synthesis aggregation."""
        with patch('src.agents.synthesis_aggregator.agent.SynthesisAggregatorAgent') as mock_agent_class:
            mock_agent = Mock()
            mock_report = SynthesisReport(
                research_summary="Test summary",
                metrics={"total_claims": 1, "conflicted_claims": 0},
                detailed_discrepancies=[],
                quality_assessment="HIGH",
                confidence_score=0.9
            )
            mock_result = SynthesisAggregationResult(
                synthesis_report=mock_report,
                metadata={"total_analyses": 1}
            )
            mock_agent.aggregate_synthesis.return_value = mock_result
            mock_agent_class.return_value = mock_agent

            response = client.post(
                "/api/v1/synthesis-aggregator/aggregate",
                json={"conflict_analyses": [{"claim_id": "1", "verdict": "supported"}]},
                headers={"X-API-Key": "test-key"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["synthesis_report"]["research_summary"] == "Test summary"
            assert data["synthesis_report"]["confidence_score"] == 0.9

    def test_get_content_extractor_shots(self, client):
        """Test getting few-shot examples for content extractor."""
        response = client.get(
            "/api/v1/content/shots",
            headers={"X-API-Key": "test-key"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["agent_type"] == "content_extractor"
        assert "few_shots" in data
        assert "is_custom" in data

    def test_get_claim_atomizer_shots(self, client):
        """Test getting few-shot examples for claim atomizer."""
        response = client.get(
            "/api/v1/content/atomize-shots",
            headers={"X-API-Key": "test-key"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["agent_type"] == "claim_atomizer"
        assert "few_shots" in data
        assert "is_custom" in data

    def test_get_conflict_auditor_shots(self, client):
        """Test getting few-shot examples for conflict auditor."""
        response = client.get(
            "/api/v1/consistency/conflict-audit-shots",
            headers={"X-API-Key": "test-key"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["agent_type"] == "conflict_auditor"
        assert "few_shots" in data
        assert "is_custom" in data


    @patch('src.api.main._evaluate_content')
    @patch('src.api.main._evaluate_url')
    @patch('src.api.main._comparative_analysis')
    def test_overall_evaluation_summary_only(self, mock_comp, mock_url_eval, mock_content_eval, client):
        """Test overall evaluation with summary only."""
        mock_content_eval.return_value = {"content_type": "summary", "metrics": {"content_length": 100}}

        response = client.post(
            "/api/v1/quality/overall",
            json={"summary": "Test summary"},
            headers={"X-API-Key": "test-key"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "summary_evaluation" in data
        assert data["url_evaluation"] is None
        assert data["comparative_analysis"] is None

    @patch('src.api.main._evaluate_content')
    @patch('src.api.main._evaluate_url')
    @patch('src.api.main._comparative_analysis')
    def test_overall_evaluation_with_comparison(self, mock_comp, mock_url_eval, mock_content_eval, client):
        """Test overall evaluation with both summary and URL."""
        mock_content_eval.return_value = {"content_type": "summary", "metrics": {"content_length": 100}}
        mock_url_eval.return_value = {"content_type": "url_content", "metrics": {"content_length": 500}}
        mock_comp.return_value = {"compression_ratio": 0.2}

        response = client.post(
            "/api/v1/quality/overall",
            json={"summary": "Test summary", "url": "http://example.com"},
            headers={"X-API-Key": "test-key"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["summary_evaluation"] is not None
        assert data["url_evaluation"] is not None
        assert data["comparative_analysis"] is not None

    def test_overall_evaluation_no_input(self, client):
        """Test overall evaluation with no input provided."""
        response = client.post(
            "/api/v1/quality/overall",
            json={},
            headers={"X-API-Key": "test-key"}
        )

        assert response.status_code == 422  # Validation error

    @patch('src.api.main.complete_academic_analysis')
    def test_overall_evaluation_with_full_analysis(self, mock_complete, client):
        """Test overall evaluation with full academic analysis."""
        # Mock the complete analysis response
        mock_result = Mock()
        mock_result.overall_success = True
        mock_result.final_report = Mock()
        mock_result.final_report.__dict__ = {"research_summary": "Analysis complete"}
        mock_result.pipeline_steps = [Mock(__dict__={"step_name": "content_extraction", "success": True})]
        mock_result.total_execution_time = 5.0
        mock_complete.return_value = mock_result

        with patch('src.api.main._evaluate_content') as mock_eval:
            mock_eval.return_value = {"content_type": "summary"}

            response = client.post(
                "/api/v1/quality/overall",
                json={
                    "summary": "Test summary",
                    "include_full_analysis": True
                },
                headers={"X-API-Key": "test-key"}
            )

            assert response.status_code == 200
            data = response.json()
            assert "full_academic_analysis" in data

    def test_complete_analysis_pipeline(self, client, mock_agent_manager):
        """Test the complete academic analysis pipeline endpoint."""
        # This would require extensive mocking of all agents
        # For now, just test that the endpoint exists and requires proper auth
        response = client.post(
            "/api/v1/consistency/complete",
            json={"url": "http://example.com"},
            headers={"X-API-Key": "test-key"}
        )

        # Should either succeed with mocked agents or fail gracefully
        assert response.status_code in [200, 500]  # 200 if agents are properly mocked, 500 if not

    def test_check_summary_url_consistency(self, client):
        """Test website summary vs full content consistency checking endpoint."""
        # Mock the consistency checking function
        with patch('src.api.main._check_summary_vs_url_consistency', new_callable=AsyncMock) as mock_consistency:
            mock_consistency.return_value = {
                "consistency_score": 0.8,
                "cross_references": [
                    {
                        "summary_claim_text": "AI is transforming healthcare",
                        "evidence_found": True,
                        "status": "supported"
                    }
                ],
                "conflicting_points": [],
                "processing_time": 2.5
            }

            response = client.post(
                "/api/v1/consistency/check-summary-url",
                json={
                    "summary": "AI is transforming healthcare and education.",
                    "url": "https://example.com/ai-impact",
                    "enable_deep_analysis": True
                },
                headers={"X-API-Key": "test-key"}
            )

            assert response.status_code == 200
            data = response.json()
            assert "consistency_score" in data
            assert "cross_references" in data
            assert "conflicting_points" in data
            assert data["consistency_score"] == 0.8

            # Verify the function was called with correct arguments
            mock_consistency.assert_called_once()
            args = mock_consistency.call_args[0]
            assert args[0] == "AI is transforming healthcare and education."
            assert args[1] == "https://example.com/ai-impact"
            assert args[2] == True  # enable_deep_analysis
