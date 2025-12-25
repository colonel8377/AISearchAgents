"""
Tests for quality/overall and consistency/check-summary-url API endpoints.
Tests the new cot_mode and use_few_shots parameters.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.api.main import CheckConsistencyRequest, OverallEvaluationRequest, CompareClaimsRequest


class TestCheckConsistencyAPI:
    """Tests for /api/v1/consistency/check-summary-url endpoint."""
    
    def test_check_consistency_basic(self):
        """Test basic consistency check without new parameters."""
        with patch('src.api.main._check_summary_vs_url_consistency', new_callable=AsyncMock) as mock_check:
            mock_check.return_value = {
                "summary": "Test summary",
                "url": "https://example.com",
                "summary_claims": [],
                "url_claims": [],
                "claim_comparisons": [],
                "processing_time": 1.0
            }
            
            response = client.post(
                "/api/v1/consistency/check-summary-url",
                json={
                    "summary": "Test summary",
                    "url": "https://example.com",
                    "enable_deep_analysis": True,
                    "similarity_threshold": 0.5
                },
                headers={"X-API-Key": "test-key"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "summary" in data
            assert "url" in data
    
    def test_check_consistency_with_cot_mode(self):
        """Test consistency check with cot_mode parameter."""
        with patch('src.api.main._check_summary_vs_url_consistency', new_callable=AsyncMock) as mock_check:
            mock_check.return_value = {
                "summary": "Test summary",
                "url": "https://example.com",
                "summary_claims": [],
                "url_claims": [],
                "claim_comparisons": [],
                "processing_time": 1.0
            }
            
            response = client.post(
                "/api/v1/consistency/check-summary-url",
                json={
                    "summary": "Test summary",
                    "url": "https://example.com",
                    "enable_deep_analysis": True,
                    "similarity_threshold": 0.5,
                    "use_cot": True,
                    "cot_mode": "chain_local",
                    "use_few_shots": True
                },
                headers={"X-API-Key": "test-key"}
            )
            
            assert response.status_code == 200
            # Verify that cot_mode was passed to the function
            call_args = mock_check.call_args
            assert call_args[1]['cot_mode'] == "chain_local"
            assert call_args[1]['use_few_shots'] is True
    
    def test_check_consistency_with_use_few_shots_false(self):
        """Test consistency check with use_few_shots=False."""
        with patch('src.api.main._check_summary_vs_url_consistency', new_callable=AsyncMock) as mock_check:
            mock_check.return_value = {
                "summary": "Test summary",
                "url": "https://example.com",
                "summary_claims": [],
                "url_claims": [],
                "claim_comparisons": [],
                "processing_time": 1.0
            }
            
            response = client.post(
                "/api/v1/consistency/check-summary-url",
                json={
                    "summary": "Test summary",
                    "url": "https://example.com",
                    "enable_deep_analysis": True,
                    "similarity_threshold": 0.5,
                    "use_cot": False,
                    "cot_mode": "no_chain",
                    "use_few_shots": False
                },
                headers={"X-API-Key": "test-key"}
            )
            
            assert response.status_code == 200
            call_args = mock_check.call_args
            assert call_args[1]['use_few_shots'] is False
            assert call_args[1]['cot_mode'] == "no_chain"
    
    def test_check_consistency_request_model_validation(self):
        """Test that CheckConsistencyRequest model accepts new parameters."""
        request = CheckConsistencyRequest(
            summary="Test summary",
            url="https://example.com",
            cot_mode="chain_online",
            use_few_shots=False
        )
        
        assert request.cot_mode == "chain_online"
        assert request.use_few_shots is False
        assert request.use_cot is False  # default value


class TestOverallEvaluationAPI:
    """Tests for /api/v1/quality/overall endpoint."""
    
    def test_overall_evaluation_basic(self):
        """Test basic overall evaluation without new parameters."""
        with patch('src.api.main._evaluate_content', new_callable=AsyncMock) as mock_eval_content, \
             patch('src.api.main._evaluate_url', new_callable=AsyncMock) as mock_eval_url:
            
            mock_eval_content.return_value = {
                "content_type": "summary",
                "metrics": {},
                "basic_stats": {}
            }
            mock_eval_url.return_value = {
                "content_type": "url_content",
                "metrics": {},
                "basic_stats": {}
            }
            
            response = client.post(
                "/api/v1/quality/overall",
                json={
                    "summary": "Test summary",
                    "url": "https://example.com",
                    "enable_deep_analysis": True
                },
                headers={"X-API-Key": "test-key"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "summary_evaluation" in data
            assert "url_evaluation" in data
    
    def test_overall_evaluation_with_cot_mode(self):
        """Test overall evaluation with cot_mode parameter."""
        with patch('src.api.main._evaluate_content', new_callable=AsyncMock) as mock_eval_content, \
             patch('src.api.main._evaluate_url', new_callable=AsyncMock) as mock_eval_url, \
             patch('src.api.main._check_summary_vs_url_consistency', new_callable=AsyncMock) as mock_check:
            
            mock_eval_content.return_value = {
                "content_type": "summary",
                "metrics": {},
                "basic_stats": {}
            }
            mock_eval_url.return_value = {
                "content_type": "url_content",
                "metrics": {},
                "basic_stats": {}
            }
            mock_check.return_value = {
                "summary": "Test summary",
                "url": "https://example.com",
                "summary_claims": [],
                "url_claims": [],
                "claim_comparisons": [],
                "processing_time": 1.0
            }
            
            response = client.post(
                "/api/v1/quality/overall",
                json={
                    "summary": "Test summary",
                    "url": "https://example.com",
                    "enable_deep_analysis": True,
                    "enable_consistency_check": True,
                    "use_cot": True,
                    "cot_mode": "chain_online",
                    "use_few_shots": True
                },
                headers={"X-API-Key": "test-key"}
            )
            
            assert response.status_code == 200
            # Verify that parameters were passed
            check_call_args = mock_check.call_args
            assert check_call_args[1]['cot_mode'] == "chain_online"
            assert check_call_args[1]['use_few_shots'] is True
    
    def test_overall_evaluation_with_full_analysis(self):
        """Test overall evaluation with full analysis pipeline."""
        with patch('src.api.main._evaluate_url', new_callable=AsyncMock) as mock_eval_url, \
             patch('src.api.main._run_full_analysis_pipeline', new_callable=AsyncMock) as mock_full_analysis:
            
            mock_eval_url.return_value = {
                "content_type": "url_content",
                "metrics": {},
                "basic_stats": {}
            }
            mock_full_analysis.return_value = {
                "overall_success": True,
                "steps": [],
                "final_report": None,
                "total_execution_time": 1.0,
                "error": None
            }
            
            response = client.post(
                "/api/v1/quality/overall",
                json={
                    "url": "https://example.com",
                    "include_full_analysis": True,
                    "enable_deep_analysis": True,
                    "use_cot": True,
                    "cot_mode": "chain_local",
                    "use_few_shots": False
                },
                headers={"X-API-Key": "test-key"}
            )
            
            assert response.status_code == 200
            # Verify that parameters were passed to full analysis
            full_analysis_call_args = mock_full_analysis.call_args
            assert full_analysis_call_args[1]['cot_mode'] == "chain_local"
            assert full_analysis_call_args[1]['use_few_shots'] is False
    
    def test_overall_evaluation_request_model_validation(self):
        """Test that OverallEvaluationRequest model accepts new parameters."""
        request = OverallEvaluationRequest(
            summary="Test summary",
            url="https://example.com",
            cot_mode="chain_local",
            use_few_shots=False
        )
        
        assert request.cot_mode == "chain_local"
        assert request.use_few_shots is False
        assert request.use_cot is False  # default value
    
    def test_overall_evaluation_all_cot_modes(self):
        """Test overall evaluation with all cot_mode options."""
        cot_modes = ["chain_online", "chain_local", "no_chain"]
        
        for cot_mode in cot_modes:
            with patch('src.api.main._evaluate_content', new_callable=AsyncMock) as mock_eval_content, \
                 patch('src.api.main._evaluate_url', new_callable=AsyncMock) as mock_eval_url:
                
                mock_eval_content.return_value = {
                    "content_type": "summary",
                    "metrics": {},
                    "basic_stats": {}
                }
                mock_eval_url.return_value = {
                    "content_type": "url_content",
                    "metrics": {},
                    "basic_stats": {}
                }
                
                response = client.post(
                    "/api/v1/quality/overall",
                    json={
                        "summary": "Test summary",
                        "url": "https://example.com",
                        "enable_deep_analysis": True,
                        "cot_mode": cot_mode,
                        "use_few_shots": True
                    },
                    headers={"X-API-Key": "test-key"}
                )
                
                assert response.status_code == 200, f"Failed for cot_mode={cot_mode}"


class TestCompareClaimsAPI:
    """Tests for /api/v1/consistency/compare-claims endpoint."""
    
    def test_compare_claims_basic(self):
        """Test basic claim comparison without new parameters."""
        with patch('src.api.main.ChatOpenAI') as mock_llm_class:
            mock_response = Mock()
            mock_response.content = """STATUS: supported
CONFIDENCE: 0.85
REASON: The URL claim provides specific evidence that supports the summary claim."""
            
            mock_llm = Mock()
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm
            
            response = client.post(
                "/api/v1/consistency/compare-claims",
                json={
                    "summary_claim": "The company reported record profits",
                    "url_claim": "Q3 earnings exceeded all previous quarters"
                },
                headers={"X-API-Key": "test-key"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "summary_claim" in data
            assert "url_claim" in data
            assert "comparison" in data
            assert "reasoning" in data["comparison"]  # Must have reasoning
            assert data["comparison"]["reasoning"]  # Must not be empty
    
    def test_compare_claims_with_cot_mode(self):
        """Test claim comparison with cot_mode parameter."""
        with patch('src.api.main.ChatOpenAI') as mock_llm_class:
            mock_response = Mock()
            mock_response.content = """STATUS: contradicted
CONFIDENCE: 0.90
REASON: The URL claim explicitly contradicts the summary claim."""
            
            mock_llm = Mock()
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm
            
            response = client.post(
                "/api/v1/consistency/compare-claims",
                json={
                    "summary_claim": "The policy will reduce unemployment",
                    "url_claim": "The new policy is expected to increase unemployment",
                    "use_cot": True,
                    "cot_mode": "chain_online",
                    "use_few_shots": True
                },
                headers={"X-API-Key": "test-key"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "reasoning" in data["comparison"]
            assert data["comparison"]["reasoning"]
            # Verify max_tokens was increased for CoT
            assert mock_llm.max_tokens == 400
    
    def test_compare_claims_with_use_few_shots_false(self):
        """Test claim comparison with use_few_shots=False."""
        with patch('src.api.main.ChatOpenAI') as mock_llm_class:
            mock_response = Mock()
            mock_response.content = """STATUS: neutral
CONFIDENCE: 0.60
REASON: The claims are related but neither directly supports nor contradicts the other."""
            
            mock_llm = Mock()
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm
            
            response = client.post(
                "/api/v1/consistency/compare-claims",
                json={
                    "summary_claim": "The meeting discussed budget allocations",
                    "url_claim": "The meeting covered various topics",
                    "use_cot": False,
                    "cot_mode": "no_chain",
                    "use_few_shots": False
                },
                headers={"X-API-Key": "test-key"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "reasoning" in data["comparison"]
            assert data["comparison"]["reasoning"]
    
    def test_compare_claims_force_reasoning(self):
        """Test that reasoning is always present even if LLM doesn't provide it."""
        with patch('src.api.main.ChatOpenAI') as mock_llm_class:
            # Simulate LLM response without REASON: line
            mock_response = Mock()
            mock_response.content = """STATUS: supported
CONFIDENCE: 0.85"""
            
            mock_llm = Mock()
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm
            
            response = client.post(
                "/api/v1/consistency/compare-claims",
                json={
                    "summary_claim": "Test claim 1",
                    "url_claim": "Test claim 2"
                },
                headers={"X-API-Key": "test-key"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "reasoning" in data["comparison"]
            # Reasoning should be generated even if not in LLM response
            assert data["comparison"]["reasoning"]
            assert len(data["comparison"]["reasoning"]) > 0
    
    def test_compare_claims_request_model_validation(self):
        """Test that CompareClaimsRequest model accepts new parameters and rejects url_content."""
        # Test with new parameters
        request = CompareClaimsRequest(
            summary_claim="Test summary claim",
            url_claim="Test URL claim",
            cot_mode="chain_local",
            use_few_shots=False
        )
        
        assert request.cot_mode == "chain_local"
        assert request.use_few_shots is False
        assert request.use_cot is False  # default value
        
        # Verify url_content is not in the model
        assert not hasattr(request, 'url_content')
    
    def test_compare_claims_all_cot_modes(self):
        """Test claim comparison with all cot_mode options."""
        cot_modes = ["chain_online", "chain_local", "no_chain"]
        
        for cot_mode in cot_modes:
            with patch('src.api.main.ChatOpenAI') as mock_llm_class:
                mock_response = Mock()
                mock_response.content = f"""STATUS: neutral
CONFIDENCE: 0.50
REASON: Test reasoning for {cot_mode} mode."""
                
                mock_llm = Mock()
                mock_llm.invoke.return_value = mock_response
                mock_llm_class.return_value = mock_llm
                
                response = client.post(
                    "/api/v1/consistency/compare-claims",
                    json={
                        "summary_claim": "Test summary claim",
                        "url_claim": "Test URL claim",
                        "cot_mode": cot_mode,
                        "use_few_shots": True
                    },
                    headers={"X-API-Key": "test-key"}
                )
                
                assert response.status_code == 200, f"Failed for cot_mode={cot_mode}"
                data = response.json()
                assert "reasoning" in data["comparison"]
                assert data["comparison"]["reasoning"]


class TestCompareClaimsFewShotsAPI:
    """Tests for compare-claims few-shot management endpoints."""
    
    def test_get_compare_claims_shots(self):
        """Test getting default few-shot examples."""
        response = client.get(
            "/api/v1/consistency/compare-claims/shots",
            headers={"X-API-Key": "test-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "endpoint" in data
        assert data["endpoint"] == "compare-claims"
        assert "few_shots" in data
        assert "is_custom" in data
        assert isinstance(data["few_shots"], str)
        assert len(data["few_shots"]) > 0
    
    def test_set_compare_claims_custom_shots(self):
        """Test setting custom few-shot examples."""
        custom_shots = """CUSTOM EXAMPLES:

Example 1:
SUMMARY CLAIM: "Test claim 1"
URL CLAIM: "Test claim 2"
STATUS: supported
CONFIDENCE: 0.85
REASON: Custom reasoning example.
"""
        
        response = client.post(
            "/api/v1/consistency/compare-claims/custom-shots",
            json={"custom_few_shots": custom_shots},
            headers={"X-API-Key": "test-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoint" in data
        assert data["endpoint"] == "compare-claims"
    
    def test_get_compare_claims_custom_shots(self):
        """Test getting custom few-shot examples."""
        # First set custom shots
        custom_shots = """CUSTOM EXAMPLES:
Example test.
"""
        client.post(
            "/api/v1/consistency/compare-claims/custom-shots",
            json={"custom_few_shots": custom_shots},
            headers={"X-API-Key": "test-key"}
        )
        
        # Then get them
        response = client.get(
            "/api/v1/consistency/compare-claims/custom-shots",
            headers={"X-API-Key": "test-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "endpoint" in data
        assert data["endpoint"] == "compare-claims"
        assert "custom_few_shots" in data
        assert data["custom_few_shots"] == custom_shots
    
    def test_reset_compare_claims_custom_shots(self):
        """Test resetting custom few-shot examples."""
        # First set custom shots
        custom_shots = """CUSTOM EXAMPLES:
Example test.
"""
        client.post(
            "/api/v1/consistency/compare-claims/custom-shots",
            json={"custom_few_shots": custom_shots},
            headers={"X-API-Key": "test-key"}
        )
        
        # Then reset them
        response = client.delete(
            "/api/v1/consistency/compare-claims/custom-shots",
            headers={"X-API-Key": "test-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoint" in data
        
        # Verify they are reset
        get_response = client.get(
            "/api/v1/consistency/compare-claims/custom-shots",
            headers={"X-API-Key": "test-key"}
        )
        assert get_response.status_code == 200
        get_data = get_response.json()
        assert get_data["custom_few_shots"] is None
    
    def test_compare_claims_uses_custom_shots(self):
        """Test that compare-claims endpoint uses custom few-shot examples when set."""
        # Set custom shots
        custom_shots = """CUSTOM EXAMPLES:

Example 1:
SUMMARY CLAIM: "Custom test claim 1"
URL CLAIM: "Custom test claim 2"
STATUS: supported
CONFIDENCE: 0.90
REASON: This is a custom example.
"""
        client.post(
            "/api/v1/consistency/compare-claims/custom-shots",
            json={"custom_few_shots": custom_shots},
            headers={"X-API-Key": "test-key"}
        )
        
        # Test that the endpoint uses custom shots
        with patch('src.api.main.ChatOpenAI') as mock_llm_class:
            mock_response = Mock()
            mock_response.content = """STATUS: supported
CONFIDENCE: 0.85
REASON: Test reasoning."""
            
            mock_llm = Mock()
            mock_llm.invoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm
            
            response = client.post(
                "/api/v1/consistency/compare-claims",
                json={
                    "summary_claim": "Test summary claim",
                    "url_claim": "Test URL claim",
                    "use_few_shots": True
                },
                headers={"X-API-Key": "test-key"}
            )
            
            assert response.status_code == 200
            # Verify that custom shots were used (check the system prompt)
            call_args = mock_llm.invoke.call_args
            system_prompt = call_args[0][0][0]["content"]
            assert "CUSTOM EXAMPLES" in system_prompt
            assert "Custom test claim 1" in system_prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

