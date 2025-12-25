"""
Tests for Web Opinion Extract API endpoints.

This module tests the FastAPI endpoints for web opinion extraction,
including HTML extraction, opinion analysis, and bias scoring.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.agents.web_opinion_extractor import (
    OpinionExtractionResult,
    AtomicOpinion,
    BiasDistribution
)


class TestWebOpinionExtractApi:
    """Tests for web opinion extract API endpoints."""
    
    @pytest.fixture
    def mock_analyzer(self):
        """Create a mock WebOpinionAnalyzer."""
        with patch('src.api.main.WebOpinionAnalyzer') as mock:
            analyzer_instance = Mock()
            mock.return_value = analyzer_instance
            yield analyzer_instance
    
    @pytest.fixture
    def sample_html(self):
        """Sample HTML content for testing."""
        return """
        <html>
        <head><title>Test Article</title></head>
        <body>
            <article>
                <h1>Breaking News</h1>
                <p>This is a test article with some content.</p>
            </article>
        </body>
        </html>
        """
    
    @pytest.fixture
    def sample_result(self):
        """Sample OpinionExtractionResult for testing."""
        opinion_bias = BiasDistribution(left=0.2, right=0.6, neutral=0.2)
        opinion = AtomicOpinion(
            text="Support for policy",
            opinion_type="opinion",
            bias_probabilities=opinion_bias,
            confidence=0.9
        )
        
        fact_bias = BiasDistribution(left=0.1, right=0.1, neutral=0.8)
        fact = AtomicOpinion(
            text="The bill was passed",
            opinion_type="fact",
            bias_probabilities=fact_bias,
            confidence=0.95
        )
        
        overall_bias = BiasDistribution(left=0.2, right=0.6, neutral=0.2)
        
        return OpinionExtractionResult(
            url="https://example.com",
            title="Test Article",
            atomic_opinions=[opinion, fact],
            facts=[fact],
            opinions=[opinion],
            overall_bias_distribution=overall_bias,
            text_length=100,
            truncated=False
        )
    
    def test_extractandclean_success(self, mock_analyzer, sample_html):
        """Test successful combined extract and clean."""
        mock_analyzer.extract_html.return_value = sample_html
        mock_analyzer.clean_html.return_value = ("Clean text content", "Test Title")
        
        response = client.post(
            "/api/v1/web-opinion/extractandclean",
            json={"url": "https://example.com"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["url"] == "https://example.com"
        assert data["text"] == "Clean text content"
        assert data["title"] == "Test Title"
        assert data["text_length"] == len("Clean text content")
        assert data["error"] is None
        
        # Verify both methods were called
        mock_analyzer.extract_html.assert_called_once_with("https://example.com")
        mock_analyzer.clean_html.assert_called_once_with(sample_html)
    
    def test_extractandclean_reuses_settings_proxy(self, mock_analyzer, sample_html):
        """Test combined extract and clean reuses proxy from settings."""
        mock_analyzer.extract_html.return_value = sample_html
        mock_analyzer.clean_html.return_value = ("Clean text content", "Test Title")
        
        response = client.post(
            "/api/v1/web-opinion/extractandclean",
            json={
                "url": "https://example.com"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["url"] == "https://example.com"
        assert data["text"] == "Clean text content"
        assert data["error"] is None
        # Proxy is configured via settings.openai_proxy or environment variables
    
    def test_extractandclean_fetch_failure(self, mock_analyzer):
        """Test combined extract and clean with fetch failure."""
        mock_analyzer.extract_html.return_value = None
        
        response = client.post(
            "/api/v1/web-opinion/extractandclean",
            json={"url": "https://invalid.url"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["error"] == "fetch_failed"
        assert data["text"] is None
    
    def test_extractandclean_cleaning_failure(self, mock_analyzer, sample_html):
        """Test combined extract and clean with cleaning failure."""
        mock_analyzer.extract_html.return_value = sample_html
        mock_analyzer.clean_html.return_value = (None, None)
        
        response = client.post(
            "/api/v1/web-opinion/extractandclean",
            json={"url": "https://example.com"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["error"] == "cleaning_failed"
        assert data["text"] is None
    
    def test_extract_opinions_success(self, mock_analyzer, sample_result):
        """Test successful opinion extraction."""
        mock_analyzer.analyze_text.return_value = sample_result
        
        response = client.post(
            "/api/v1/web-opinion/extract-opinions",
            json={
                "text": "Test text content",
                "url": "https://example.com",
                "title": "Test Article"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["url"] == "https://example.com"
        assert data["title"] == "Test Article"
        assert len(data["atomic_opinions"]) == 2
        assert len(data["facts"]) == 1
        assert len(data["opinions"]) == 1
        assert data["overall_bias_distribution"] is not None
        assert data["overall_bias_distribution"]["dominant_bias"] == "right"
        assert data["error"] is None
    
    def test_extract_opinions_with_execution_mode(self, mock_analyzer, sample_result):
        """Test opinion extraction with specific execution mode."""
        mock_analyzer.analyze_text.return_value = sample_result
        
        response = client.post(
            "/api/v1/web-opinion/extract-opinions",
            json={
                "text": "Test text content",
                "execution_mode": "chain_local"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["opinions"]) == 1
    
    def test_extract_opinions_error(self, mock_analyzer):
        """Test opinion extraction with error."""
        error_result = OpinionExtractionResult(
            atomic_opinions=[],
            facts=[],
            opinions=[],
            text_length=0,
            truncated=False,
            extraction_metadata={
                "error": "llm_analysis_error",
                "error_message": "LLM request failed"
            }
        )
        mock_analyzer.analyze_text.return_value = error_result
        
        response = client.post(
            "/api/v1/web-opinion/extract-opinions",
            json={"text": "Test text"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["error"] == "llm_analysis_error"
        assert "LLM request failed" in data["error_message"]
    
    def test_analyze_url_success(self, mock_analyzer, sample_result):
        """Test complete URL analysis."""
        mock_analyzer.extract_and_analyze.return_value = sample_result
        
        response = client.post(
            "/api/v1/web-opinion/analyze",
            json={"url": "https://example.com"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["url"] == "https://example.com"
        assert data["title"] == "Test Article"
        # atomic_opinions should only contain items with opinion_type == "opinion"
        assert len(data["atomic_opinions"]) == 1
        assert len(data["opinions"]) == 1
        assert data["overall_bias_distribution"] is not None
        
        # Check bias distribution structure
        bias = data["overall_bias_distribution"]
        assert "left" in bias
        assert "right" in bias
        assert "neutral" in bias
        assert "dominant_bias" in bias
        assert "bias_score" in bias
        assert bias["right"] == 0.6
        assert bias["dominant_bias"] == "right"
    
    def test_analyze_url_with_execution_mode(self, mock_analyzer, sample_result):
        """Test URL analysis with specific execution mode."""
        mock_analyzer.extract_and_analyze.return_value = sample_result
        
        response = client.post(
            "/api/v1/web-opinion/analyze",
            json={
                "url": "https://example.com",
                "execution_mode": "no_chain"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["url"] == "https://example.com"
    
    def test_analyze_url_network_error(self, mock_analyzer):
        """Test URL analysis with network error."""
        error_result = OpinionExtractionResult(
            atomic_opinions=[],
            facts=[],
            opinions=[],
            text_length=0,
            truncated=False,
            extraction_metadata={
                "error": "network_error",
                "error_message": "Connection timeout"
            }
        )
        mock_analyzer.extract_and_analyze.return_value = error_result
        
        response = client.post(
            "/api/v1/web-opinion/analyze",
            json={"url": "https://invalid.url"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["error"] == "network_error"
        assert "timeout" in data["error_message"].lower()
    
    def test_bias_score_success(self, mock_analyzer, sample_result):
        """Test getting overall bias score with URL."""
        mock_analyzer.extract_and_analyze.return_value = sample_result
        
        response = client.post(
            "/api/v1/opinion/bias-score",
            json={"url": "https://example.com"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["url"] == "https://example.com"
        assert data["content_provided"] is False
        assert data["overall_bias_distribution"] is not None
        assert data["opinions_count"] == 1
        assert data["facts_count"] == 1
        assert data["error"] is None
        
        # Check bias distribution
        bias = data["overall_bias_distribution"]
        assert bias["left"] == 0.2
        assert bias["right"] == 0.6
        assert bias["neutral"] == 0.2
        assert bias["dominant_bias"] == "right"
    
    def test_bias_score_with_content(self, mock_analyzer, sample_result):
        """Test getting overall bias score with content."""
        # Mock the engine methods for content analysis
        with patch('src.api.main.WebOpinionEngine') as mock_engine_class:
            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine
            
            # Mock atomize_text and calculate_bias
            from src.agents.web_opinion_extractor.models import AtomicUnit, BiasResult, BiasDistribution
            mock_atoms = [
                AtomicUnit(statement="Test opinion", type="opinion", original_sentence="Test"),
                AtomicUnit(statement="Test fact", type="fact", original_sentence="Test")
            ]
            mock_bias_result = BiasResult(
                bias_distribution=BiasDistribution(left=0.3, right=0.5, neutral=0.2),
                reasoning="Test reasoning",
                metadata_used=False,
                mbfc_influence_note=None
            )
            
            mock_engine.atomize_text.return_value = mock_atoms
            mock_engine.calculate_bias.return_value = mock_bias_result
            
            response = client.post(
                "/api/v1/opinion/bias-score",
                json={
                    "content": "This is a test article about politics. Some experts believe the policy will help.",
                    "title": "Test Article",
                    "mode": "LOCAL_CHAIN",
                    "use_mbfc": False,
                    "use_few_shots": True
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["url"] is None
            assert data["content_provided"] is True
            assert data["overall_bias_distribution"] is not None
            assert data["opinions_count"] == 1
            assert data["facts_count"] == 1
    
    def test_bias_score_with_execution_mode(self, mock_analyzer, sample_result):
        """Test bias score with specific execution mode."""
        mock_analyzer.extract_and_analyze.return_value = sample_result
        
        response = client.post(
            "/api/v1/opinion/bias-score",
            json={
                "url": "https://example.com",
                "mode": "PURE_ONLINE"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["overall_bias_distribution"] is not None
    
    def test_bias_score_error(self, mock_analyzer):
        """Test bias score with error."""
        error_result = OpinionExtractionResult(
            atomic_opinions=[],
            facts=[],
            opinions=[],
            text_length=0,
            truncated=False,
            extraction_metadata={
                "error": "content_extraction_error",
                "error_message": "No content found"
            }
        )
        mock_analyzer.extract_and_analyze.return_value = error_result
        
        response = client.post(
            "/api/v1/opinion/bias-score",
            json={"url": "https://empty-page.com"}
        )
        
        assert response.status_code == 200
        data = response.json()
            assert data["error"] == "content_extraction_error"
            assert data["overall_bias_distribution"] is None
    
    def test_bias_score_validation_error_no_input(self):
        """Test bias score with validation error (neither url nor content)."""
        response = client.post(
            "/api/v1/opinion/bias-score",
            json={}
        )
        
        assert response.status_code == 422  # Validation error
        error_detail = response.json()["detail"]
        assert "url" in str(error_detail).lower() or "content" in str(error_detail).lower()
    
    def test_bias_score_validation_error_both_provided(self):
        """Test bias score with both url and content (should fail validation)."""
        response = client.post(
            "/api/v1/opinion/bias-score",
            json={
                "url": "https://example.com",
                "content": "Test content"
            }
        )
        
        assert response.status_code == 422  # Validation error
        error_detail = response.json()["detail"]
        assert "both" in str(error_detail).lower() or "cannot" in str(error_detail).lower()
    
    def test_atomic_opinion_response_structure(self, mock_analyzer, sample_result):
        """Test that atomic opinion response has correct structure."""
        mock_analyzer.extract_and_analyze.return_value = sample_result
        
        response = client.post(
            "/api/v1/web-opinion/analyze",
            json={"url": "https://example.com"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check opinion structure
        opinion = data["opinions"][0]
        assert "text" in opinion
        assert "opinion_type" in opinion
        assert "bias_probabilities" in opinion
        assert "confidence" in opinion
        
        # Check bias probabilities structure
        bias_probs = opinion["bias_probabilities"]
        assert "left" in bias_probs
        assert "right" in bias_probs
        assert "neutral" in bias_probs
        assert "dominant_bias" in bias_probs
        assert "bias_score" in bias_probs
        
        # Verify probabilities sum to approximately 1.0
        total = bias_probs["left"] + bias_probs["right"] + bias_probs["neutral"]
        assert abs(total - 1.0) < 0.01
    
    def test_root_endpoint_includes_web_opinion_apis(self):
        """Test that root endpoint documents web opinion APIs."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check that web opinion endpoints are documented
        endpoints = data["endpoints"]
        assert "web_opinion_extract_html" in endpoints
        assert "web_opinion_clean_html" in endpoints
        assert "web_opinion_extract_opinions" in endpoints
        assert "web_opinion_analyze" in endpoints
        assert "web_opinion_bias_score" in endpoints
        
        # Check that features mention web opinion extraction
        features = data["features"]
        assert any("Opinion" in f for f in features)


class TestWebOpinionApiIntegration:
    """Integration tests for web opinion API endpoints."""
    
    def test_full_pipeline_simulation(self, mock_analyzer, sample_html, sample_result):
        """Simulate a full pipeline: extract HTML -> clean -> analyze."""
        with patch('src.api.main.WebOpinionAnalyzer') as mock_class:
            analyzer_instance = Mock()
            mock_class.return_value = analyzer_instance
            
            # Step 1: Extract HTML
            analyzer_instance.extract_html.return_value = sample_html
            response1 = client.post(
                "/api/v1/web-opinion/extract-html",
                json={"url": "https://example.com"}
            )
            assert response1.status_code == 200
            html = response1.json()["html"]
            
            # Step 2: Clean HTML
            analyzer_instance.clean_html.return_value = ("Cleaned text", "Title")
            response2 = client.post(
                "/api/v1/web-opinion/clean-html",
                json={"html": html}
            )
            assert response2.status_code == 200
            text = response2.json()["text"]
            
            # Step 3: Extract opinions
            analyzer_instance.analyze_text.return_value = sample_result
            response3 = client.post(
                "/api/v1/web-opinion/extract-opinions",
                json={"text": text}
            )
            assert response3.status_code == 200
            assert len(response3.json()["opinions"]) > 0
    
    def test_simplified_pipeline(self, mock_analyzer, sample_result):
        """Test simplified pipeline: just analyze URL."""
        with patch('src.api.main.WebOpinionAnalyzer') as mock_class:
            analyzer_instance = Mock()
            mock_class.return_value = analyzer_instance
            analyzer_instance.extract_and_analyze.return_value = sample_result
            
            response = client.post(
                "/api/v1/web-opinion/analyze",
                json={"url": "https://example.com"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["opinions"]) > 0
            assert data["overall_bias_distribution"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
