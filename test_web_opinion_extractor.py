"""
Tests for Web Opinion Extractor Agent.

These tests verify the functionality of the WebOpinionExtractor agent
without requiring external network or LLM API access.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pydantic import ValidationError

from src.agents.web_opinion_extractor import (
    WebOpinionExtractor,
    AtomicOpinion,
    OpinionExtractionResult,
    BiasDistribution,
    WebExtractionError,
    NetworkError,
    ContentExtractionError,
)


class TestBiasDistribution:
    """Tests for BiasDistribution model."""
    
    def test_create_bias_distribution(self):
        """Test creating a bias distribution with valid probabilities."""
        bias = BiasDistribution(left=0.3, right=0.6, neutral=0.1)
        assert bias.left == 0.3
        assert bias.right == 0.6
        assert bias.neutral == 0.1
    
    def test_probabilities_normalized(self):
        """Test that probabilities are normalized to sum to 1.0 after validation."""
        # Input values within bounds but not summing to 1.0
        bias = BiasDistribution(left=0.4, right=0.4, neutral=0.4)  # Sum = 1.2
        total = bias.left + bias.right + bias.neutral
        assert abs(total - 1.0) < 0.01  # Should be normalized
    
    def test_dominant_bias(self):
        """Test getting the dominant bias category."""
        bias = BiasDistribution(left=0.7, right=0.2, neutral=0.1)
        assert bias.dominant_bias == "left"
        
        bias = BiasDistribution(left=0.1, right=0.8, neutral=0.1)
        assert bias.dominant_bias == "right"
        
        bias = BiasDistribution(left=0.2, right=0.2, neutral=0.6)
        assert bias.dominant_bias == "neutral"
    
    def test_bias_score_property(self):
        """Test converting probability distribution to single score."""
        # Fully left should be -1.0
        bias = BiasDistribution(left=1.0, right=0.0, neutral=0.0)
        assert bias.bias_score == -1.0
        
        # Fully right should be +1.0
        bias = BiasDistribution(left=0.0, right=1.0, neutral=0.0)
        assert bias.bias_score == 1.0
        
        # Fully neutral should be 0.0
        bias = BiasDistribution(left=0.0, right=0.0, neutral=1.0)
        assert bias.bias_score == 0.0
        
        # Mixed should be weighted average
        bias = BiasDistribution(left=0.5, right=0.5, neutral=0.0)
        assert bias.bias_score == 0.0  # -0.5 + 0.5 = 0


class TestAtomicOpinion:
    """Tests for AtomicOpinion model."""
    
    def test_create_atomic_opinion(self):
        """Test creating an atomic opinion with all fields."""
        bias = BiasDistribution(left=0.2, right=0.6, neutral=0.2)
        opinion = AtomicOpinion(
            text="I support the tax cut",
            opinion_type="opinion",
            bias_probabilities=bias,
            original_sentence="I support the tax cut but oppose the trade war",
            confidence=0.9
        )
        assert opinion.text == "I support the tax cut"
        assert opinion.opinion_type == "opinion"
        # bias_score = -1*left + 0*neutral + 1*right = -0.2 + 0 + 0.6 = 0.4
        assert abs(opinion.bias_score - 0.4) < 0.01
        assert opinion.original_sentence == "I support the tax cut but oppose the trade war"
        assert opinion.confidence == 0.9
        assert opinion.bias_probabilities.right == 0.6
    
    def test_bias_probabilities_bounds(self):
        """Test that bias probabilities are within valid bounds."""
        # Valid probabilities
        bias = BiasDistribution(left=0.5, right=0.3, neutral=0.2)
        opinion = AtomicOpinion(text="test", opinion_type="opinion", bias_probabilities=bias)
        assert opinion.bias_probabilities.left == 0.5
        
        # Out of bounds (negative) should fail validation
        with pytest.raises(ValidationError):
            BiasDistribution(left=-0.1, right=0.5, neutral=0.6)
        
        # Values > 1.0 are normalized, not rejected
        # This tests that normalization works correctly
        bias = BiasDistribution(left=1.5, right=0.5, neutral=0.0)
        # 1.5 + 0.5 + 0.0 = 2.0, normalized: left=0.75, right=0.25, neutral=0.0
        assert abs(bias.left - 0.75) < 0.01
        assert abs(bias.right - 0.25) < 0.01
    
    def test_opinion_type_values(self):
        """Test that opinion_type only accepts valid values."""
        bias = BiasDistribution(left=0.33, right=0.33, neutral=0.34)
        opinion = AtomicOpinion(text="test", opinion_type="fact", bias_probabilities=bias)
        assert opinion.opinion_type == "fact"
        
        opinion = AtomicOpinion(text="test", opinion_type="opinion", bias_probabilities=bias)
        assert opinion.opinion_type == "opinion"
        
        with pytest.raises(ValidationError):
            AtomicOpinion(text="test", opinion_type="invalid", bias_probabilities=bias)


class TestOpinionExtractionResult:
    """Tests for OpinionExtractionResult model."""
    
    def test_create_result(self):
        """Test creating an extraction result."""
        opinion_bias = BiasDistribution(left=0.2, right=0.6, neutral=0.2)
        opinion = AtomicOpinion(
            text="I support the tax cut",
            opinion_type="opinion",
            bias_probabilities=opinion_bias
        )
        fact_bias = BiasDistribution(left=0.1, right=0.1, neutral=0.8)
        fact = AtomicOpinion(
            text="The bill was passed",
            opinion_type="fact",
            bias_probabilities=fact_bias
        )
        
        overall_bias = BiasDistribution(left=0.2, right=0.6, neutral=0.2)
        result = OpinionExtractionResult(
            url="https://example.com",
            title="Test Article",
            atomic_opinions=[opinion, fact],
            facts=[fact],
            opinions=[opinion],
            overall_bias_distribution=overall_bias,
            text_length=100,
            truncated=False
        )
        
        assert result.url == "https://example.com"
        assert len(result.atomic_opinions) == 2
        assert len(result.facts) == 1
        assert len(result.opinions) == 1
        assert result.overall_bias_distribution.right == 0.6
    
    def test_calculate_overall_bias(self):
        """Test overall bias calculation."""
        bias1 = BiasDistribution(left=0.7, right=0.1, neutral=0.2)
        bias2 = BiasDistribution(left=0.1, right=0.7, neutral=0.2)
        opinion1 = AtomicOpinion(text="test1", opinion_type="opinion", bias_probabilities=bias1)
        opinion2 = AtomicOpinion(text="test2", opinion_type="opinion", bias_probabilities=bias2)
        
        result = OpinionExtractionResult(
            opinions=[opinion1, opinion2],
            text_length=100
        )
        
        bias = result.calculate_overall_bias()
        # Average: left=(0.7+0.1)/2=0.4, right=(0.1+0.7)/2=0.4, neutral=(0.2+0.2)/2=0.2
        assert abs(bias.left - 0.4) < 0.01
        assert abs(bias.right - 0.4) < 0.01
        assert abs(bias.neutral - 0.2) < 0.01


class TestExceptions:
    """Tests for custom exceptions."""
    
    def test_network_error(self):
        """Test NetworkError exception."""
        error = NetworkError(
            message="Request timed out",
            url="https://example.com",
            status_code=408
        )
        assert error.url == "https://example.com"
        assert error.status_code == 408
        assert "timed out" in error.message
    
    def test_content_extraction_error(self):
        """Test ContentExtractionError exception."""
        error = ContentExtractionError(
            message="No content found",
            url="https://example.com"
        )
        assert error.url == "https://example.com"
        assert "No content" in error.message
    
    def test_exception_inheritance(self):
        """Test exception inheritance."""
        error = NetworkError("test", "url")
        assert isinstance(error, WebExtractionError)
        
        error = ContentExtractionError("test", "url")
        assert isinstance(error, WebExtractionError)


class TestWebOpinionExtractorHTMLCleaning:
    """Tests for HTML cleaning functionality."""
    
    @pytest.fixture
    def extractor(self):
        """Create a test extractor without LLM."""
        class TestExtractor(WebOpinionExtractor):
            def __init__(self):
                self.request_timeout = 30.0
                self._extraction_history = []
        return TestExtractor()
    
    def test_clean_html_removes_script_tags(self, extractor):
        """Test that script tags are removed."""
        html = '<html><script>alert("xss")</script><p>Content</p></html>'
        text, _ = extractor._clean_html(html)
        assert 'alert' not in text
        assert 'Content' in text
    
    def test_clean_html_removes_style_tags(self, extractor):
        """Test that style tags are removed."""
        html = '<html><style>.class { color: red; }</style><p>Content</p></html>'
        text, _ = extractor._clean_html(html)
        assert 'color' not in text
        assert 'Content' in text
    
    def test_clean_html_removes_nav_footer_iframe(self, extractor):
        """Test that nav, footer, and iframe tags are removed."""
        html = '''
        <html>
        <nav>Navigation links</nav>
        <main><p>Main content</p></main>
        <footer>Footer content</footer>
        <iframe src="ad.html"></iframe>
        </html>
        '''
        text, _ = extractor._clean_html(html)
        assert 'Navigation links' not in text
        assert 'Footer content' not in text
        assert 'ad.html' not in text
        assert 'Main content' in text
    
    def test_clean_html_extracts_title(self, extractor):
        """Test that page title is extracted."""
        html = '<html><head><title>Test Title</title></head><body>Content</body></html>'
        text, title = extractor._clean_html(html)
        assert title == "Test Title"
    
    def test_clean_html_handles_missing_title(self, extractor):
        """Test handling of missing title."""
        html = '<html><body>Content</body></html>'
        text, title = extractor._clean_html(html)
        assert title is None


class TestWebOpinionExtractorTruncation:
    """Tests for text truncation functionality."""
    
    @pytest.fixture
    def extractor(self):
        """Create a test extractor without LLM."""
        class TestExtractor(WebOpinionExtractor):
            def __init__(self):
                self.request_timeout = 30.0
                self._extraction_history = []
        return TestExtractor()
    
    def test_short_text_not_truncated(self, extractor):
        """Test that short text is not truncated."""
        text = "Short text."
        result, truncated = extractor._truncate_text(text)
        assert result == text
        assert truncated is False
    
    def test_long_text_truncated(self, extractor):
        """Test that long text is truncated."""
        text = "This is a sentence. " * 2000
        result, truncated = extractor._truncate_text(text)
        assert len(result) < len(text)
        assert truncated is True
    
    def test_truncation_at_sentence_boundary(self, extractor):
        """Test that truncation prefers sentence boundaries."""
        text = "This is a sentence. " * 2000
        result, _ = extractor._truncate_text(text)
        # Should end at a period (sentence boundary)
        assert result.rstrip().endswith('.')


class TestWebOpinionExtractorAtomicOpinionCreation:
    """Tests for atomic opinion creation."""
    
    @pytest.fixture
    def extractor(self):
        """Create a test extractor without LLM."""
        class TestExtractor(WebOpinionExtractor):
            def __init__(self):
                self.request_timeout = 30.0
                self._extraction_history = []
        return TestExtractor()
    
    def test_create_atomic_opinion_valid_data(self, extractor):
        """Test creating atomic opinion from valid probability data."""
        data = {
            "text": "I support the policy",
            "opinion_type": "opinion",
            "bias_probabilities": {
                "left": 0.2,
                "right": 0.6,
                "neutral": 0.2
            },
            "original_sentence": "I support the policy strongly",
            "confidence": 0.9
        }
        opinion = extractor._create_atomic_opinion(data)
        assert opinion.text == "I support the policy"
        assert opinion.opinion_type == "opinion"
        assert opinion.bias_probabilities.right == 0.6
        assert opinion.confidence == 0.9
    
    def test_create_atomic_opinion_from_legacy_bias_score(self, extractor):
        """Test creating atomic opinion from legacy single bias score."""
        data = {
            "text": "I support the policy",
            "opinion_type": "opinion",
            "bias_score": 0.5,  # Old format
            "confidence": 0.9
        }
        opinion = extractor._create_atomic_opinion(data)
        # Should convert to probability distribution
        assert opinion.bias_probabilities.right == 0.5
        assert opinion.bias_probabilities.neutral == 0.5
        assert opinion.bias_probabilities.left == 0.0
    
    def test_create_atomic_opinion_string_bias_score(self, extractor):
        """Test handling of string bias score (legacy format)."""
        data = {
            "text": "Test",
            "opinion_type": "opinion",
            "bias_score": "0.7"
        }
        opinion = extractor._create_atomic_opinion(data)
        # Converted to probability: right=0.7, neutral=0.3, left=0.0
        assert opinion.bias_probabilities.right == 0.7
        assert abs(opinion.bias_probabilities.neutral - 0.3) < 0.01
    
    def test_create_atomic_opinion_clamps_legacy_bias_score(self, extractor):
        """Test that legacy bias score is clamped to valid range."""
        data = {
            "text": "Test",
            "opinion_type": "opinion",
            "bias_score": 2.5  # Out of range, should be clamped to 1.0
        }
        opinion = extractor._create_atomic_opinion(data)
        # Clamped to 1.0, which means right=1.0
        assert opinion.bias_probabilities.right == 1.0
        
        data["bias_score"] = -2.5  # Should be clamped to -1.0
        opinion = extractor._create_atomic_opinion(data)
        # Clamped to -1.0, which means left=1.0
        assert opinion.bias_probabilities.left == 1.0
    
    def test_create_atomic_opinion_invalid_type_defaults_to_opinion(self, extractor):
        """Test that invalid opinion_type defaults to opinion."""
        data = {
            "text": "Test",
            "opinion_type": "invalid_type",
            "bias_probabilities": {"left": 0.33, "right": 0.33, "neutral": 0.34}
        }
        opinion = extractor._create_atomic_opinion(data)
        assert opinion.opinion_type == "opinion"


class TestWebOpinionExtractorIntegration:
    """Integration tests for WebOpinionExtractor."""
    
    def test_extract_from_html_with_mocked_llm(self):
        """Test full extraction flow with mocked LLM."""
        # Mock response from LLM with new bias_probabilities format
        mock_response = Mock()
        mock_response.content = '''
        {
            "atomic_opinions": [
                {
                    "text": "Support for tax cuts",
                    "opinion_type": "opinion",
                    "bias_probabilities": {"left": 0.1, "right": 0.7, "neutral": 0.2},
                    "original_sentence": "I support the tax cut",
                    "confidence": 0.9
                },
                {
                    "text": "The bill was passed on Monday",
                    "opinion_type": "fact",
                    "bias_probabilities": {"left": 0.1, "right": 0.1, "neutral": 0.8},
                    "original_sentence": "The bill was passed on Monday",
                    "confidence": 0.95
                }
            ]
        }
        '''
        
        # Create extractor with mocked LLM
        with patch.object(WebOpinionExtractor, '__init__', lambda self, **kwargs: None):
            extractor = WebOpinionExtractor()
            extractor.request_timeout = 30.0
            extractor._extraction_history = []
            extractor.llm = Mock()
            extractor.llm.invoke = Mock(return_value=mock_response)
            extractor.llm.model_name = "test-model"
            extractor.llm.temperature = 0.3
        
        html = '''
        <html>
        <head><title>Test Article</title></head>
        <body>
            <main>
                <p>I support the tax cut.</p>
                <p>The bill was passed on Monday.</p>
            </main>
        </body>
        </html>
        '''
        
        result = extractor.extract_from_html(html, url="https://example.com")
        
        assert result.title == "Test Article"
        assert result.url == "https://example.com"
        assert len(result.atomic_opinions) == 2
        assert len(result.facts) == 1
        assert len(result.opinions) == 1
        assert result.opinions[0].bias_probabilities.right == 0.7
        assert result.facts[0].bias_probabilities.neutral == 0.8
    
    def test_extract_from_text_compound_sentences(self):
        """Test that compound sentences are properly handled."""
        mock_response = Mock()
        mock_response.content = '''
        {
            "atomic_opinions": [
                {
                    "text": "Support for tax cuts",
                    "opinion_type": "opinion",
                    "bias_probabilities": {"left": 0.2, "right": 0.6, "neutral": 0.2},
                    "original_sentence": "I support the tax cut but oppose the trade war"
                },
                {
                    "text": "Opposition to trade war",
                    "opinion_type": "opinion",
                    "bias_probabilities": {"left": 0.5, "right": 0.2, "neutral": 0.3},
                    "original_sentence": "I support the tax cut but oppose the trade war"
                }
            ]
        }
        '''
        
        with patch.object(WebOpinionExtractor, '__init__', lambda self, **kwargs: None):
            extractor = WebOpinionExtractor()
            extractor.request_timeout = 30.0
            extractor._extraction_history = []
            extractor.llm = Mock()
            extractor.llm.invoke = Mock(return_value=mock_response)
            extractor.llm.model_name = "test-model"
            extractor.llm.temperature = 0.3
        
        result = extractor.extract_from_text("I support the tax cut but oppose the trade war.")
        
        # Should have 2 atomic opinions from compound sentence
        assert len(result.atomic_opinions) == 2
        assert result.opinions[0].original_sentence == result.opinions[1].original_sentence


class TestWebOpinionExtractorNetworkErrors:
    """Tests for network error handling."""
    
    @pytest.fixture
    def extractor(self):
        """Create a test extractor without LLM."""
        class TestExtractor(WebOpinionExtractor):
            def __init__(self):
                self.request_timeout = 1.0
                self._extraction_history = []
        return TestExtractor()
    
    def test_connection_refused_raises_network_error(self, extractor):
        """Test that connection refused raises NetworkError."""
        with pytest.raises(NetworkError) as exc_info:
            extractor._fetch_html("http://127.0.0.1:99999")
        assert "127.0.0.1" in exc_info.value.url
    
    def test_invalid_domain_raises_network_error(self, extractor):
        """Test that invalid domain raises NetworkError."""
        with pytest.raises(NetworkError) as exc_info:
            extractor._fetch_html("http://this.domain.definitely.does.not.exist.xyz")
        assert "this.domain.definitely.does.not.exist.xyz" in exc_info.value.url


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
