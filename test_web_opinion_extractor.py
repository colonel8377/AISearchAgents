"""
Tests for Web Opinion Extractor Agent.

These tests verify the functionality of the WebOpinionExtractor agent
without requiring external network or LLM API access.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.agents.web_opinion_extractor import (
    WebOpinionExtractor,
    AtomicOpinion,
    OpinionExtractionResult,
    WebExtractionError,
    NetworkError,
    ContentExtractionError,
)


class TestAtomicOpinion:
    """Tests for AtomicOpinion model."""
    
    def test_create_atomic_opinion(self):
        """Test creating an atomic opinion with all fields."""
        opinion = AtomicOpinion(
            text="I support the tax cut",
            opinion_type="opinion",
            bias_score=0.6,
            original_sentence="I support the tax cut but oppose the trade war",
            confidence=0.9
        )
        assert opinion.text == "I support the tax cut"
        assert opinion.opinion_type == "opinion"
        assert opinion.bias_score == 0.6
        assert opinion.original_sentence == "I support the tax cut but oppose the trade war"
        assert opinion.confidence == 0.9
    
    def test_bias_score_bounds(self):
        """Test that bias score is within valid bounds."""
        # Valid scores
        opinion = AtomicOpinion(text="test", opinion_type="opinion", bias_score=-1.0)
        assert opinion.bias_score == -1.0
        
        opinion = AtomicOpinion(text="test", opinion_type="opinion", bias_score=1.0)
        assert opinion.bias_score == 1.0
        
        opinion = AtomicOpinion(text="test", opinion_type="opinion", bias_score=0.0)
        assert opinion.bias_score == 0.0
        
        # Out of bounds should fail validation
        with pytest.raises(Exception):  # Pydantic ValidationError
            AtomicOpinion(text="test", opinion_type="opinion", bias_score=-1.5)
        
        with pytest.raises(Exception):
            AtomicOpinion(text="test", opinion_type="opinion", bias_score=1.5)
    
    def test_opinion_type_values(self):
        """Test that opinion_type only accepts valid values."""
        opinion = AtomicOpinion(text="test", opinion_type="fact", bias_score=0.0)
        assert opinion.opinion_type == "fact"
        
        opinion = AtomicOpinion(text="test", opinion_type="opinion", bias_score=0.0)
        assert opinion.opinion_type == "opinion"
        
        with pytest.raises(Exception):  # Pydantic ValidationError
            AtomicOpinion(text="test", opinion_type="invalid", bias_score=0.0)


class TestOpinionExtractionResult:
    """Tests for OpinionExtractionResult model."""
    
    def test_create_result(self):
        """Test creating an extraction result."""
        opinion = AtomicOpinion(
            text="I support the tax cut",
            opinion_type="opinion",
            bias_score=0.6
        )
        fact = AtomicOpinion(
            text="The bill was passed",
            opinion_type="fact",
            bias_score=0.0
        )
        
        result = OpinionExtractionResult(
            url="https://example.com",
            title="Test Article",
            atomic_opinions=[opinion, fact],
            facts=[fact],
            opinions=[opinion],
            overall_bias_score=0.6,
            text_length=100,
            truncated=False
        )
        
        assert result.url == "https://example.com"
        assert len(result.atomic_opinions) == 2
        assert len(result.facts) == 1
        assert len(result.opinions) == 1
        assert result.overall_bias_score == 0.6
    
    def test_calculate_overall_bias(self):
        """Test overall bias calculation."""
        opinion1 = AtomicOpinion(text="test1", opinion_type="opinion", bias_score=-0.5)
        opinion2 = AtomicOpinion(text="test2", opinion_type="opinion", bias_score=0.5)
        
        result = OpinionExtractionResult(
            opinions=[opinion1, opinion2],
            text_length=100
        )
        
        bias = result.calculate_overall_bias()
        assert bias == 0.0  # Average of -0.5 and 0.5


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
        """Test creating atomic opinion from valid data."""
        data = {
            "text": "I support the policy",
            "opinion_type": "opinion",
            "bias_score": 0.5,
            "original_sentence": "I support the policy strongly",
            "confidence": 0.9
        }
        opinion = extractor._create_atomic_opinion(data)
        assert opinion.text == "I support the policy"
        assert opinion.opinion_type == "opinion"
        assert opinion.bias_score == 0.5
        assert opinion.confidence == 0.9
    
    def test_create_atomic_opinion_string_bias_score(self, extractor):
        """Test handling of string bias score."""
        data = {
            "text": "Test",
            "opinion_type": "opinion",
            "bias_score": "0.7"
        }
        opinion = extractor._create_atomic_opinion(data)
        assert opinion.bias_score == 0.7
    
    def test_create_atomic_opinion_clamps_bias_score(self, extractor):
        """Test that bias score is clamped to valid range."""
        data = {
            "text": "Test",
            "opinion_type": "opinion",
            "bias_score": 2.5  # Out of range
        }
        opinion = extractor._create_atomic_opinion(data)
        assert opinion.bias_score == 1.0  # Clamped to max
        
        data["bias_score"] = -2.5
        opinion = extractor._create_atomic_opinion(data)
        assert opinion.bias_score == -1.0  # Clamped to min
    
    def test_create_atomic_opinion_invalid_type_defaults_to_opinion(self, extractor):
        """Test that invalid opinion_type defaults to opinion."""
        data = {
            "text": "Test",
            "opinion_type": "invalid_type",
            "bias_score": 0.0
        }
        opinion = extractor._create_atomic_opinion(data)
        assert opinion.opinion_type == "opinion"


class TestWebOpinionExtractorIntegration:
    """Integration tests for WebOpinionExtractor."""
    
    def test_extract_from_html_with_mocked_llm(self):
        """Test full extraction flow with mocked LLM."""
        # Mock response from LLM
        mock_response = Mock()
        mock_response.content = '''
        {
            "atomic_opinions": [
                {
                    "text": "Support for tax cuts",
                    "opinion_type": "opinion",
                    "bias_score": 0.7,
                    "original_sentence": "I support the tax cut",
                    "confidence": 0.9
                },
                {
                    "text": "The bill was passed on Monday",
                    "opinion_type": "fact",
                    "bias_score": 0.0,
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
        assert result.opinions[0].bias_score == 0.7
        assert result.facts[0].bias_score == 0.0
    
    def test_extract_from_text_compound_sentences(self):
        """Test that compound sentences are properly handled."""
        mock_response = Mock()
        mock_response.content = '''
        {
            "atomic_opinions": [
                {
                    "text": "Support for tax cuts",
                    "opinion_type": "opinion",
                    "bias_score": 0.6,
                    "original_sentence": "I support the tax cut but oppose the trade war"
                },
                {
                    "text": "Opposition to trade war",
                    "opinion_type": "opinion",
                    "bias_score": -0.3,
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
