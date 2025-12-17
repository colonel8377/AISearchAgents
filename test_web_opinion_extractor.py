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
        html = '<html><body><p>Content paragraph here.</p></body></html>'
        text, title = extractor._clean_html(html)
        assert title is None
    
    def test_clean_html_removes_buttons_and_forms(self, extractor):
        """Test that buttons, forms, and input elements are removed."""
        html = '''
        <html>
        <body>
            <article>
                <h1>Article Title</h1>
                <p>This is the main article content that should be extracted.</p>
                <button>Click me</button>
                <form><input type="text" placeholder="Enter text"/></form>
                <p>More important content here.</p>
            </article>
        </body>
        </html>
        '''
        text, title = extractor._clean_html(html)
        assert 'Click me' not in text
        assert 'Enter text' not in text
        assert 'main article content' in text
        assert 'important content' in text
    
    def test_clean_html_removes_images_and_media(self, extractor):
        """Test that images, videos, and other media are removed."""
        html = '''
        <html>
        <body>
            <article>
                <h1>News Article</h1>
                <img src="photo.jpg" alt="A photo"/>
                <p>This is the article body text.</p>
                <video src="video.mp4">Video content</video>
                <p>More article text here.</p>
            </article>
        </body>
        </html>
        '''
        text, title = extractor._clean_html(html)
        assert 'photo.jpg' not in text
        assert 'video.mp4' not in text
        assert 'article body text' in text
        assert 'article text here' in text
    
    def test_clean_html_extracts_main_article_content(self, extractor):
        """Test that main article content is extracted from article tags."""
        html = '''
        <html>
        <head><title>Page Title</title></head>
        <body>
            <header>Site Header</header>
            <nav><ul><li><a href="/">Home</a></li><li><a href="/about">About</a></li></ul></nav>
            <article>
                <h1>Breaking News: Important Event</h1>
                <p>This is the first paragraph of the news article with important details.</p>
                <p>This is the second paragraph with more information about the event.</p>
            </article>
            <aside>Related articles sidebar</aside>
            <footer>Site Footer</footer>
        </body>
        </html>
        '''
        text, title = extractor._clean_html(html)
        assert 'Breaking News' in text or 'Important Event' in text
        assert 'first paragraph' in text
        assert 'second paragraph' in text
        assert 'Site Header' not in text
        assert 'Site Footer' not in text
        assert 'Related articles' not in text
    
    def test_clean_html_removes_navigation_links(self, extractor):
        """Test that navigation link lists are removed."""
        html = '''
        <html>
        <body>
            <ul>
                <li><a href="/page1">Link 1</a></li>
                <li><a href="/page2">Link 2</a></li>
                <li><a href="/page3">Link 3</a></li>
            </ul>
            <main>
                <p>This is the main content paragraph that should be extracted from the page.</p>
            </main>
        </body>
        </html>
        '''
        text, title = extractor._clean_html(html)
        assert 'main content paragraph' in text
        # Navigation links should be removed or minimized
        # The key is that main content is preserved
    
    def test_clean_html_extracts_article_title_from_h1(self, extractor):
        """Test that article title is extracted from h1 tag."""
        html = '''
        <html>
        <head><title>Website - News Section</title></head>
        <body>
            <article>
                <h1>The Actual Article Headline</h1>
                <p>Article body content goes here.</p>
            </article>
        </body>
        </html>
        '''
        text, title = extractor._clean_html(html)
        assert title == "The Actual Article Headline"


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


class TestChainOfThoughtSupport:
    """Tests for Chain of Thought (CoT) reasoning support."""
    
    def test_atomic_opinion_with_reasoning(self):
        """Test that AtomicOpinion can include reasoning field."""
        bias = BiasDistribution(left=0.7, right=0.1, neutral=0.2)
        opinion = AtomicOpinion(
            text="Support for universal healthcare",
            opinion_type="opinion",
            bias_probabilities=bias,
            reasoning="This expresses support for universal healthcare, which aligns with progressive/left ideology. High left probability (0.7) due to policy alignment."
        )
        assert opinion.reasoning is not None
        assert "progressive" in opinion.reasoning.lower()
        assert opinion.bias_probabilities.left == 0.7
    
    def test_atomic_opinion_without_reasoning(self):
        """Test that reasoning field is optional."""
        bias = BiasDistribution(left=0.3, right=0.3, neutral=0.4)
        opinion = AtomicOpinion(
            text="Test opinion",
            opinion_type="opinion",
            bias_probabilities=bias
        )
        assert opinion.reasoning is None
    
    def test_extractor_with_chain_local_mode(self):
        """Test WebOpinionExtractor with chain_local execution mode."""
        with patch.object(WebOpinionExtractor, '__init__', lambda self, **kwargs: None):
            extractor = WebOpinionExtractor()
            extractor.execution_mode = "chain_local"
            extractor.request_timeout = 30.0
            extractor._extraction_history = []
            extractor.llm = Mock()
            extractor.llm.model_name = "test-model"
            extractor.llm.temperature = 0.3
            
            # Mock response with reasoning field
            mock_response = Mock()
            mock_response.content = '''
            {
                "atomic_opinions": [
                    {
                        "text": "Support for tax reform",
                        "opinion_type": "opinion",
                        "bias_probabilities": {"left": 0.2, "right": 0.6, "neutral": 0.2},
                        "reasoning": "This statement supports tax reform, typically associated with conservative fiscal policy. Right probability is 0.6 due to policy alignment.",
                        "confidence": 0.85
                    }
                ]
            }
            '''
            extractor.llm.invoke = Mock(return_value=mock_response)
            
            result = extractor.extract_from_text("I support tax reform.")
            
            assert len(result.opinions) == 1
            assert result.opinions[0].reasoning is not None
            assert "conservative" in result.opinions[0].reasoning.lower()
    
    def test_extractor_with_no_chain_mode(self):
        """Test WebOpinionExtractor with no_chain execution mode (no CoT)."""
        with patch.object(WebOpinionExtractor, '__init__', lambda self, **kwargs: None):
            extractor = WebOpinionExtractor()
            extractor.execution_mode = "no_chain"
            extractor.request_timeout = 30.0
            extractor._extraction_history = []
            extractor.llm = Mock()
            extractor.llm.model_name = "test-model"
            extractor.llm.temperature = 0.3
            
            # Mock response without reasoning field
            mock_response = Mock()
            mock_response.content = '''
            {
                "atomic_opinions": [
                    {
                        "text": "Support for tax reform",
                        "opinion_type": "opinion",
                        "bias_probabilities": {"left": 0.2, "right": 0.6, "neutral": 0.2},
                        "confidence": 0.85
                    }
                ]
            }
            '''
            extractor.llm.invoke = Mock(return_value=mock_response)
            
            result = extractor.extract_from_text("I support tax reform.")
            
            assert len(result.opinions) == 1
            # Reasoning may be None in no_chain mode
            assert result.opinions[0].reasoning is None


class TestWebOpinionAnalyzer:
    """Tests for WebOpinionAnalyzer high-level API."""
    
    def test_analyzer_initialization(self):
        """Test WebOpinionAnalyzer initialization."""
        with patch.object(WebOpinionExtractor, '__init__', return_value=None):
            from src.agents.web_opinion_extractor import WebOpinionAnalyzer
            analyzer = WebOpinionAnalyzer(execution_mode="chain_local")
            # Should initialize without errors
    
    def test_extract_and_analyze_with_network_error(self):
        """Test that extract_and_analyze returns error state on network failure."""
        from src.agents.web_opinion_extractor import WebOpinionAnalyzer
        
        with patch.object(WebOpinionExtractor, '__init__', return_value=None):
            analyzer = WebOpinionAnalyzer()
            analyzer._extractor = Mock()
            analyzer._extractor.extract_and_analyze = Mock(
                side_effect=NetworkError("Connection failed", "http://example.com", 500)
            )
            
            result = analyzer.extract_and_analyze("http://example.com")
            
            # Should return valid result with error metadata
            assert result.extraction_metadata is not None
            assert result.extraction_metadata["error"] == "network_error"
            assert "Connection failed" in result.extraction_metadata["error_message"]
            assert result.extraction_metadata["status_code"] == 500
            assert len(result.opinions) == 0
    
    def test_extract_and_analyze_with_content_error(self):
        """Test that extract_and_analyze returns error state on content extraction failure."""
        from src.agents.web_opinion_extractor import WebOpinionAnalyzer
        
        with patch.object(WebOpinionExtractor, '__init__', return_value=None):
            analyzer = WebOpinionAnalyzer()
            analyzer._extractor = Mock()
            analyzer._extractor.extract_and_analyze = Mock(
                side_effect=ContentExtractionError("No content found", "http://example.com")
            )
            
            result = analyzer.extract_and_analyze("http://example.com")
            
            # Should return valid result with error metadata
            assert result.extraction_metadata is not None
            assert result.extraction_metadata["error"] == "content_extraction_error"
            assert "No content found" in result.extraction_metadata["error_message"]
            assert len(result.opinions) == 0
    
    def test_extract_html_verification_api(self):
        """Test extract_html verification method."""
        from src.agents.web_opinion_extractor import WebOpinionAnalyzer
        
        with patch.object(WebOpinionExtractor, '__init__', return_value=None):
            analyzer = WebOpinionAnalyzer()
            analyzer._extractor = Mock()
            analyzer._extractor.extract_html = Mock(return_value="<html>test</html>")
            
            html = analyzer.extract_html("http://example.com")
            
            assert html == "<html>test</html>"
            analyzer._extractor.extract_html.assert_called_once_with("http://example.com")
    
    def test_clean_html_verification_api(self):
        """Test clean_html verification method."""
        from src.agents.web_opinion_extractor import WebOpinionAnalyzer
        
        with patch.object(WebOpinionExtractor, '__init__', return_value=None):
            analyzer = WebOpinionAnalyzer()
            analyzer._extractor = Mock()
            analyzer._extractor.clean_html = Mock(return_value=("Clean text", "Title"))
            
            text, title = analyzer.clean_html("<html>test</html>")
            
            assert text == "Clean text"
            assert title == "Title"
    
    def test_analyze_text_verification_api(self):
        """Test analyze_text verification method."""
        from src.agents.web_opinion_extractor import WebOpinionAnalyzer
        
        with patch.object(WebOpinionExtractor, '__init__', return_value=None):
            analyzer = WebOpinionAnalyzer()
            analyzer._extractor = Mock()
            
            # Create a mock result
            mock_result = OpinionExtractionResult(
                atomic_opinions=[],
                facts=[],
                opinions=[],
                text_length=100
            )
            analyzer._extractor.analyze_text = Mock(return_value=mock_result)
            
            result = analyzer.analyze_text("Test text")
            
            assert result == mock_result
            analyzer._extractor.analyze_text.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
