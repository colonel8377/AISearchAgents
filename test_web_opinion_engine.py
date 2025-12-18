"""
Tests for WebOpinionEngine.

These tests verify the modular "glass box" design and MBFC integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sqlite3
import tempfile
import os

from src.agents.web_opinion_extractor import (
    WebOpinionEngine,
    ArticleContent,
    SourceMetadata,
    AtomicUnit,
    BiasResult,
    BiasDistribution,
    NetworkError,
    ContentExtractionError,
)


class TestArticleContent:
    """Tests for ArticleContent model."""
    
    def test_create_article_content(self):
        """Test creating ArticleContent."""
        article = ArticleContent(
            title="Test Article",
            full_text="This is the full text.",
            domain="example.com",
            url="https://example.com/article"
        )
        assert article.title == "Test Article"
        assert article.full_text == "This is the full text."
        assert article.domain == "example.com"
        assert article.url == "https://example.com/article"


class TestSourceMetadata:
    """Tests for SourceMetadata model."""
    
    def test_create_source_metadata_exact(self):
        """Test creating SourceMetadata with exact match."""
        metadata = SourceMetadata(
            source_name="CNN",
            raw_db_row={"source_name": "CNN", "bias_rating": "left-center"},
            match_type="exact",
            bias_rating="left-center",
            factual_reporting="high"
        )
        assert metadata.source_name == "CNN"
        assert metadata.match_type == "exact"
        assert metadata.bias_rating == "left-center"
    
    def test_create_source_metadata_none(self):
        """Test creating SourceMetadata with no match."""
        metadata = SourceMetadata(
            source_name=None,
            raw_db_row=None,
            match_type="none",
            bias_rating=None,
            factual_reporting=None
        )
        assert metadata.source_name is None
        assert metadata.match_type == "none"
    
    def test_create_source_metadata_disabled(self):
        """Test creating SourceMetadata when MBFC is disabled."""
        metadata = SourceMetadata(
            source_name=None,
            raw_db_row=None,
            match_type="disabled",
            bias_rating=None,
            factual_reporting=None
        )
        assert metadata.match_type == "disabled"


class TestAtomicUnit:
    """Tests for AtomicUnit model."""
    
    def test_create_atomic_unit_fact(self):
        """Test creating an AtomicUnit for a fact."""
        unit = AtomicUnit(
            statement="The bill was passed on January 5th",
            type="fact",
            original_sentence="The bill was passed on January 5th, 2024.",
            confidence=0.95
        )
        assert unit.statement == "The bill was passed on January 5th"
        assert unit.type == "fact"
        assert unit.confidence == 0.95
    
    def test_create_atomic_unit_opinion(self):
        """Test creating an AtomicUnit for an opinion."""
        unit = AtomicUnit(
            statement="The policy is harmful",
            type="opinion",
            original_sentence="I believe the policy is harmful to workers."
        )
        assert unit.statement == "The policy is harmful"
        assert unit.type == "opinion"


class TestBiasResult:
    """Tests for BiasResult model."""
    
    def test_create_bias_result_with_metadata(self):
        """Test creating BiasResult when metadata was used."""
        bias_dist = BiasDistribution(left=0.6, right=0.2, neutral=0.2)
        result = BiasResult(
            bias_distribution=bias_dist,
            reasoning="Analysis shows left-leaning bias based on prior.",
            metadata_used=True
        )
        assert result.bias_distribution.left == 0.6
        assert result.metadata_used is True
        assert "prior" in result.reasoning.lower()
    
    def test_create_bias_result_without_metadata(self):
        """Test creating BiasResult when no metadata was used."""
        bias_dist = BiasDistribution(left=0.3, right=0.3, neutral=0.4)
        result = BiasResult(
            bias_distribution=bias_dist,
            reasoning="Neutral analysis without prior.",
            metadata_used=False
        )
        assert result.bias_distribution.neutral == 0.4
        assert result.metadata_used is False


class TestWebOpinionEngine:
    """Tests for WebOpinionEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create a WebOpinionEngine instance for testing."""
        return WebOpinionEngine(
            model_name="gpt-3.5-turbo",
            temperature=0.3,
            db_path=None,
            request_timeout=10.0
        )
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary SQLite database for testing."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        # Create test database
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE media_sources (
                id INTEGER PRIMARY KEY,
                source_name TEXT NOT NULL,
                source_url TEXT NOT NULL,
                bias_rating TEXT,
                factual_reporting TEXT
            )
        """)
        cursor.execute("""
            INSERT INTO media_sources VALUES
                (1, 'CNN', 'cnn.com', 'left-center', 'high'),
                (2, 'Fox News', 'foxnews.com', 'right', 'mixed'),
                (3, 'BBC', 'bbc.com', 'center', 'high')
        """)
        conn.commit()
        conn.close()
        
        yield path
        
        # Cleanup
        os.unlink(path)
    
    def test_engine_initialization(self, engine):
        """Test that engine initializes correctly."""
        assert engine is not None
        assert engine.llm is not None
        assert engine.request_timeout == 10.0
        assert engine.db_path is None
    
    def test_normalize_domain(self, engine):
        """Test domain normalization."""
        assert engine._normalize_domain("www.example.com") == "example.com"
        assert engine._normalize_domain("EXAMPLE.COM") == "example.com"
        assert engine._normalize_domain("www.bbc.co.uk") == "bbc.com"
        assert engine._normalize_domain("bbc.co.uk") == "bbc.com"
    
    @patch('httpx.Client')
    @patch('src.agents.web_opinion_extractor.engine.trafilatura')
    def test_extract_content_with_trafilatura(self, mock_trafilatura, mock_httpx, engine):
        """Test extract_content using trafilatura."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.text = "<html><body>Test content</body></html>"
        mock_response.raise_for_status = Mock()
        
        mock_client = Mock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client.get = Mock(return_value=mock_response)
        mock_httpx.return_value = mock_client
        
        # Mock trafilatura
        mock_trafilatura.fetch_url = Mock(return_value="<html>Test</html>")
        mock_trafilatura.extract = Mock(return_value="This is a test article with enough content to pass validation.")
        
        mock_metadata = Mock()
        mock_metadata.title = "Test Title"
        mock_trafilatura.extract_metadata = Mock(return_value=mock_metadata)
        
        # Test extraction
        result = engine.extract_content("https://example.com/article")
        
        assert isinstance(result, ArticleContent)
        assert result.domain == "example.com"
        assert result.url == "https://example.com/article"
        assert len(result.full_text) > 0
    
    def test_resolve_metadata_no_db(self, engine):
        """Test resolve_metadata when no database is configured."""
        metadata = engine.resolve_metadata("https://example.com/article")
        
        assert isinstance(metadata, SourceMetadata)
        assert metadata.match_type == "disabled"
        assert metadata.source_name is None
    
    def test_resolve_metadata_exact_match(self, temp_db):
        """Test resolve_metadata with exact database match."""
        engine = WebOpinionEngine(db_path=temp_db)
        
        metadata = engine.resolve_metadata("https://cnn.com/article")
        
        assert isinstance(metadata, SourceMetadata)
        assert metadata.match_type == "exact"
        assert metadata.source_name == "CNN"
        assert metadata.bias_rating == "left-center"
    
    def test_resolve_metadata_no_match(self, temp_db):
        """Test resolve_metadata when domain is not in database."""
        engine = WebOpinionEngine(db_path=temp_db)
        
        metadata = engine.resolve_metadata("https://unknown-site.com/article")
        
        assert isinstance(metadata, SourceMetadata)
        assert metadata.match_type == "none"
        assert metadata.source_name is None
    
    @patch.object(WebOpinionEngine, 'llm')
    def test_atomize_text(self, mock_llm, engine):
        """Test atomize_text method."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = """[
            {
                "statement": "Support for policy X",
                "type": "opinion",
                "original_sentence": "I support policy X."
            },
            {
                "statement": "Opposition to policy Y",
                "type": "opinion",
                "original_sentence": "I oppose policy Y."
            }
        ]"""
        engine.llm.invoke = Mock(return_value=mock_response)
        
        # Test atomization
        text = "I support policy X. I oppose policy Y."
        units = engine.atomize_text(text)
        
        assert isinstance(units, list)
        assert len(units) == 2
        assert all(isinstance(u, AtomicUnit) for u in units)
        assert units[0].statement == "Support for policy X"
        assert units[0].type == "opinion"
    
    @patch.object(WebOpinionEngine, 'llm')
    def test_calculate_bias_with_metadata(self, mock_llm, engine):
        """Test calculate_bias with MBFC metadata."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = """{
            "bias_distribution": {
                "left": 0.6,
                "right": 0.2,
                "neutral": 0.2
            },
            "reasoning": "Analysis based on MBFC prior.",
            "metadata_used": true
        }"""
        engine.llm.invoke = Mock(return_value=mock_response)
        
        # Create test data
        units = [
            AtomicUnit(statement="Support for policy", type="opinion")
        ]
        metadata = SourceMetadata(
            source_name="CNN",
            raw_db_row={},
            match_type="exact",
            bias_rating="left-center",
            factual_reporting="high"
        )
        
        # Test bias calculation
        result = engine.calculate_bias(units, metadata)
        
        assert isinstance(result, BiasResult)
        assert result.metadata_used is True
        assert result.bias_distribution.left == 0.6
        assert result.bias_distribution.dominant_bias == "left"
    
    @patch.object(WebOpinionEngine, 'llm')
    def test_calculate_bias_without_metadata(self, mock_llm, engine):
        """Test calculate_bias without MBFC metadata."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = """{
            "bias_distribution": {
                "left": 0.33,
                "right": 0.33,
                "neutral": 0.34
            },
            "reasoning": "Neutral analysis without prior.",
            "metadata_used": false
        }"""
        engine.llm.invoke = Mock(return_value=mock_response)
        
        # Create test data
        units = [
            AtomicUnit(statement="Balanced statement", type="opinion")
        ]
        metadata = SourceMetadata(
            source_name=None,
            raw_db_row=None,
            match_type="none",
            bias_rating=None,
            factual_reporting=None
        )
        
        # Test bias calculation
        result = engine.calculate_bias(units, metadata)
        
        assert isinstance(result, BiasResult)
        assert result.metadata_used is False
        assert result.bias_distribution.dominant_bias == "neutral"
    
    @patch.object(WebOpinionEngine, 'calculate_bias')
    @patch.object(WebOpinionEngine, 'atomize_text')
    @patch.object(WebOpinionEngine, 'resolve_metadata')
    @patch.object(WebOpinionEngine, 'extract_content')
    def test_run_pipeline(
        self,
        mock_extract,
        mock_resolve,
        mock_atomize,
        mock_calc_bias,
        engine
    ):
        """Test the complete run_pipeline method."""
        # Mock all agents
        mock_extract.return_value = ArticleContent(
            title="Test Article",
            full_text="Full text here.",
            domain="example.com",
            url="https://example.com/article"
        )
        
        mock_resolve.return_value = SourceMetadata(
            source_name="Example",
            raw_db_row={},
            match_type="exact",
            bias_rating="center",
            factual_reporting="high"
        )
        
        mock_atomize.return_value = [
            AtomicUnit(statement="Statement 1", type="opinion"),
            AtomicUnit(statement="Statement 2", type="fact")
        ]
        
        mock_calc_bias.return_value = BiasResult(
            bias_distribution=BiasDistribution(left=0.3, right=0.3, neutral=0.4),
            reasoning="Test reasoning",
            metadata_used=True
        )
        
        # Run pipeline
        result = engine.run_pipeline("https://example.com/article", use_mbfc=True)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert "url" in result
        assert "article" in result
        assert "metadata" in result
        assert "atomic_units" in result
        assert "bias_analysis" in result
        assert "pipeline_metadata" in result
        
        assert result["url"] == "https://example.com/article"
        assert result["article"]["domain"] == "example.com"
        assert len(result["atomic_units"]) == 2
        assert result["bias_analysis"]["dominant_bias"] == "neutral"
        assert result["pipeline_metadata"]["use_mbfc"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
