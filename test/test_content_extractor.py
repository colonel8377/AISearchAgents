"""
Tests for Content Extractor Agent.

These tests verify the functionality of the ContentExtractorAgent
without requiring external network or LLM API access.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from bs4 import BeautifulSoup

from src.agents.content_extractor.agent import (
    ContentExtractorAgent,
    ContentExtractionResult
)


class TestContentExtractorAgent:
    """Tests for ContentExtractorAgent."""

    @pytest.fixture
    def agent(self):
        """Create a ContentExtractorAgent instance for testing."""
        with patch('src.agents.content_extractor.agent.llm_manager'):
            agent = ContentExtractorAgent(
                model_name="test-model",
                api_key="test-key",
                temperature=0.1
            )
        return agent

    def test_agent_initialization(self, agent):
        """Test agent initialization with default parameters."""
        assert agent.execution_mode == "no_chain"
        assert agent.request_timeout == 30.0

    @patch('src.agents.content_extractor.agent.httpx.Client')
    def test_fetch_html_success(self, mock_client_class, agent):
        """Test successful HTML fetching."""
        mock_response = Mock()
        mock_response.text = "<html><body>Test content</body></html>"
        mock_response.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        result = agent._fetch_html("http://example.com")

        assert result == "<html><body>Test content</body></html>"
        mock_client.get.assert_called_once_with("http://example.com", follow_redirects=True)

    @patch('src.agents.content_extractor.agent.httpx.Client')
    def test_fetch_html_timeout(self, mock_client_class, agent):
        """Test HTML fetching with timeout."""
        from httpx import TimeoutException

        mock_client = Mock()
        mock_client.get.side_effect = TimeoutException("Request timed out")
        mock_client_class.return_value.__enter__.return_value = mock_client

        with pytest.raises(TimeoutException):
            agent._fetch_html("http://example.com")

    def test_clean_html_basic(self, agent):
        """Test basic HTML cleaning functionality."""
        html = """
        <html>
            <head><title>Test Title</title></head>
            <body>
                <script>alert('test');</script>
                <nav>Navigation</nav>
                <article>
                    <h1>Article Title</h1>
                    <p>Main content paragraph.</p>
                    <footer>Footer content</footer>
                </article>
            </body>
        </html>
        """

        text, title = agent._clean_html(html)

        assert title == "Test Title"
        assert "Main content paragraph" in text
        assert "alert('test');" not in text
        assert "Navigation" not in text
        assert "Footer content" not in text

    def test_clean_html_main_content_selector(self, agent):
        """Test HTML cleaning with main content selector."""
        html = """
        <html>
            <body>
                <div class="sidebar">Sidebar content</div>
                <article class="post-content">
                    <h1>Post Title</h1>
                    <p>Article content here.</p>
                </article>
            </body>
        </html>
        """

        text, title = agent._clean_html(html)

        assert title == "Post Title"
        assert "Article content here" in text
        assert "Sidebar content" not in text

    def test_remove_navigation_lists(self, agent):
        """Test removal of navigation-style lists."""
        html = """
        <div>
            <ul>
                <li><a href="/">Home</a></li>
                <li><a href="/about">About</a></li>
                <li><a href="/contact">Contact</a></li>
            </ul>
            <p>Actual content here.</p>
        </div>
        """

        soup = BeautifulSoup(html, 'html.parser')
        agent._remove_navigation_lists(soup)

        # Navigation list should be removed
        ul_tags = soup.find_all('ul')
        assert len(ul_tags) == 0

        # Content should remain
        p_tags = soup.find_all('p')
        assert len(p_tags) == 1

    def test_extract_article_text(self, agent):
        """Test article text extraction from HTML elements."""
        html = """
        <div>
            <h2>Section Header</h2>
            <p>First paragraph content.</p>
            <p>Second paragraph content.</p>
            <blockquote>Quoted text</blockquote>
        </div>
        """

        soup = BeautifulSoup(html, 'html.parser')
        text = agent._extract_article_text(soup, title="Test Title")

        assert "Test Title" in text
        assert "Section Header" in text
        assert "First paragraph content" in text
        assert "Second paragraph content" in text
        assert "Quoted text" in text

    @patch('src.agents.content_extractor.agent.ChatOpenAI')
    def test_extract_with_llm_cot(self, mock_chat_class, agent):
        """Test LLM extraction with Chain of Thought."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "- TITLE: Extracted Title\n- MAIN BODY: Extracted content"
        mock_llm.invoke.return_value = mock_response
        mock_chat_class.return_value = mock_llm

        # Create agent with mocked LLM
        with patch('src.agents.content_extractor.agent.llm_manager'):
            test_agent = ContentExtractorAgent(execution_mode="chain_online")

        title, body = test_agent._extract_with_llm("test content", use_cot=True)

        assert title == "Extracted Title"
        assert body == "Extracted content"
        mock_llm.invoke.assert_called_once()

    def test_truncate_text(self, agent):
        """Test text truncation functionality."""
        long_text = "Word " * 1000  # Very long text
        truncated, was_truncated = agent._truncate_text(long_text, max_length=100)

        assert was_truncated == True
        assert len(truncated) <= 100
        # Should end at word boundary
        assert not truncated.endswith("Word ")

    def test_extract_from_text_basic(self, agent):
        """Test basic text extraction without LLM."""
        text = "This is a test article content."

        result = agent.extract_from_text(text, title="Test Title", use_llm=False)

        assert isinstance(result, ContentExtractionResult)
        assert result.title == "Test Title"
        assert result.main_body == text
        assert result.text_length == len(text)
        assert result.truncated == False

    @patch('src.agents.content_extractor.agent.ChatOpenAI')
    def test_extract_from_text_with_llm(self, mock_chat_class, agent):
        """Test text extraction with LLM refinement."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "- TITLE: LLM Title\n- MAIN BODY: LLM content"
        mock_llm.invoke.return_value = mock_response
        mock_chat_class.return_value = mock_llm

        result = agent.extract_from_text("input text", use_llm=True)

        assert result.title == "LLM Title"
        assert result.main_body == "LLM content"

    def test_get_default_few_shots(self):
        """Test getting default few-shot examples."""
        few_shots = ContentExtractorAgent.get_default_few_shots()

        assert isinstance(few_shots, str)
        assert "Example 1" in few_shots
        assert "TITLE:" in few_shots
        assert "MAIN BODY:" in few_shots

    def test_reset_agent(self, agent):
        """Test agent reset functionality."""
        # Reset should not raise any errors
        agent.reset()
        # Agent should still be functional after reset
        assert agent.execution_mode == "no_chain"
