"""
HTML Extractor Base Class.

This module provides a robust HTML extraction and cleaning base class
that can be used by various content extraction agents.
"""

import re
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod

import httpx
from bs4 import BeautifulSoup, Tag

from ..config.settings import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Tags to strip from HTML for cleaning
NON_CONTENT_TAGS = [
    "script", "style", "nav", "footer", "iframe", "noscript", "header", "aside",
    "form", "button", "input", "select", "textarea", "label",  # Form elements
    "img", "figure", "figcaption", "picture", "video", "audio", "source", "track",  # Media
    "svg", "canvas", "map", "area",  # Graphics
    "menu", "menuitem",  # Menu elements
    "dialog", "details", "summary",  # Interactive elements
    "template", "slot",  # Template elements
]

# Tags that often contain navigation/non-content links
NAVIGATION_TAGS = ["ul", "ol", "li"]

# Common class/id patterns for main content areas
MAIN_CONTENT_SELECTORS = [
    "article",
    "[role='main']",
    "main",
    ".article-content",
    ".article-body",
    ".post-content",
    ".post-body",
    ".entry-content",
    ".content-body",
    ".story-body",
    ".news-body",
    "#article-body",
    "#main-content",
    "#content",
    ".main-content",
]

# Common class/id patterns for article titles
TITLE_SELECTORS = [
    "h1",
    ".article-title",
    ".post-title",
    ".entry-title",
    ".headline",
    "[itemprop='headline']",
]

# Maximum text length before truncation (characters)
# Roughly 4 chars per token, targeting ~8000 tokens for context
MAX_TEXT_LENGTH = 32000


class HTMLExtractor(ABC):
    """
    Base class for HTML content extraction with robust cleaning and parsing.

    This class provides common HTML extraction functionality that can be
    inherited by various content extraction agents.

    Features:
    - Robust HTML fetching with error handling
    - Intelligent content area detection
    - Navigation and non-content element removal
    - Text extraction and cleaning
    - Configurable content selectors
    """

    def __init__(self, request_timeout: float = 30.0):
        """
        Initialize the HTML extractor.

        Args:
            request_timeout: Timeout for HTTP requests in seconds
        """
        self.request_timeout = request_timeout

    def fetch_html(self, url: str) -> str:
        """
        Fetch HTML content from a URL with comprehensive error handling.

        Args:
            url: The URL to fetch

        Returns:
            HTML content as string

        Raises:
            httpx.TimeoutException: If the request times out
            httpx.HTTPStatusError: If the response has an error status
            httpx.RequestError: If there's a network error
        """
        logger.info(f"Fetching HTML from: {url}")

        try:
            with httpx.Client(timeout=self.request_timeout, proxy=settings.openai_proxy) as client:
                response = client.get(url, follow_redirects=True)
                response.raise_for_status()
                logger.debug(f"Successfully fetched {len(response.text)} characters from {url}")
                return response.text

        except httpx.TimeoutException as e:
            logger.error(f"Timeout fetching {url}: {e}")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching {url}: {e}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error fetching {url}: {e}")
            raise

    def clean_html(self, html: str) -> Tuple[str, Optional[str]]:
        """
        Parse and clean HTML content, extracting only the main article text.

        This method focuses on extracting the main content body (like news article text)
        while excluding navigation, buttons, links, images, and other non-content elements.

        Args:
            html: Raw HTML string

        Returns:
            Tuple of (cleaned_text, page_title)
        """
        logger.debug("Cleaning HTML content")

        soup = BeautifulSoup(html, "html.parser")

        # Extract page title from <title> tag
        page_title = None
        title_tag = soup.find("title")
        if title_tag:
            page_title = title_tag.get_text(strip=True)

        # Try to find article title from common heading selectors
        article_title = None
        for selector in TITLE_SELECTORS:
            title_element = soup.select_one(selector)
            if title_element:
                article_title = title_element.get_text(strip=True)
                break

        # Use article title if found, otherwise fall back to page title
        title = article_title or page_title

        # Try to find main content area using common selectors FIRST
        main_content = None
        for selector in MAIN_CONTENT_SELECTORS:
            main_content = soup.select_one(selector)
            if main_content:
                logger.debug(f"Found main content using selector: {selector}")
                break

        # If main content area found, extract directly from it (most efficient)
        if main_content:
            # Remove navigation-style lists that might be inside main content
            self._remove_navigation_lists(main_content)
            # Remove non-content tags from main content area only
            for tag_name in NON_CONTENT_TAGS:
                for tag in main_content.find_all(tag_name):
                    tag.decompose()
            # Remove all anchor tags (links) but keep their text
            for a_tag in main_content.find_all("a"):
                a_tag.unwrap()
            text = self._extract_article_text(main_content, title)
        else:
            # Fallback: try to extract from body, removing obvious non-content
            body = soup.find("body")
            if body:
                # Remove non-content tags first
                for tag_name in NON_CONTENT_TAGS:
                    for tag in body.find_all(tag_name):
                        tag.decompose()
                # Remove all anchor tags (links) but keep their text
                for a_tag in body.find_all("a"):
                    a_tag.unwrap()
                self._remove_navigation_lists(body)
                text = self._extract_article_text(body, title)
            else:
                # Last resort: extract from entire cleaned soup
                # Remove non-content tags first
                for tag_name in NON_CONTENT_TAGS:
                    for tag in soup.find_all(tag_name):
                        tag.decompose()
                # Remove all anchor tags (links) but keep their text
                for a_tag in soup.find_all("a"):
                    a_tag.unwrap()
                text = soup.get_text(separator="\n", strip=True)

        # Clean up whitespace
        text = re.sub(r'\n\s*\n+', '\n\n', text)  # Reduce multiple newlines to double
        text = re.sub(r'[ \t]+', ' ', text)  # Reduce multiple spaces to single
        text = text.strip()

        logger.debug(f"Cleaned text length: {len(text)} characters, title: {title}")

        return text, title

    def _remove_navigation_lists(self, element: Tag) -> None:
        """
        Remove list elements that appear to be navigation menus.

        Navigation lists typically have many links and short text items.

        Args:
            element: BeautifulSoup element to clean
        """
        for ul in element.find_all(["ul", "ol"]):
            # Count links vs total list items
            list_items = ul.find_all("li")
            if not list_items:
                continue

            links_count = len(ul.find_all("a"))
            items_count = len(list_items)

            # If most items are links, it's likely navigation
            if items_count > 0 and links_count / items_count > 0.7:
                # Check if items are short (navigation-like)
                avg_text_len = sum(len(li.get_text(strip=True)) for li in list_items) / items_count
                if avg_text_len < 50:  # Short items suggest navigation
                    ul.decompose()

    def _extract_article_text(self, element: Tag, title: Optional[str] = None) -> str:
        """
        Extract article text from an element, focusing on paragraph content.

        Args:
            element: BeautifulSoup element to extract text from
            title: Optional title to prepend

        Returns:
            Extracted text content
        """
        text_parts = []

        # Add title if provided
        if title:
            text_parts.append(title)

        # Extract text from paragraph-like elements (excluding h1 if it matches title)
        content_tags = ["p", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre"]

        for tag in element.find_all(content_tags):
            tag_text = tag.get_text(strip=True)
            if tag_text and len(tag_text) > 10:  # Skip very short fragments
                # Skip if this is the same as the title we already added
                if title and tag_text == title:
                    continue
                text_parts.append(tag_text)

        # If no paragraphs found, fall back to full text extraction
        if len(text_parts) <= 1:  # Only title or nothing
            fallback_text = element.get_text(separator="\n", strip=True)
            if fallback_text:
                text_parts.append(fallback_text)

        return "\n\n".join(text_parts)

    def truncate_text(self, text: str, max_length: int = MAX_TEXT_LENGTH) -> Tuple[str, bool]:
        """
        Truncate text to fit within context window limits.

        Args:
            text: The text to truncate
            max_length: Maximum length in characters

        Returns:
            Tuple of (truncated_text, was_truncated)
        """
        if len(text) <= max_length:
            return text, False

        logger.warning(f"Text length ({len(text)}) exceeds max ({max_length}), truncating")

        # Truncate at a sentence boundary if possible
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')

        # Use the later boundary to preserve more context
        boundary = max(last_period, last_newline)
        if boundary > max_length * 0.8:  # Only if we're not losing too much
            truncated = truncated[:boundary + 1]

        return truncated, True

    @abstractmethod
    def reset(self) -> None:
        """Reset the extractor to initial state."""
        pass
