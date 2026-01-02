"""
HTML Extractor Base Class.

This module provides a robust HTML extraction and cleaning base class
that can be used by various content extraction agents.
"""

import importlib
import re
import unicodedata
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

import httpx
from bs4 import BeautifulSoup, Tag, FeatureNotFound

from ..utils.logger import get_logger

logger = get_logger(__name__)

class HTMLExtractor(ABC):
    """
    Base class for HTML content extraction with robust cleaning and parsing.

    Features:
    - Robust HTML fetching with encoding fallback
    - Intelligent content area detection
    - Navigation/Ad removal based on link density
    - Unicode normalization
    - Extensible selector configuration
    """

    # --- Configuration: Override these in subclasses if needed ---

    # Tags to strip entirely from HTML
    NON_CONTENT_TAGS: List[str] = [
        "script", "style", "nav", "footer", "iframe", "noscript", "header", "aside",
        "form", "button", "input", "select", "textarea", "label", "fieldset",
        "img", "figure", "figcaption", "picture", "video", "audio", "source", "track",
        "svg", "canvas", "map", "area",
        "menu", "menuitem", "dialog", "details", "summary",
        "template", "slot", "ins", "ads", "div.advertisement"
    ]

    # Tags usually associated with navigation lists
    NAVIGATION_TAGS: List[str] = ["ul", "ol", "li", "dl", "dt", "dd"]

    # Selectors for finding the primary content container
    # Order matters: Specific ID/Classes first, semantic tags later
    MAIN_CONTENT_SELECTORS: List[str] = [
        "article",
        "[role='main']",
        ".article-body", "#article-body",
        ".post-content", ".entry-content",
        ".story-body", ".news-body",
        "#main-content", ".main-content",
        "main",
        "#content", ".content"
    ]

    # Selectors for finding the Headline/Title
    TITLE_SELECTORS: List[str] = [
        "h1",
        "[itemprop='headline']",
        ".article-title",
        ".post-title",
        ".entry-title",
        ".headline"
    ]

    # Tags considered to hold meaningful text content
    TEXT_BLOCK_TAGS: List[str] = [
        "p", "h2", "h3", "h4", "h5", "h6",
        "blockquote", "pre", "li", "div"
    ]

    def __init__(self, request_timeout: float = 30.0, user_agent: Optional[str] = None):
        """
        Initialize the HTML extractor.

        Args:
            request_timeout: Timeout for HTTP requests in seconds.
            user_agent: Optional custom User-Agent string.
        """
        self.request_timeout = request_timeout
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )

    def fetch_html(self, url: str) -> str:
        """
        Fetch HTML content from a URL with robust error handling and encoding detection.

        Args:
            url: The URL to fetch

        Returns:
            HTML content as string

        Raises:
            httpx.RequestError: For network-level errors.
            httpx.HTTPStatusError: For 4xx/5xx errors (unless handled).
        """
        logger.info(f"Fetching HTML from: {url}")

        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        try:
            with httpx.Client(timeout=self.request_timeout, trust_env=True) as client:
                response = client.get(url, headers=headers, follow_redirects=True)

                # Robust Encoding Handling:
                # If charset isn't provided, httpx defaults to ISO-8859-1.
                # If the content looks like UTF-8 but was decoded as ISO, fix it.
                if response.encoding == "ISO-8859-1":
                    try:
                        # Peek at the content to see if it's actually valid UTF-8
                        response.content.decode("utf-8")
                        response.encoding = "utf-8"
                    except UnicodeDecodeError:
                        pass # It was actually ISO-8859-1 or something else

                # Custom Error Handling for "Soft" Failures
                if response.status_code >= 400:
                    # Allow non-standard codes if they contain substantial body content (anti-bot triggers, etc.)
                    is_standard_error = response.status_code in {401, 403, 404, 408, 429, 500, 502, 503, 504}

                    if not is_standard_error and len(response.content) > 500:
                        logger.warning(
                            f"Non-standard HTTP status {response.status_code} from {url}, "
                            f"but response has {len(response.content)} bytes. Attempting to parse."
                        )
                        return response.text

                    response.raise_for_status()

                logger.debug(f"Successfully fetched {len(response.text)} chars from {url}")
                return response.text

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP {e.response.status_code} error fetching {url}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Network error fetching {url}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {str(e)}")
            raise

    def clean_html(self, html: str) -> Tuple[str, Optional[str]]:
        """
        Parse and clean HTML content, extracting only the main article text.

        Args:
            html: Raw HTML string

        Returns:
            Tuple of (cleaned_text, page_title)
        """
        logger.debug("Cleaning HTML content")

        if not html:
            return "", None

        # Prefer lxml for speed, fallback to html.parser
        try:
            soup = BeautifulSoup(html, "lxml")
        except FeatureNotFound:
            soup = BeautifulSoup(html, "html.parser")

        # 1. Extract Title
        title = self._extract_title(soup)

        # 2. Identify Main Content Area
        main_content = self._find_main_content(soup)

        # 3. Aggressive Cleaning on the isolated content
        self._clean_element(main_content)

        # 4. Extract Text
        text = self._extract_article_text(main_content, title)

        # 5. Final String Normalization
        text = self._normalize_text(text)

        logger.debug(f"Cleaned text length: {len(text)} characters")
        return text, title

    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract the best possible title."""
        # Try specific article title selectors first
        for selector in self.TITLE_SELECTORS:
            tag = soup.select_one(selector)
            if tag:
                return tag.get_text(strip=True)

        # Fallback to <title> tag
        title_tag = soup.find("title")
        if title_tag:
            return title_tag.get_text(strip=True)

        return None

    def _find_main_content(self, soup: BeautifulSoup) -> Tag:
        """Find the DOM node containing the main article."""
        for selector in self.MAIN_CONTENT_SELECTORS:
            element = soup.select_one(selector)
            if element:
                logger.debug(f"Found main content using selector: {selector}")
                return element

        # Fallback to body
        body = soup.find("body")
        if body:
            return body

        return soup

    def _clean_element(self, element: Tag) -> None:
        """Mutate the element by removing non-content noise."""
        # 1. Remove banned tags
        for tag_name in self.NON_CONTENT_TAGS:
            for tag in element.find_all(tag_name):
                tag.decompose()

        # 2. Remove comments
        for comment in element.find_all(string=lambda text: isinstance(text, importlib.import_module('bs4').Comment)):
            comment.extract()

        # 3. Remove Navigation Lists (Link Density Heuristic)
        self._remove_link_heavy_areas(element)

        # 4. Unwrap simple styling tags but keep text
        # (b, i, strong, em, span, a)
        for tag_name in ["b", "strong", "i", "em", "span", "a", "font"]:
            for tag in element.find_all(tag_name):
                tag.unwrap()

    def _remove_link_heavy_areas(self, element: Tag) -> None:
        """Remove container elements that are primarily links (menus, related posts)."""
        # Check lists and generic divs
        for container in element.find_all(self.NAVIGATION_TAGS + ["div"]):
            # Get all text length
            text_content = container.get_text(strip=True)
            if not text_content:
                container.decompose()
                continue

            total_len = len(text_content)

            # Get link text length
            links = container.find_all("a")
            if not links:
                continue

            link_text_len = sum(len(a.get_text(strip=True)) for a in links)

            # If > 60% of text is inside links, it's likely navigation/sidebar/related links
            # Only apply this if the text is short (< 500 chars) to avoid deleting
            # a body paragraph that happens to have many citations.
            if total_len < 500 and (link_text_len / total_len) > 0.6:
                container.decompose()

    def _extract_article_text(self, element: Tag, title: Optional[str] = None) -> str:
        """Smart extraction of paragraphs."""
        lines = []

        # Add title if provided
        if title:
            lines.append(title)

        # Iterate over significant block tags
        for tag in element.find_all(self.TEXT_BLOCK_TAGS):
            # Clean text
            text = tag.get_text(separator=" ", strip=True)
            if len(text) < 15:  # Skip artifacts, page numbers, etc.
                continue

            # Skip if this block is identical to the title (deduplication)
            if title and text.lower() == title.lower():
                continue

            lines.append(text)

        # Fallback: If structure was flattened or selectors missed, get all text
        if len(lines) <= 1:
            raw_text = element.get_text(separator="\n", strip=True)
            # Basic dedupe of title
            if title and raw_text.startswith(title):
                raw_text = raw_text[len(title):].strip()
                lines.append(raw_text)
            else:
                return raw_text

        return "\n\n".join(lines)

    def _normalize_text(self, text: str) -> str:
        """Normalize unicode and whitespace."""
        # NFKC normalizes compatibility characters (like \xa0 becomes space)
        text = unicodedata.normalize("NFKC", text)

        # Collapse multiple newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)

        # Collapse multiple spaces
        text = re.sub(r'[ \t]+', ' ', text)

        return text.strip()

    def truncate_text(self, text: str, max_length: int = 32000) -> Tuple[str, bool]:
        """
        Truncate text ensuring sentence boundaries are respected.
        """
        if len(text) <= max_length:
            return text, False

        # Take a slice slightly larger to look for boundary
        snippet = text[:max_length]

        # Try to cut at the last paragraph or sentence ending
        for separator in ["\n\n", "\n", ". ", "? ", "! "]:
            last_pos = snippet.rfind(separator)
            # If we find a separator within the last 10% of the allowed length, cut there
            if last_pos > max_length * 0.9:
                return snippet[:last_pos + len(separator)].strip(), True

        # Hard cut if no punctuation found
        return snippet, True

    @abstractmethod
    def reset(self) -> None:
        """Reset the extractor to initial state."""
        pass
