"""Shared LLM client with optimized HTTP connection pooling.

This module provides a singleton LLM client that reuses HTTP connections
and applies optimized settings for better performance.
"""

from typing import Optional, Union
import httpx
from langchain_openai import ChatOpenAI

from ..config.settings import settings
from .logger import get_logger

# Try to import Google Gemini integration
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    ChatGoogleGenerativeAI = None

logger = get_logger(__name__)


class LLMClientManager:
    """
    Manager for shared LLM clients with optimized HTTP connection pooling.
    
    Uses singleton pattern to ensure HTTP connections are reused across
    all agent instances, reducing connection overhead and improving performance.
    """
    
    _instance = None
    _http_client: Optional[httpx.Client] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMClientManager, cls).__new__(cls)
        return cls._instance
    
    def get_http_client(self, proxy: Optional[str] = None) -> httpx.Client:
        """
        Get or create the shared HTTP client with optimized settings.
        
        Args:
            proxy: Optional HTTP proxy URL (e.g., 'http://127.0.0.1:7890').
                   Note: This parameter is deprecated. Please use HTTP_PROXY and 
                   HTTPS_PROXY environment variables instead for better compatibility.
        
        Returns:
            Configured httpx.Client instance
        """
        # Proxy configuration is now handled via environment variables (HTTP_PROXY, HTTPS_PROXY)
        # to ensure compatibility across different versions of httpx and OpenAI SDK.
        # The trust_env=True parameter enables httpx to respect these environment variables.
        if proxy:
            logger.warning(
                "Proxy parameter is deprecated. Please use HTTP_PROXY and HTTPS_PROXY "
                "environment variables instead. The proxy parameter will be ignored."
            )
        
        # Check if optimized mode is disabled
        if not settings.use_optimized_mode or not settings.use_shared_http_client:
            logger.debug("Creating new HTTP client (optimized mode disabled)")
            client_kwargs = {
                "timeout": settings.openai_timeout,
                "trust_env": True  # Enable environment variable support for proxies
            }
            return httpx.Client(**client_kwargs)
        
        if self._http_client is None:
            logger.info("Creating shared HTTP client with optimized connection pooling")
            client_kwargs = {
                "limits": httpx.Limits(
                    max_connections=settings.openai_max_connections,
                    max_keepalive_connections=settings.openai_max_keepalive_connections,
                    keepalive_expiry=settings.openai_keepalive_expiry
                ),
                "timeout": settings.openai_timeout,
                "trust_env": True  # Enable environment variable support for proxies
            }
            
            self._http_client = httpx.Client(**client_kwargs)
            logger.debug(
                f"HTTP client configured: "
                f"max_connections={settings.openai_max_connections}, "
                f"max_keepalive={settings.openai_max_keepalive_connections}, "
                f"keepalive_expiry={settings.openai_keepalive_expiry}s, "
                f"trust_env=True (proxy via environment variables)"
            )
        return self._http_client
    
    def get_llm_client(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None
    ) -> Union[ChatOpenAI, "ChatGoogleGenerativeAI"]:
        """
        Get or create a shared LLM client.

        Args:
            model_name: Model name (defaults to settings)
            temperature: Temperature (defaults to settings)
            api_key: API key (defaults to settings)
            api_base: API base URL (defaults to settings)

        Returns:
            Configured LLM client (ChatOpenAI for OpenAI-compatible APIs,
            ChatGoogleGenerativeAI for official Google Gemini API)
        """
        # Use defaults from settings if not provided
        model_name = model_name or settings.openai_model
        temperature = temperature if temperature is not None else settings.agent_temperature
        api_key = api_key or settings.openai_api_key
        api_base = api_base or settings.openai_api_base

        logger.info(f"Creating LLM client: model={model_name}, temperature={temperature}")

        # Check if this is a Google Gemini model AND using Google's official API
        # Only use Google integration when the API base is Google's official endpoint
        is_google_official = api_base and "googleapis.com" in api_base
        is_gemini_model = "gemini" in model_name.lower()

        if GOOGLE_AVAILABLE and is_gemini_model and is_google_official:
            logger.info(f"Using Google Gemini integration for model: {model_name}")
            llm_client = ChatGoogleGenerativeAI(
                model=model_name,
                api_key=api_key,
                temperature=temperature,
                max_retries=settings.openai_max_retries,
                timeout=settings.openai_timeout
            )
        else:
            # Use OpenAI-compatible client for custom API endpoints or non-Gemini models
            if is_gemini_model and not is_google_official:
                logger.info(f"Using OpenAI-compatible client for custom API endpoint: {model_name} -> {api_base}")
            # Create http_client with proxy configured via environment variables
            http_client = self.get_http_client()

            llm_client = ChatOpenAI(
                model_name=model_name,
                api_key=api_key,
                base_url=api_base,
                temperature=temperature,
                max_retries=settings.openai_max_retries,
                timeout=settings.openai_timeout,
                http_client=http_client
            )

        logger.debug("LLM client created")
        return llm_client
    
    def close(self):
        """Close the HTTP client and release resources."""
        if self._http_client is not None:
            logger.info("Closing shared HTTP client")
            self._http_client.close()
            self._http_client = None


# Global singleton instance
llm_manager = LLMClientManager()
