"""
Shared LLM client with optimized HTTP persistence pooling.

This module provides a singleton LLM client that reuses HTTP connections
and applies optimized settings for better performance.
"""

import atexit
import threading
from typing import Optional, Union, Any
import httpx
from langchain_openai import ChatOpenAI

from ...shared.config.settings import settings
from ...shared.utils.logger import get_logger

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    ChatGoogleGenerativeAI = Any


logger = get_logger(__name__)


class LLMClientManager:
    """
    Manager for shared LLM clients with optimized HTTP persistence pooling.
    
    Uses singleton pattern to ensure HTTP connections are reused across
    all agent instances, reducing persistence overhead and improving performance.
    """
    
    _instance = None
    _lock = threading.Lock()
    _http_client: Optional[httpx.Client] = None
    
    def __new__(cls):
        """Create singleton instance with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(LLMClientManager, cls).__new__(cls)
                    atexit.register(cls._instance.close)
        return cls._instance
    
    def _create_http_client(self, use_pooling: bool = True) -> httpx.Client:
        """
        Create HTTP client with appropriate configuration.
        
        Args:
            use_pooling: Whether to use persistence pooling
            
        Returns:
            Configured httpx.Client instance
        """
        if use_pooling:
            logger.info(
                "Creating shared HTTP client with optimized persistence pooling"
            )
            client_kwargs = {
                "limits": httpx.Limits(
                    max_connections=settings.openai_max_connections,
                    max_keepalive_connections=settings.openai_max_keepalive_connections,
                    keepalive_expiry=settings.openai_keepalive_expiry
                ),
                "timeout": settings.openai_timeout,
                "trust_env": True  # Enable environment variable support for proxies
            }
            logger.debug(
                f"HTTP client configured: "
                f"max_connections={settings.openai_max_connections}, "
                f"max_keepalive={settings.openai_max_keepalive_connections}, "
                f"keepalive_expiry={settings.openai_keepalive_expiry}s, "
                f"trust_env=True (proxy via environment variables)"
            )
        else:
            logger.debug("Creating new HTTP client (optimized mode disabled)")
            client_kwargs = {
                "timeout": settings.openai_timeout,
                "trust_env": True
            }
        
        return httpx.Client(**client_kwargs)
    
    def get_http_client(self, proxy: Optional[str] = None) -> httpx.Client:
        """
        Get or create the shared HTTP client with optimized settings.
        
        Args:
            proxy: Optional HTTP proxy URL (deprecated).
                   Use HTTP_PROXY and HTTPS_PROXY environment variables instead.
        
        Returns:
            Configured httpx.Client instance
        """
        # Warn about deprecated proxy parameter
        if proxy:
            logger.warning(
                "Proxy parameter is deprecated. Please use HTTP_PROXY and "
                "HTTPS_PROXY environment variables instead. "
                "The proxy parameter will be ignored."
            )
        
        # Check if optimized mode is disabled
        use_pooling = (
            settings.use_optimized_mode and
            settings.use_shared_http_client
        )
        
        if not use_pooling:
            return self._create_http_client(use_pooling=False)
        
        # Create or return shared client
        if self._http_client is None:
            with self._lock:
                if self._http_client is None:
                    self._http_client = self._create_http_client(use_pooling=True)

        return self._http_client
    
    def _should_use_google_integration(
        self,
        model_name: str,
        api_base: Optional[str]
    ) -> bool:
        """
        Determine if Google Gemini integration should be used.
        
        Args:
            model_name: Model name
            api_base: API base URL
            
        Returns:
            True if Google integration should be used
        """
        is_google_official = api_base and "googleapis.com" in api_base
        is_gemini_model = "gemini" in model_name.lower()
        return GOOGLE_AVAILABLE and is_gemini_model and is_google_official
    
    def _create_google_llm_client(
        self,
        model_name: str,
        api_key: str,
        temperature: float
    ) -> "ChatGoogleGenerativeAI":
        """
        Create Google Gemini LLM client.
        
        Args:
            model_name: Model name
            api_key: API key
            temperature: Temperature setting
            
        Returns:
            Configured ChatGoogleGenerativeAI instance
        """
        if not GOOGLE_AVAILABLE:
            raise ImportError(
                "langchain_google_genai is required for Google Gemini integration"
            )
        
        logger.info(f"Using Google Gemini integration for model: {model_name}")
        return ChatGoogleGenerativeAI(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            max_retries=settings.openai_max_retries,
            timeout=settings.openai_timeout
        )
    
    def _create_openai_llm_client(
        self,
        model_name: str,
        api_key: str,
        api_base: Optional[str],
        temperature: float
    ) -> ChatOpenAI:
        """
        Create OpenAI-compatible LLM client.
        
        Args:
            model_name: Model name
            api_key: API key
            api_base: API base URL
            temperature: Temperature setting
            
        Returns:
            Configured ChatOpenAI instance
        """
        http_client = self.get_http_client()
        
        # Log if using custom endpoint for Gemini
        if "gemini" in model_name.lower() and api_base and "googleapis.com" not in api_base:
            logger.info(
                f"Using OpenAI-compatible client for custom API endpoint: "
                f"{model_name} -> {api_base}"
            )
        
        return ChatOpenAI(
            model_name=model_name,
            api_key=api_key,
            base_url=api_base,
            temperature=temperature,
            max_retries=settings.openai_max_retries,
            timeout=settings.openai_timeout,
            http_client=http_client
        )
    
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
        temperature = (
            temperature if temperature is not None
            else settings.agent_temperature
        )
        api_key = api_key or settings.openai_api_key
        api_base = api_base or settings.openai_api_base

        logger.info(
            f"Creating LLM client: model={model_name}, "
            f"temperature={temperature}"
        )

        # Choose appropriate client based on model and API base
        if self._should_use_google_integration(model_name, api_base):
            llm_client = self._create_google_llm_client(
                model_name,
                api_key,
                temperature
            )
        else:
            llm_client = self._create_openai_llm_client(
                model_name,
                api_key,
                api_base,
                temperature
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