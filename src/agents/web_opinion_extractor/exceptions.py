"""Custom exceptions for Web Opinion Extractor."""


class WebExtractionError(Exception):
    """Base exception for web extraction errors."""
    
    def __init__(self, message: str, url: str = ""):
        self.url = url
        self.message = message
        super().__init__(self.message)


class NetworkError(WebExtractionError):
    """Exception raised for network-related errors (timeouts, 404, connection issues)."""
    
    def __init__(self, message: str, url: str = "", status_code: int = None):
        self.status_code = status_code
        super().__init__(message, url)


class ContentExtractionError(WebExtractionError):
    """Exception raised when content cannot be extracted from HTML."""
    pass
