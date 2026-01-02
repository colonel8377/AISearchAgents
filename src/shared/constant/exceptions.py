"""Common exceptions for the application."""

class BaseError(Exception):
    """Base exception for the application."""
    pass


class WebExtractionError(BaseError):
    """Base exception for web extraction errors."""
    
    def __init__(self, message: str, url: str = ""):
        self.url = url
        self.message = message
        super().__init__(self.message)


class NetworkError(BaseError):
    """Exception raised for network-related errors (timeouts, 404, persistence issues)."""
    
    def __init__(self, message: str, url: str = "", status_code: int = None):
        self.status_code = status_code
        super().__init__(message, url)


class ContentExtractionError(BaseError):
    """Exception raised when content cannot be extracted from HTML."""

    def __init__(self, message: str, url: str = "", status_code: int = None):
        self.status_code = status_code
        super().__init__(message, url)

