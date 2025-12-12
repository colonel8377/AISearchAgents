"""Authentication middleware for API security."""

from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from ..config.settings import settings


# API Key header definition
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """
    Verify the API key from the request header.
    
    Args:
        api_key: API key from the X-API-Key header
        
    Returns:
        The validated API key
        
    Raises:
        HTTPException: If API key is missing or invalid
    """
    # If no API key is configured, skip authentication
    if not settings.api_key_required or not settings.api_keys:
        return "no-auth-required"
    
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is required. Please provide X-API-Key header."
        )
    
    # Check if the provided key is in the list of valid keys
    if api_key not in settings.api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return api_key
