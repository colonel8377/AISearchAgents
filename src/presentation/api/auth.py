"""Authentication middleware for API security."""

from typing import Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from src.shared.config.settings import settings

# API Key header definition
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """
    Verify the API key from the request header.
    
    Authentication logic:
    1. If api_key_required is False: Allow all requests (no authentication required)
    2. If api_key_required is True but no api_keys configured: Allow all requests (no keys to validate)
    3. If api_key_required is True and api_keys are configured: Require valid API key
    
    Args:
        api_key: API key from the X-API-Key header (optional)
        
    Returns:
        The validated API key or "no-auth-required" if authentication is disabled
        
    Raises:
        HTTPException: If API key is required but missing or invalid
    """
    # Case 1: Authentication is disabled
    if not settings.api_key_required:
        return "no-auth-required"
    
    # Case 2: Authentication is enabled but no keys are configured
    if not settings.api_keys:
        # If no keys configured, allow all requests
        return "no-auth-required"
    
    # Case 3: Authentication is enabled and keys are configured - require valid key
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is required. Please provide X-API-Key header."
        )
    
    # Validate the provided key
    if api_key not in settings.api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return api_key
