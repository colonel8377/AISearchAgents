"""Core interfaces for privacy detection system.

This module defines abstract base classes (ABCs) for the privacy detection
architecture, enabling dependency injection and clean separation of concerns.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

from src.application.agents.privacy_detector.core.model import PrivacyEntity


class IPrivacyDetector(ABC):
    """Interface for privacy detection engines."""
    
    @abstractmethod
    def detect(self, text: str) -> List[PrivacyEntity]:
        """
        Detect privacy entities in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected PrivacyEntity objects
        """
        pass


class ISanitizer(ABC):
    """Interface for text sanitization."""
    
    @abstractmethod
    def sanitize(
        self, 
        text: str, 
        entities: List[PrivacyEntity]
    ) -> Tuple[str, Dict[str, Dict[str, Any]]]:
        """
        Sanitize text by replacing privacy entities with synthetic data.
        
        This method generates synthetic replacements (e.g., fake names, emails)
        for use in LLM processing. It maintains consistency within a session.
        
        Args:
            text: Original text containing privacy entities
            entities: List of detected privacy entities
            
        Returns:
            Tuple of (sanitized_text, registry) where:
            - sanitized_text: Text with entities replaced with synthetic data
            - registry: Dict mapping original values to metadata
        """
        pass
    
    @abstractmethod
    def mask(
        self,
        text: str,
        entities: List[PrivacyEntity]
    ) -> str:
        """
        Mask privacy entities in text for UI display.
        
        This method performs pure string manipulation (e.g., "138****0000")
        without calling any external APIs. It is distinct from sanitize()
        which generates synthetic data for LLM processing.
        
        Args:
            text: Original text containing privacy entities
            entities: List of detected privacy entities
            
        Returns:
            Masked text with entities replaced using asterisks/patterns
            (e.g., "138****0000" for phone numbers, "j***@example.com" for emails)
        """
        pass


