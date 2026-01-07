from dataclasses import dataclass
from typing import Optional, Any, Dict


@dataclass
class ContentExtractionResult:
    """Result of content extraction."""
    url: Optional[str] = None
    title: Optional[str] = None
    main_body: Optional[str] = None
    text_length: int = 0
    truncated: bool = False
    extraction_metadata: Optional[Dict[str, Any]] = None
