from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class PrivacyEntity:
    text: str
    entity_type: str
    start: int
    end: int
    confidence: float
    category: str = "UNKNOWN"
    severity: str = "low"
    source: str = "detector"
    # --- 新增这个字段 ---
    metadata: Dict[str, Any] = field(default_factory=dict)