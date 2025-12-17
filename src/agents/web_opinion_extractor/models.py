"""Pydantic models for Web Opinion Extractor."""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class AtomicOpinion(BaseModel):
    """
    Represents a single atomic opinion extracted from web content.
    
    Atomic opinions are the smallest unit of opinion that cannot be
    further broken down. Compound sentences are split into multiple
    atomic opinions.
    """
    text: str = Field(..., description="The atomic opinion text")
    opinion_type: Literal["fact", "opinion"] = Field(
        ...,
        description="Whether this is an objective fact or subjective opinion"
    )
    bias_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Political bias score: -1.0 (Left/Progressive) to +1.0 (Right/Conservative), 0.0 is Neutral"
    )
    original_sentence: Optional[str] = Field(
        default=None,
        description="The original sentence from which this atomic opinion was extracted"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for the extraction (0.0 to 1.0)"
    )


class OpinionExtractionResult(BaseModel):
    """
    Result of opinion extraction from web content.
    
    Contains all atomic opinions extracted from the content along with
    metadata about the extraction process.
    """
    url: Optional[str] = Field(default=None, description="Source URL of the content")
    title: Optional[str] = Field(default=None, description="Title of the web page")
    atomic_opinions: List[AtomicOpinion] = Field(
        default_factory=list,
        description="List of atomic opinions extracted from the content"
    )
    facts: List[AtomicOpinion] = Field(
        default_factory=list,
        description="List of objective facts (filtered from atomic_opinions)"
    )
    opinions: List[AtomicOpinion] = Field(
        default_factory=list,
        description="List of subjective opinions (filtered from atomic_opinions)"
    )
    overall_bias_score: Optional[float] = Field(
        default=None,
        description="Average bias score across all opinions"
    )
    text_length: int = Field(default=0, description="Length of the cleaned text")
    truncated: bool = Field(default=False, description="Whether the text was truncated")
    extraction_metadata: Optional[dict] = Field(
        default=None,
        description="Additional metadata about the extraction process"
    )
    
    def calculate_overall_bias(self) -> float:
        """Calculate the overall bias score as average of opinion bias scores."""
        opinions_with_bias = [op for op in self.opinions if op.bias_score is not None]
        if not opinions_with_bias:
            return 0.0
        return sum(op.bias_score for op in opinions_with_bias) / len(opinions_with_bias)
