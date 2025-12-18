"""Pydantic models for Web Opinion Extractor."""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, model_validator


class BiasDistribution(BaseModel):
    """
    Probability distribution of political bias across three categories.
    
    The probabilities should sum to 1.0 (or close to it within tolerance).
    If they don't sum to 1.0, they are automatically normalized.
    """
    left: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of Left/Progressive bias (0.0 to 1.0)"
    )
    right: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of Right/Conservative bias (0.0 to 1.0)"
    )
    neutral: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of Neutral/Centrist bias (0.0 to 1.0)"
    )
    
    @model_validator(mode='before')
    @classmethod
    def normalize_probabilities(cls, data):
        """Normalize probabilities to sum to 1.0 before validation."""
        if isinstance(data, dict):
            left = data.get('left', 0.0)
            right = data.get('right', 0.0)
            neutral = data.get('neutral', 0.0)
            
            total = left + right + neutral
            if total > 0 and not (0.99 <= total <= 1.01):
                # Normalize to sum to 1.0
                data = {
                    'left': left / total,
                    'right': right / total,
                    'neutral': neutral / total
                }
        return data
    
    @property
    def dominant_bias(self) -> str:
        """Return the dominant bias category."""
        if self.left >= self.right and self.left >= self.neutral:
            return "left"
        elif self.right >= self.left and self.right >= self.neutral:
            return "right"
        else:
            return "neutral"
    
    @property
    def bias_score(self) -> float:
        """
        Convert to a single bias score for backward compatibility.
        Returns a value from -1.0 (left) to +1.0 (right).
        """
        # Weight: left contributes -1, neutral contributes 0, right contributes +1
        return -1.0 * self.left + 0.0 * self.neutral + 1.0 * self.right


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
    bias_probabilities: BiasDistribution = Field(
        ...,
        description="Probability distribution of political bias: left (Progressive), right (Conservative), neutral"
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
    reasoning: Optional[str] = Field(
        default=None,
        description="Chain of Thought (CoT) reasoning explaining the analysis step-by-step"
    )
    
    @property
    def bias_score(self) -> float:
        """
        Get a single bias score for backward compatibility.
        Returns a value from -1.0 (left) to +1.0 (right).
        """
        return self.bias_probabilities.bias_score


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
    overall_bias_distribution: Optional[BiasDistribution] = Field(
        default=None,
        description="Aggregated bias distribution across all opinions"
    )
    text_length: int = Field(default=0, description="Length of the cleaned text")
    truncated: bool = Field(default=False, description="Whether the text was truncated")
    extraction_metadata: Optional[dict] = Field(
        default=None,
        description="Additional metadata about the extraction process"
    )
    
    def calculate_overall_bias(self) -> Optional[BiasDistribution]:
        """Calculate the overall bias distribution as average of opinion bias distributions."""
        if not self.opinions:
            return None
        
        total_left = sum(op.bias_probabilities.left for op in self.opinions)
        total_right = sum(op.bias_probabilities.right for op in self.opinions)
        total_neutral = sum(op.bias_probabilities.neutral for op in self.opinions)
        
        n = len(self.opinions)
        return BiasDistribution(
            left=total_left / n,
            right=total_right / n,
            neutral=total_neutral / n
        )


# ========== NEW MODELS FOR WebOpinionEngine ==========

class ArticleContent(BaseModel):
    """
    Represents extracted article content.
    Output of Agent 1 (extract_content).
    """
    title: Optional[str] = Field(default=None, description="Article title")
    full_text: str = Field(..., description="Complete extracted text (no truncation)")
    domain: str = Field(..., description="Domain extracted from URL")
    url: Optional[str] = Field(default=None, description="Source URL")


class SourceMetadata(BaseModel):
    """
    Represents metadata about the news source from MBFC database.
    Output of Agent 2 (resolve_metadata).
    """
    source_name: Optional[str] = Field(default=None, description="Name of the news source")
    raw_db_row: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Raw database row data if found"
    )
    match_type: Literal["exact", "fuzzy", "none", "disabled"] = Field(
        ...,
        description="Type of match: exact (1 row), fuzzy (>1 rows), none (0 rows), disabled (use_mbfc=False)"
    )
    bias_rating: Optional[str] = Field(
        default=None,
        description="Historical bias rating from database (e.g., 'left', 'right', 'center')"
    )
    factual_reporting: Optional[str] = Field(
        default=None,
        description="Factual reporting rating from database"
    )


class AtomicUnit(BaseModel):
    """
    Represents an atomic unit of fact or opinion.
    Output of Agent 3 (atomize_text).
    Extends AtomicOpinion with additional fields.
    """
    statement: str = Field(..., description="The atomic statement text")
    type: Literal["fact", "opinion"] = Field(
        ...,
        description="Whether this is an objective fact or subjective opinion"
    )
    original_sentence: Optional[str] = Field(
        default=None,
        description="The original sentence from which this was extracted"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for the extraction (0.0 to 1.0)"
    )


class BiasResult(BaseModel):
    """
    Result of bias calculation with Bayesian scoring.
    Output of Agent 4 (calculate_bias).
    """
    bias_distribution: BiasDistribution = Field(
        ...,
        description="Probability distribution of political bias"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Chain of Thought reasoning for the bias calculation"
    )
    metadata_used: bool = Field(
        ...,
        description="Whether MBFC metadata was used as prior probability"
    )
    individual_biases: Optional[List[BiasDistribution]] = Field(
        default=None,
        description="Individual bias distributions for each atomic unit"
    )

