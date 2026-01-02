"""Internal models for Web Opinion Extractor engine.

These models are used internally by WebOpinionEngine and are not exposed to the API layer.
For API-facing models (BiasDistribution, AtomicOpinion, OpinionExtractionResult), see src/api/schemas.py.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


# Re-export API contract models for backward compatibility
__all__ = ['BiasDistribution', 'AtomicOpinion', 'OpinionExtractionResult']

from src.presentation.api.schemas import BiasDistribution, AtomicOpinion, OpinionExtractionResult


# ========== NEW MODELS FOR WebOpinionEngine ==========



class PipelineConfig(BaseModel):
    """
    Configuration for the WebOpinionEngine pipeline.
    
    Stores OpenAI model settings, persistence path, and paths to JSON resource files.
    """
    openai_model: str = Field(
        default="gpt-3.5-turbo",
        description="OpenAI model name"
    )
    mbfc_db_path: Optional[str] = Field(
        default=None,
        description="Path to MBFC SQLite persistence (optional, defaults to settings.mbfc_db_path)"
    )
    atomizer_shots_path: Optional[str] = Field(
        default=None,
        description="Path to JSON file with atomizer few-shot examples (default: src/few_shots/web_opinion_extractor/shots/atomizer_shots.json)"
    )
    scorer_shots_path: Optional[str] = Field(
        default=None,
        description="Path to JSON file with scorer few-shot examples (default: src/few_shots/web_opinion_extractor/shots/scorer_shots.json)"
    )


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
    Represents metadata about the news source from MBFC persistence.
    Output of Agent 2 (resolve_metadata).
    """
    source_name: Optional[str] = Field(default=None, description="Name of the news source")
    raw_db_row: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Raw persistence row data if found"
    )
    match_type: Literal["exact", "fuzzy", "none", "disabled"] = Field(
        ...,
        description="Type of match: exact (1 row), fuzzy (>1 rows), none (0 rows), disabled (use_mbfc=False)"
    )
    bias_rating: Optional[str] = Field(
        default=None,
        description="Historical bias rating from persistence (e.g., 'left', 'right', 'center')"
    )
    factual_reporting: Optional[str] = Field(
        default=None,
        description="Factual reporting rating from persistence"
    )


class AtomicUnit(BaseModel):
    """
    Represents an atomic unit of fact or opinion.
    Output of Agent 3 (atomize_text).
    
    This is a lightweight version of AtomicOpinion used inside the
    WebOpinionEngine pipeline. It mirrors the most important fields
    (statement/text, type, original_sentence, confidence, reasoning)
    so that high-level APIs like the /analyze endpoint can expose
    rich per-opinion metadata.
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
    reasoning: Optional[str] = Field(
        default=None,
        description="Chain of Thought reasoning for this specific atomic unit"
    )


class BiasResult(BaseModel):
    """
    Result of bias calculation with Bayesian scoring.
    Output of Agent 4 (calculate_bias).
    
    Internal model used by WebOpinionEngine.
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
    mbfc_influence_note: Optional[str] = Field(
        default=None,
        description="Brief note describing how much the MBFC prior influenced the final bias assessment (e.g., 'strong', 'moderate', 'weak', 'overridden')"
    )
    individual_biases: Optional[List[BiasDistribution]] = Field(
        default=None,
        description="Individual bias distributions for each atomic unit"
    )

