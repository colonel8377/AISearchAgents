"""API request and response schemas."""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, model_validator

from ..agents.web_opinion_extractor import BiasDistribution, AtomicOpinion
from ..config.settings import ExecutionMode
from ..debate.schemas import PersonaConfig, AgentMetadata, RoundReasoningResponse, RoundVotingResponse

class CreateAgentRequest(BaseModel):
    """Request model for creating a new agent instance."""
    agent_type: str = Field(..., description="Type of agent: 'nudge_collapse', 'summarizer', 'bot_creator', or 'demographic_evaluator'")
    agent_id: Optional[str] = Field(default=None, description="Custom agent ID (auto-generated if not provided)")
    use_memory: bool = Field(default=False, description="Whether to use vector memory for this agent")
    persona_mode: Optional[str] = Field(default="system_prompt", description="Persona mode for bot_creator: 'system_prompt' or 'user_instruction'")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "agent_type": "nudge_collapse",
                    "use_memory": True
                }
            ]
        }
    }


class AgentIdResponse(BaseModel):
    """Response model for agent creation."""
    agent_id: str
    agent_type: str
    status: str
    message: str
    persona_mode: Optional[str] = None


class GenerateTurnRequest(BaseModel):
    """Request model for generating a turn (nudge_collapse agent)."""
    user_query: str = Field(..., description="User's query or input")
    search_summary: str = Field(default="", description="Summary from search engine")
    search_urls: Optional[List[str]] = Field(default=None, description="URLs from search results")
    history_mode: Optional[bool] = Field(
        default=None,
        description="History mode: True (include conversation history), False (stateless). Defaults to system setting."
    )
    use_few_shots: bool = Field(
        default=True,
        description="Whether to use few-shot examples in the prompt (default: True)"
    )
    custom_few_shots: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional custom few-shot examples. Should be a dict with keys 'turn_0', 'turn_1', 'turn_2', 'turn_3'. If provided, use_few_shots must be True."
    )


class ResetAgentRequest(BaseModel):
    """Request model for resetting an agent."""
    reset_conversation: bool = Field(default=True, description="Whether to reset conversation history")
    clear_memory: bool = Field(default=False, description="Whether to clear vector memory")


class SetCustomFewShotsRequest(BaseModel):
    """Request model for setting custom few-shot examples."""
    custom_few_shots: Optional[Union[str, Dict[str, str], Dict[str, List[dict]]]] = Field(
        default=None,
        description="Custom few-shot examples. For nudge-collapse: dict with keys 'turn_0', 'turn_1', 'turn_2', 'turn_3'. For web opinion extractor: dict with 'atomizer_shots' and 'scorer_shots' keys containing lists. For others: string. Set to null to clear custom shots."
    )


class TurnResponse(BaseModel):
    """Response model for a turn."""
    turn: int
    query: str
    response: str
    search_summary: str
    search_urls: List[str]
    strategy: str
    history_mode: Optional[bool] = None


class ConversationHistoryResponse(BaseModel):
    """Response model for conversation history."""
    current_turn: int
    max_turns: int
    history: List[Dict[str, Any]]


class EvaluateSentencesRequest(BaseModel):
    """Request model for evaluating sentences."""
    demography_json: Dict[str, Any] = Field(..., description="Demographic profile as JSON object")
    sentences: List[str] = Field(..., description="List of sentences to evaluate")
    use_cot: bool = Field(default=False, description="Whether to use Chain of Thought reasoning (default: False)")
    execution_mode: Optional[ExecutionMode] = Field(
        default=None,
        description="Execution mode for CoT: 'chain_online', 'chain_local', or 'no_chain'. If provided, overrides use_cot parameter."
    )
    use_few_shots: bool = Field(
        default=True,
        description="Whether to use few-shot examples in the prompt (default: True)"
    )
    custom_few_shots: Optional[str] = Field(
        default=None,
        description="Optional custom few-shot examples to use instead of defaults. If provided, use_few_shots must be True."
    )
    
    @model_validator(mode='after')
    def validate_custom_few_shots(self):
        """Ensure that if custom_few_shots is provided, use_few_shots must be True."""
        if self.custom_few_shots is not None and not self.use_few_shots:
            raise ValueError("If custom_few_shots is provided, use_few_shots must be True")
        return self


class JudgmentResponse(BaseModel):
    """Response model for a single judgment."""
    index: int
    sentence: str
    agree: int = Field(..., ge=0, le=1, description="1 for agree, 0 for disagree")
    reason: str


class EvaluateSentencesResponse(BaseModel):
    """Response model for sentence evaluation."""
    judgments: List[JudgmentResponse]


class AgentStatusResponse(BaseModel):
    """Response model for agent status."""
    agent_id: str
    agent_type: str
    status: str
    current_turn: Optional[int] = None
    additional_info: Optional[Dict[str, Any]] = None


class SummarizeRequest(BaseModel):
    """Request model for summarizing conversations."""
    conversation_records: List[Dict[str, str]] = Field(..., description="List of conversation records to summarize")
    execution_mode: Optional[ExecutionMode] = Field(
        default=None,
        description="Execution mode: 'chain_online' (LLM chains), 'chain_local' (local decomposition), 'no_chain' (pure prompt). Defaults to system setting."
    )
    use_few_shots: bool = Field(
        default=True,
        description="Whether to use few-shot examples in the prompt (default: True)"
    )
    custom_few_shots: Optional[str] = Field(
        default=None,
        description="Optional custom few-shot examples to use instead of defaults. If provided, use_few_shots must be True."
    )


class SummaryResponse(BaseModel):
    """Response model for summary."""
    summary: str
    conversation_length: int
    original_length: int
    truncated: bool
    execution_mode: str
    metadata: Dict[str, Any]


class CreateBotRequest(BaseModel):
    """Request model for creating a bot."""
    persona_prompt: str = Field(..., description="Persona prompt corpus for the bot")
    bot_name: Optional[str] = Field(default=None, description="Optional name for the bot")
    execution_mode: Optional[ExecutionMode] = Field(
        default=None,
        description="Execution mode: 'chain_online' (LLM chains), 'chain_local' (local decomposition), 'no_chain' (pure prompt). Defaults to system setting."
    )
    use_few_shots: bool = Field(
        default=True,
        description="Whether to use few-shot examples in the prompt (default: True)"
    )
    custom_few_shots: Optional[str] = Field(
        default=None,
        description="Optional custom few-shot examples to use instead of defaults. If provided, use_few_shots must be True."
    )


class BotCreationResponse(BaseModel):
    """Response model for bot creation."""
    bot_id: str
    bot_name: str
    status: str
    persona_prompt: str
    persona_mode: Optional[str] = None
    execution_mode: Optional[str] = None
    bot_configuration: str
    message: str


class ListAgentsResponse(BaseModel):
    """Response model for listing agents."""
    agents: List[Dict[str, str]]
    total_count: int


class ExtractContentRequest(BaseModel):
    """Request model for content extraction."""
    url: Optional[str] = Field(default=None, description="URL to extract content from")
    html: Optional[str] = Field(default=None, description="Raw HTML to extract content from")
    text: Optional[str] = Field(default=None, description="Plain text to process (alternative to URL/HTML)")
    title: Optional[str] = Field(default=None, description="Optional title (used when text is provided)")
    summary: Optional[str] = Field(default=None, description="Optional summary text for claim-level comparison with URL content")
    use_llm: bool = Field(default=False, description="Whether to use LLM for content refinement")
    use_cot: bool = Field(default=False, description="Whether to use Chain of Thought reasoning (only when use_llm=True)")
    custom_few_shots: Optional[str] = Field(default=None, description="Optional custom few-shot examples (only when use_llm=True)")
    compare_claims: bool = Field(default=False, description="Whether to perform claim-level comparison (requires summary to be provided)")

    @model_validator(mode='after')
    def validate_input_source(self):
        """Ensure exactly one input source is provided."""
        sources = [self.url, self.html, self.text]
        provided_sources = [s for s in sources if s is not None]
        if len(provided_sources) != 1:
            raise ValueError("Exactly one of 'url', 'html', or 'text' must be provided")
        if self.compare_claims and not self.summary:
            raise ValueError("summary must be provided when compare_claims is True")
        return self


class AtomicClaimResponse(BaseModel):
    """Response model for a single atomic claim."""
    id: str
    text: str
    original_sentence: str
    confidence: float
    paragraph_index: Optional[int] = None
    paragraph_text: Optional[str] = None


class ClaimComparisonResult(BaseModel):
    """Response model for a single claim comparison.
    
    Evaluates whether URL content agrees, disagrees, or has no relevant claim for each summary claim.
    Includes detailed reasoning for academic rigor.
    """
    summary_claim_id: str
    summary_claim_text: str
    url_claim_id: Optional[str] = Field(default=None, description="ID of the best matching URL claim (if found)")
    url_claim_text: Optional[str] = Field(default=None, description="Text of the best matching URL claim (if found)")
    relationship: str = Field(..., description="Overall relationship: 'agree' (URL agrees/supports), 'disagree' (URL disagrees/contradicts), 'missing' (no relevant claim)")
    similarity_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Semantic similarity score (0.0 to 1.0), only for agree/disagree")
    reasoning: str = Field(..., description="Detailed explanation: specific aspects of agreement/disagreement, why this relationship was determined, what evidence supports this conclusion")


class ClaimComparisonResponse(BaseModel):
    """Response model for comprehensive claim comparison between summary and URL content.
    
    For each claim in the summary, determines whether URL content agrees, disagrees, or has no relevant claim.
    Includes detailed statistics and categorized lists for academic analysis.
    """
    summary_claims: List[AtomicClaimResponse] = Field(..., description="All claims extracted from the summary")
    url_claims: List[AtomicClaimResponse] = Field(..., description="All claims extracted from the URL content")
    comparisons: List[ClaimComparisonResult] = Field(..., description="For each summary claim, its relationship to URL claims with detailed reasoning")
    statistics: Dict[str, Any] = Field(..., description="Comprehensive statistics including counts, rates, and categorized claim lists")


class ParagraphResponse(BaseModel):
    """Response model for a single paragraph."""
    index: int
    text: str
    text_length: int


class ContentExtractionResponse(BaseModel):
    """Response model for content extraction."""
    url: Optional[str] = None
    title: Optional[str] = None
    main_body: Optional[str] = None  # Keep for backward compatibility
    paragraphs: List[ParagraphResponse] = Field(default_factory=list, description="Content split into paragraphs")
    text_length: int
    truncated: bool
    extraction_metadata: Optional[Dict[str, Any]] = None
    claim_comparison: Optional[ClaimComparisonResponse] = Field(default=None, description="Claim-level comparison result (if compare_claims=True)")


class AtomizeClaimsRequest(BaseModel):
    """Request model for claim atomization."""
    text: str = Field(..., description="Text snippet to decompose into atomic claims")
    use_cot: bool = Field(default=False, description="Whether to use Chain of Thought reasoning")
    custom_few_shots: Optional[str] = Field(default=None, description="Optional custom few-shot examples")
    split_into_paragraphs: bool = Field(default=False, description="Whether to split text into paragraphs before atomization")


class ParagraphClaimsResponse(BaseModel):
    """Response model for claims from a single paragraph."""
    paragraph_index: int
    paragraph_text: str
    atomic_claims: List[AtomicClaimResponse]


class ClaimAtomizationResponse(BaseModel):
    """Response model for claim atomization."""
    atomic_claims: List[AtomicClaimResponse]  # Flat list for backward compatibility
    paragraphs: List[ParagraphClaimsResponse]  # New: claims grouped by paragraphs
    original_text: str
    execution_mode: str
    metadata: Optional[Dict[str, Any]] = None


class LocateEvidenceRequest(BaseModel):
    """Request model for evidence location."""
    claims: List[Dict[str, Any]] = Field(..., description="List of claims with 'id' and 'text' keys")
    main_body: str = Field(..., description="Main body text to search for evidence")
    use_llm: bool = Field(default=True, description="Whether to use LLM for evidence location")


class EvidenceQuoteResponse(BaseModel):
    """Response model for evidence quotes."""
    text: str
    location: str
    start_pos: int
    end_pos: int


class ClaimEvidenceResponse(BaseModel):
    """Response model for claim evidence."""
    claim_id: str
    claim_text: str
    evidence_found: bool
    quotes: List[EvidenceQuoteResponse]
    reasoning: str


class EvidenceLocationResponse(BaseModel):
    """Response model for evidence location."""
    claim_evidences: List[ClaimEvidenceResponse]
    main_body_text: str
    metadata: Optional[Dict[str, Any]] = None


class AuditConflictsRequest(BaseModel):
    """Request model for conflict auditing."""
    claim_evidences: List[Dict[str, Any]] = Field(..., description="List of claim-evidence pairs")
    use_cot: bool = Field(default=False, description="Whether to use Chain of Thought reasoning")
    custom_few_shots: Optional[str] = Field(default=None, description="Optional custom few-shot examples")


class ConflictAnalysisResponse(BaseModel):
    """Response model for conflict analysis."""
    claim_id: str
    claim_text: str
    evidence_quotes: List[str]
    verdict: str
    conflict_type: str
    analysis: str
    confidence: float


class ConflictAuditResponse(BaseModel):
    """Response model for conflict auditing."""
    conflict_analyses: List[ConflictAnalysisResponse]
    summary_stats: Dict[str, Any]
    execution_mode: str
    metadata: Optional[Dict[str, Any]] = None



class ChatWithBotRequest(BaseModel):
    """Request model for chatting with a bot."""
    bot_id: str = Field(..., description="UUID of the bot to chat with")
    message: str = Field(..., description="User message to send to the bot")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "bot_id": "550e8400-e29b-41d4-a716-446655440000",
                    "message": "What is machine learning?"
                }
            ]
        }
    }


class ChatWithBotResponse(BaseModel):
    """Response model for bot chat."""
    bot_id: str
    bot_name: str
    response: str
    conversation_history: List[Dict[str, str]]
    history_mode: str  # "stateless" or "stateful"


class DeleteConversationTurnRequest(BaseModel):
    """Request model for deleting a conversation turn."""
    turn_index: int = Field(..., ge=0, description="Index of the turn to delete (0-based, each turn is user+assistant)")


class AddTurnRequest(BaseModel):
    """Request model for adding a turn to a conversation."""
    user_message: str = Field(..., description="User's message in this turn")
    assistant_message: str = Field(..., description="Assistant's response in this turn")


class InitDebateResponse(BaseModel):
    """Response model for debate initialization."""
    session_id: str
    agents: List[AgentMetadata]
    topic: str
    max_rounds: int = Field(default=10, description="Maximum number of debate rounds")
    execution_mode: Optional[str] = Field(default=None, description="Persona generation mode used")


class StabilityCheckRequest(BaseModel):
    """Request model for stability check."""
    votes: List[int] = Field(..., description="List of votes from the current round")
    reasonings: Optional[List[str]] = Field(default=None, description="Optional reasoning strings from each agent")


class RoundStatistics(BaseModel):
    """Statistics for a single debate round."""
    round_number: int
    votes: List[int]
    vote_distribution: Dict[str, int]
    consensus_level: float
    vote_mean: float
    vote_variance: float
    ks_statistic: Optional[float] = None


class StabilityCheckResponse(BaseModel):
    """Response model for stability check with comprehensive statistics."""
    stable: bool
    current_round: int
    max_rounds: int
    max_rounds_reached: bool
    should_continue: bool
    continue_reason: str
    round_stats: Optional[Dict[str, Any]] = None


class DebateStatisticsResponse(BaseModel):
    """Response model for debate statistics."""
    session_id: str
    topic: str
    total_rounds: int
    max_rounds: int
    max_rounds_reached: bool
    converged: bool
    convergence_round: Optional[int] = None
    final_consensus_level: Optional[float] = None
    final_vote_distribution: Optional[Dict[str, int]] = None
    winner: Optional[int] = None
    winner_vote_count: Optional[int] = None
    total_agents: int
    convergence_history: List[float]
    agent_analysis: Dict[str, Any]
    round_by_round: List[Dict[str, Any]]


class GeneratePersonasRequest(BaseModel):
    """Request model for generating personas with chain modes."""
    topic: str = Field(..., description="Debate topic for persona generation")
    context: str = Field(default="", description="Additional context for persona generation")
    num_agents: int = Field(default=3, ge=2, le=10, description="Number of agents to generate")
    corpus: Optional[List[str]] = Field(default=None, description="Optional user history statements for style detection and few-shot examples")
    execution_mode: Optional[ExecutionMode] = Field(
        default=None,
        description="Persona generation mode: 'chain_online', 'chain_local', or 'no_chain'"
    )


class GeneratePersonasResponse(BaseModel):
    """Response model for generated personas."""
    personas: List[PersonaConfig]
    topic: str


class DebateContinueResponse(BaseModel):
    """Response model for debate continue check."""
    continue_debate: bool
    reason: str
    current_round: int
    rounds_remaining: Optional[int] = None
    max_rounds: int


class ExtractAndCleanRequest(BaseModel):
    """Request model for combined HTML extraction and cleaning from URL."""
    url: str = Field(..., description="The URL to fetch and clean")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "url": "https://example.com/article",
                    "use_llm": False
                }
            ]
        }
    }


class ExtractAndCleanResponse(BaseModel):
    """Response model for combined HTML extraction and cleaning."""
    url: str
    text: Optional[str] = None  # Keep for backward compatibility
    paragraphs: List[ParagraphResponse] = Field(default_factory=list, description="Content split into paragraphs")
    title: Optional[str] = None
    text_length: Optional[int] = None
    error: Optional[str] = None
    error_message: Optional[str] = None


class ExtractOpinionsRequest(BaseModel):
    """Request model for extracting atomic opinions from text or URL."""
    url: Optional[str] = Field(default=None, description="URL to fetch and analyze (alternative to text)")
    text: Optional[str] = Field(default=None, description="Text content to analyze (alternative to URL)")
    title: Optional[str] = Field(default=None, description="Optional title (used when text is provided)")
    execution_mode: Optional[ExecutionMode] = Field(
        default=None,
        description="Execution mode: 'chain_online', 'chain_local', or 'no_chain'"
    )
    use_llm: bool = Field(default=True, description="Whether to use LLM for opinion extraction and analysis")
    use_cot: bool = Field(default=False, description="Whether to use Chain of Thought reasoning (only when use_llm=True)")
    custom_few_shots: Optional[str] = Field(default=None, description="Optional custom few-shot examples (only when use_llm=True)")
    
    @model_validator(mode='after')
    def validate_url_or_text(self):
        """Ensure either URL or text is provided."""
        if not self.url and not self.text:
            raise ValueError("Either 'url' or 'text' must be provided")
        if self.url and self.text:
            raise ValueError("Cannot provide both 'url' and 'text'. Provide either URL or text+title.")
        return self


class BiasDistributionResponse(BaseModel):
    """Response model for bias distribution."""
    left: float = Field(..., description="Probability of Left/Progressive bias (0.0 to 1.0)")
    right: float = Field(..., description="Probability of Right/Conservative bias (0.0 to 1.0)")
    neutral: float = Field(..., description="Probability of Neutral/Centrist bias (0.0 to 1.0)")
    dominant_bias: str = Field(..., description="The dominant bias category")
    bias_score: float = Field(..., description="Single bias score (-1.0 left to +1.0 right)")


class AtomicOpinionResponse(BaseModel):
    """Response model for a single atomic opinion."""
    text: str
    opinion_type: str
    bias_probabilities: BiasDistributionResponse
    original_sentence: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None


class MBFCMetadataResponse(BaseModel):
    """Response model for MBFC metadata."""
    source_name: Optional[str] = None
    match_type: Optional[str] = None
    bias_rating: Optional[str] = None
    factual_reporting: Optional[str] = None
    raw_db_row: Optional[Dict[str, Any]] = None


class ExtractOpinionsResponse(BaseModel):
    """Response model for opinion extraction."""
    url: Optional[str] = None
    title: Optional[str] = None
    atomic_opinions: List[AtomicOpinionResponse]
    facts: List[AtomicOpinionResponse]
    opinions: List[AtomicOpinionResponse]
    overall_bias_distribution: Optional[BiasDistributionResponse] = None
    mbfc_metadata: Optional[MBFCMetadataResponse] = None
    mbfc_influence_note: Optional[str] = Field(
        default=None,
        description="Brief note describing MBFC prior's influence on bias assessment (e.g., 'strong', 'moderate', 'weak', 'overridden')"
    )
    text_length: int
    truncated: bool
    error: Optional[str] = None
    error_message: Optional[str] = None


class AnalyzeUrlRequest(BaseModel):
    """Request model for complete URL analysis."""
    url: str = Field(..., description="The URL to analyze")
    mode: Optional[str] = Field(
        default="LOCAL_CHAIN",
        description="Logic mode: 'LOCAL_CHAIN', 'NO_CHAIN', or 'PURE_ONLINE'"
    )
    use_mbfc: bool = Field(
        default=True,
        description="Whether to use MBFC database for prior probability"
    )
    use_few_shots: bool = Field(
        default=True,
        description="Whether to use few-shot examples to optimize the agent"
    )
    atomizer_shots: Optional[List[dict]] = Field(
        default=None,
        description="Optional custom few-shot examples for atomization (overrides defaults)"
    )
    scorer_shots: Optional[List[dict]] = Field(
        default=None,
        description="Optional custom few-shot examples for bias scoring (overrides defaults)"
    )


class BiasScoreRequest(BaseModel):
    """Request model for bias score from URL."""
    url: str = Field(..., description="The URL to analyze for bias")
    mode: Optional[str] = Field(
        default="LOCAL_CHAIN",
        description="Logic mode: 'LOCAL_CHAIN', 'NO_CHAIN', or 'PURE_ONLINE'"
    )
    use_mbfc: bool = Field(
        default=True,
        description="Whether to use MBFC database for prior probability"
    )
    use_few_shots: bool = Field(
        default=True,
        description="Whether to use few-shot examples to optimize the agent"
    )


class BiasScoreResponse(BaseModel):
    """Response model for overall bias score."""
    url: str
    overall_bias_distribution: Optional[BiasDistributionResponse] = None
    mbfc_metadata: Optional[MBFCMetadataResponse] = None
    mbfc_influence_note: Optional[str] = Field(
        default=None,
        description="Brief note describing MBFC prior's influence on bias assessment (e.g., 'strong', 'moderate', 'weak', 'overridden')"
    )
    opinions_count: int
    facts_count: int
    error: Optional[str] = None
    error_message: Optional[str] = None


class ShotsResponse(BaseModel):
    """Response model for few-shot examples (custom if set, otherwise system default)."""
    atomizer_shots: List[dict] = Field(default_factory=list, description="Atomizer few-shot examples")
    scorer_shots: List[dict] = Field(default_factory=list, description="Scorer few-shot examples")


def _convert_bias_distribution(bias: BiasDistribution) -> BiasDistributionResponse:
    """Convert BiasDistribution to response model."""
    return BiasDistributionResponse(
        left=bias.left,
        right=bias.right,
        neutral=bias.neutral,
        dominant_bias=bias.dominant_bias,
        bias_score=bias.bias_score
    )


def _convert_atomic_opinion(opinion: AtomicOpinion) -> AtomicOpinionResponse:
    """Convert AtomicOpinion to response model."""
    return AtomicOpinionResponse(
        text=opinion.text,
        opinion_type=opinion.opinion_type,
        bias_probabilities=_convert_bias_distribution(opinion.bias_probabilities),
        original_sentence=opinion.original_sentence,
        confidence=opinion.confidence,
        reasoning=opinion.reasoning
    )


class CheckConsistencyRequest(BaseModel):
    """Request model for checking consistency between website summary and full content."""
    summary: str = Field(..., description="Website's summary/description/abstract text")
    url: str = Field(..., description="Source URL containing the full content to check against")
    enable_deep_analysis: bool = Field(default=True, description="Whether to enable deep analysis using LLM")
    similarity_threshold: float = Field(default=0.2, description="Minimum similarity score (0.0-1.0) required for LLM analysis. Lower values increase accuracy but use more tokens.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "This article discusses how plastic pollution affects marine ecosystems worldwide.",
                    "url": "https://example.com/plastic-pollution-impact",
                    "enable_deep_analysis": True
                }
            ]
        }
    }


class CompareClaimsRequest(BaseModel):
    """Request model for comparing two specific claims."""
    summary_claim: str = Field(..., description="Claim from the summary")
    url_claim: str = Field(..., description="Claim from the URL content")
    url_content: Optional[str] = Field(default=None, description="Optional full URL content for context")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary_claim": "The Chicago Fire Department will undergo significant changes next season",
                    "url_claim": "There will be hellos and goodbyes in the department",
                    "url_content": "Full content of the webpage for additional context..."
                }
            ]
        }
    }


class OverallEvaluationRequest(BaseModel):
    """Request model for overall evaluation of summary and URL."""
    summary: Optional[str] = Field(default=None, description="Text summary to evaluate")
    url: Optional[str] = Field(default=None, description="URL to evaluate")
    include_full_analysis: bool = Field(default=False, description="Whether to include full academic analysis pipeline")
    enable_deep_analysis: bool = Field(default=True, description="Whether to enable deep analysis (fact/opinion density, bias) using LLM")
    enable_consistency_check: bool = Field(default=False, description="Whether to enable consistency check between summary and URL using the 5-step verification pipeline")

    @model_validator(mode='after')
    def validate_input(self):
        """Ensure at least one input is provided."""
        if not self.summary and not self.url:
            raise ValueError("At least one of 'summary' or 'url' must be provided")
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Plastic pollution harms marine life and ecosystems worldwide.",
                    "url": "https://example.com/environmental-issues",
                    "enable_consistency_check": True,
                    "enable_deep_analysis": True
                },
                {
                    "url": "https://example.com/climate-change",
                    "include_full_analysis": True
                }
            ]
        }
    }


class EvaluationMetrics(BaseModel):
    """Evaluation metrics for content."""
    content_length: int
    readability_score: Optional[float] = None
    fact_density: Optional[float] = None
    opinion_density: Optional[float] = None
    bias_distribution: Optional[Dict[str, float]] = None
    academic_integrity_score: Optional[float] = None
    hallucination_risk: Optional[float] = None


class OverallEvaluationResponse(BaseModel):
    """Response model for overall evaluation."""
    summary_evaluation: Optional[Dict[str, Any]] = None
    url_evaluation: Optional[Dict[str, Any]] = None
    comparative_analysis: Optional[Dict[str, Any]] = None
    consistency_check: Optional[Dict[str, Any]] = None
    full_academic_analysis: Optional[Dict[str, Any]] = None
    processing_metadata: Dict[str, Any]


class CompleteAnalysisRequest(BaseModel):
    """Request model for complete academic analysis pipeline."""
    url: str = Field(..., description="URL to analyze completely")
    use_llm_content_extraction: bool = Field(default=False, description="Use LLM for content extraction refinement")
    use_cot_atomization: bool = Field(default=False, description="Use CoT for claim atomization")
    use_cot_audit: bool = Field(default=False, description="Use CoT for conflict auditing")
    custom_few_shots_atomizer: Optional[str] = Field(default=None, description="Custom few-shots for atomizer")
    custom_few_shots_auditor: Optional[str] = Field(default=None, description="Custom few-shots for auditor")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "url": "https://example.com/research-paper",
                    "use_cot_atomization": True,
                    "use_cot_audit": True,
                }
            ]
        }
    }


class PipelineStepResponse(BaseModel):
    """Response model for a pipeline step."""
    step_name: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None


class CompleteAnalysisResponse(BaseModel):
    """Response model for complete academic analysis."""
    url: str
    pipeline_steps: List[PipelineStepResponse]
    final_report: Optional[Dict[str, Any]] = None
    overall_success: bool
    total_execution_time: Optional[float] = None
    error: Optional[str] = None


class DetectPrivacyRequest(BaseModel):
    """Request model for privacy leak detection."""
    conversation_records: List[Dict[str, str]] = Field(..., description="List of conversation records with 'role' and 'content' keys")
    execution_mode: Optional[ExecutionMode] = Field(
        default=None,
        description="Execution mode: 'chain_online' (LLM chains), 'chain_local' (local decomposition), 'no_chain' (pure prompt). Defaults to system setting."
    )
    use_few_shots: bool = Field(
        default=True,
        description="Whether to use few-shot examples in the prompt (default: True)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "conversation_records": [
                        {"role": "user", "content": "Hi, I'm John Smith from 123 Main Street, Springfield. My phone is 555-0123."},
                        {"role": "assistant", "content": "Hello John! I'll help you with your request."}
                    ],
                    "use_few_shots": True
                }
            ]
        }
    }


class PrivacyLeakItem(BaseModel):
    """Individual privacy leak item."""
    privacy_type: str = Field(..., description="Type of privacy information detected: personal_identifiers, contact_info, government_ids, biometric_data, financial_accounts, financial_transactions, financial_documents, health_records, medical_history, prescriptions_medications, authentication_credentials, api_keys_tokens, system_access, encryption_keys, precise_location, location_history, email_content, messaging_content, call_logs, social_security, relationship_data, behavioral_data, none")
    severity: str = Field(..., description="Severity level of this privacy leak: low, medium, high, critical, none")
    reasoning: str = Field(..., description="Explanation specific to this privacy leak")
    detected_items: List[str] = Field(..., description="List of specific privacy items detected for this leak")


class PrivacyDetectionResponse(BaseModel):
    """Response model for privacy leak detection."""
    detection_id: str = Field(..., description="Unique identifier for this privacy detection result")
    privacy_detected: bool = Field(..., description="Whether any privacy-sensitive information was detected")
    privacy_leaks: List[PrivacyLeakItem] = Field(default_factory=list, description="List of detected privacy leaks")
    overall_severity: str = Field(..., description="Overall severity level across all detected leaks: low, medium, high, critical, none")
    reasoning: str = Field(..., description="Overall explanation of the privacy analysis")
    execution_mode: str = Field(..., description="Execution mode used for analysis")
    conversation_length: int = Field(..., description="Number of conversation records analyzed")
    analyzed_at: str = Field(..., description="Timestamp of analysis")
    agent_version: str = Field(..., description="Version of the privacy detection agent")
    error: Optional[str] = Field(default=None, description="Error message if analysis failed")


