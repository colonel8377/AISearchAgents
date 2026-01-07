"""API request and response schemas."""

from typing import List, Optional, Dict, Any, Union, Literal
from uuid import UUID
from pydantic import BaseModel, Field, model_validator, field_validator

from src.shared.constant.enums import CoTMode, PrivacyType, PrivacySeverity, ConflictType, AgentType


class EnumResponse(BaseModel):
    """Base model for enum responses that include both string and numeric values."""
    value: str = Field(..., description="The enum string value")
    code: int = Field(..., description="The enum numeric code")

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
    agent_type: EnumResponse
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
    sentences: Union[List[str], str] = Field(..., description="List of sentences to evaluate")
    use_cot: Union[CoTMode, str, int] = Field(
        default=CoTMode.NO_CHAIN,
        description="Chain of Thought mode: 'chain_online' (0), 'chain_local' (1), 'no_chain' (2), or CoTMode enum. Default: no_chain"
    )
    use_few_shots: bool = Field(
        default=True,
        description="Whether to use few-shot examples in the prompt (default: True)"
    )
    custom_few_shots: Optional[str] = Field(
        default=None,
        description="Optional custom few-shot examples to use instead of defaults. If provided, use_few_shots must be True."
    )
    per_message: bool = Field(
        default=False,
        description="If true, evaluate each sentence as a separate user message (one LLM call per sentence). Default: False"
    )
    is_binary_agreement: bool = Field(
        default=False,
        description="If true, only return 0 or 1 (binary agreement). If false (default), return 0.0 to 1.0 (continuous scale). Default: False"
    )

    @field_validator('use_cot', mode='before')
    @classmethod
    def validate_use_cot(cls, v):
        """Convert various input formats to CoTMode enum."""
        if isinstance(v, CoTMode):
            return v
        try:
            return CoTMode.from_code_or_value(v)
        except ValueError as e:
            raise ValueError(f"Invalid use_cot value: {e}")
    
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
    agree: Union[int, float] = Field(..., ge=0, le=1, description="Agreement score: 0.0-1.0 (continuous) or 0/1 (binary)")
    reason: str


class EvaluateSentencesResponse(BaseModel):
    """Response model for sentence evaluation."""
    judgments: List[JudgmentResponse]


class AgentStatusResponse(BaseModel):
    """Response model for agent status."""
    agent_id: str
    agent_type: EnumResponse
    status: str
    current_turn: Optional[int] = None
    additional_info: Optional[Dict[str, Any]] = None


class SummarizeRequest(BaseModel):
    """Request model for summarizing conversations."""
    conversation_records: List[Dict[str, str]] = Field(..., description="List of conversation records to summarize")
    use_cot: Union[CoTMode, str, int] = Field(
        default=CoTMode.CHAIN_LOCAL,
        description="Chain of Thought mode: 'chain_online' (0), 'chain_local' (1), 'no_chain' (2), or CoTMode enum. Default: chain_local"
    )
    use_few_shots: bool = Field(
        default=True,
        description="Whether to use few-shot examples in the prompt (default: True)"
    )

    @field_validator('use_cot', mode='before')
    @classmethod
    def validate_use_cot(cls, v):
        """Convert various input formats to CoTMode enum."""
        if isinstance(v, CoTMode):
            return v
        try:
            return CoTMode.from_code_or_value(v)
        except ValueError as e:
            raise ValueError(f"Invalid use_cot value: {e}")


class SummaryResponse(BaseModel):
    """Response model for summary."""
    summary: str
    conversation_length: int
    original_length: int
    truncated: bool
    metadata: Dict[str, Any]


class CreateBotRequest(BaseModel):
    """Request model for creating a bot."""
    bot_name: Optional[str] = Field(default=None, description="Optional name for the bot")
    use_few_shots: bool = Field(
        default=True,
        description="Whether to use few-shot examples in the prompt (default: True)"
    )


class BotCreationResponse(BaseModel):
    """Response model for bot creation."""
    bot_name: str
    status: str
    persona_mode: Optional[str] = None
    bot_configuration: str
    message: str


class ListAgentsResponse(BaseModel):
    """Response model for listing agents."""
    agents: List[Dict[str, str]]
    total_count: int


class ExtractContentRequest(BaseModel):
    """Request model for content extraction."""
    url: Optional[str] = Field(default=None, description="URL to extract content from")
    text: Optional[str] = Field(default=None, description="Plain text to process (alternative to URL/HTML)")
    title: Optional[str] = Field(default=None, description="Optional title (used when text is provided)")
    summary: Optional[str] = Field(default=None, description="Optional summary text for claim-level comparison with URL content")
    use_llm: bool = Field(default=False, description="Whether to use LLM for content refinement")
    use_cot: Union[CoTMode, str, int] = Field(
        default=CoTMode.NO_CHAIN,
        description="Chain of Thought mode: 'chain_online' (0), 'chain_local' (1), 'no_chain' (2), or CoTMode enum. Default: no_chain (only when use_llm=True)"
    )
    use_few_shots: bool = Field(default=True, description="Whether to use few-shot examples for claim comparison (only when compare_claims=True)")
    custom_few_shots: Optional[str] = Field(default=None, description="Optional custom few-shot examples (only when use_llm=True)")
    compare_claims: bool = Field(default=False, description="Whether to perform claim-level comparison (requires summary to be provided)")

    @field_validator('use_cot', mode='before')
    @classmethod
    def validate_use_cot(cls, v):
        """Convert various input formats to CoTMode enum."""
        if isinstance(v, CoTMode):
            return v
        try:
            return CoTMode.from_code_or_value(v)
        except ValueError as e:
            raise ValueError(f"Invalid use_cot value: {e}")

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
    relationship: EnumResponse = Field(..., description="Overall relationship with both string value and numeric code: 'agree' (0), 'disagree' (1), 'missing' (2)")
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
    use_cot: Union[CoTMode, str, int] = Field(
        default=CoTMode.NO_CHAIN,
        description="Chain of Thought mode: 'chain_online' (0), 'chain_local' (1), 'no_chain' (2), or CoTMode enum. Default: no_chain"
    )
    split_into_paragraphs: bool = Field(default=False, description="Whether to split text into paragraphs before atomization")
    use_few_shots: bool = Field(
        default=True,
        description="Whether to use few-shot examples in the prompt (default: True)"
    )

    @field_validator('use_cot', mode='before')
    @classmethod
    def validate_use_cot(cls, v):
        """Convert various input formats to CoTMode enum."""
        if isinstance(v, CoTMode):
            return v
        try:
            return CoTMode.from_code_or_value(v)
        except ValueError as e:
            raise ValueError(f"Invalid use_cot value: {e}")


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
    use_cot: Union[CoTMode, str, int] = Field(
        default=CoTMode.NO_CHAIN,
        description="Chain of Thought mode: 'chain_online' (0), 'chain_local' (1), 'no_chain' (2), or CoTMode enum. Default: no_chain"
    )
    custom_few_shots: Optional[str] = Field(default=None, description="Optional custom few-shot examples")

    @field_validator('use_cot', mode='before')
    @classmethod
    def validate_use_cot(cls, v):
        """Convert various input formats to CoTMode enum."""
        if isinstance(v, CoTMode):
            return v
        try:
            return CoTMode.from_code_or_value(v)
        except ValueError as e:
            raise ValueError(f"Invalid use_cot value: {e}")


class ConflictAnalysisResponse(BaseModel):
    """Response model for conflict analysis."""
    claim_id: str
    claim_text: str
    evidence_quotes: List[str]
    verdict: EnumResponse
    conflict_type: str
    analysis: str
    confidence: float


class ConflictAuditResponse(BaseModel):
    """Response model for conflict auditing."""
    conflict_analyses: List[ConflictAnalysisResponse]
    summary_stats: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None



class ChatWithBotRequest(BaseModel):
    """Request model for chatting with a bot."""
    bot_name: str = Field(..., description="Name of the bot to chat with (unique identifier)")
    message: str = Field(..., description="User message to send to the bot")
    conversation_id: Optional[str] = Field(
        default=None,
        description="Conversation ID for continuing a conversation thread. If provided, bot remembers all previous messages in this conversation thread."
    )
    conversation_title: Optional[str] = Field(
        default=None,
        description="Optional title for new conversations. If not provided and auto_create_conversation is True, title will be auto-generated from the first message."
    )
    auto_create_conversation: bool = Field(
        default=True,
        description="If True (default) and conversation_id is not provided, automatically creates a new conversation (ChatGPT-like behavior). If False, uses incognito mode (temporary, not saved)."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "bot_name": "MyBot",
                    "message": "What is machine learning?",
                    "conversation_id": "conv-123"  # Continue existing conversation
                },
                {
                    "bot_name": "MyBot",
                    "message": "What is machine learning?",
                    "auto_create_conversation": True  # Auto-create new conversation (ChatGPT-like)
                },
                {
                    "bot_name": "MyBot",
                    "message": "What is machine learning?",
                    "auto_create_conversation": False  # Incognito mode - temporary, not saved
                }
            ]
        }
    }


class ChatWithBotResponse(BaseModel):
    """Response model for bot chat."""
    bot_name: str
    response: str
    mode: str = Field(..., description="Mode: 'conversation' (persistent thread) or 'incognito' (temporary)")
    conversation_history: Optional[List[Dict[str, str]]] = Field(default=None, description="Conversation history (only returned in conversation mode)")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID if this was part of a conversation thread")
    conversation_title: Optional[str] = Field(default=None, description="Conversation title if applicable")
    turn_count: Optional[int] = Field(default=None, description="Number of turns in this conversation (only for conversation mode)")



class CreateConversationRequest(BaseModel):
    """Request model for creating a new conversation."""
    title: Optional[str] = Field(default=None, description="Optional title for the conversation")
    system_prompt: Optional[str] = Field(default=None, description="Optional system prompt to initialize the conversation")

class RenameConversationRequest(BaseModel):
    """Request model for renaming a conversation."""
    title: str = Field(..., description="New title for the conversation")


class BotInfo(BaseModel):
    """Bot information for listing."""
    bot_name: str = Field(..., description="Bot name")
    status: str = Field(..., description="Bot status")
    conversations_count: int = Field(..., description="Number of conversations")
    created_at: Optional[str] = Field(None, description="Creation timestamp")


class BotListResponse(BaseModel):
    """Response model for listing bots."""
    bots: List[BotInfo] = Field(..., description="List of bots with their information")
    total_count: int = Field(..., description="Total number of bots")


class ConversationInfo(BaseModel):
    """Conversation information for listing."""
    conversation_id: str = Field(..., description="Conversation UUID")
    title: str = Field(..., description="Conversation title")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    turn_count: int = Field(..., description="Number of turns in conversation")
    last_message: Optional[str] = Field(None, description="Last message in conversation")


class ConversationTurn(BaseModel):
    """A single conversation turn."""
    turn_index: int = Field(..., description="Turn index (0-based)")
    user_message: str = Field(..., description="User's message")
    assistant_response: str = Field(..., description="Assistant's response")
    user_role: str = Field(..., description="User role")
    assistant_role: str = Field(..., description="Assistant role")


class ConversationListResponse(BaseModel):
    """Response model for listing conversations."""
    bot_name: str = Field(..., description="Bot name")
    conversations: List[ConversationInfo] = Field(..., description="List of conversations with metadata")
    total_count: int = Field(..., description="Total number of conversations")


class ConversationDetailResponse(BaseModel):
    """Response model for conversation details."""
    bot_name: str = Field(..., description="Bot name")
    conversation_id: str = Field(..., description="Conversation UUID")
    title: str = Field(..., description="Conversation title")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    total_turns: int = Field(..., description="Total number of conversation turns")
    turns: List[ConversationTurn] = Field(..., description="Conversation turns with user/assistant messages")
    raw_history: List[Dict[str, Any]] = Field(..., description="Raw conversation history")


class AddTurnRequest(BaseModel):
    """Request model for adding a turn to a conversation."""
    user_message: str = Field(..., description="User's message in this turn")
    assistant_message: str = Field(..., description="Assistant's response in this turn")


# ===========================
# Debate System Schemas
# ===========================

class PersonaConfig(BaseModel):
    """Configuration for a debate agent persona."""
    name: str = Field(..., description="Name of the persona")
    description: str = Field(..., description="Description of the persona's characteristics")
    style: Optional[str] = Field(default=None, description="Optional style of the persona (e.g., 'Skeptical', 'Optimistic', 'Neutral') - auto-detected if not provided")
    corpus: Optional[List[str]] = Field(default=None, description="Optional corpus of statements specific to this persona's background and expertise")


class AgentMetadata(BaseModel):
    """Metadata for a debate agent."""
    agent_id: UUID = Field(..., description="Unique identifier for the agent")
    role_name: str = Field(..., description="Role name of the agent")
    system_prompt: str = Field(..., description="System prompt for the agent")
    few_shot_example: str = Field(..., description="Static example showing the expected JSON output format")
    corpus: Optional[List[str]] = Field(default=None, description="Optional corpus of statements specific to this agent's background and expertise")


class InitRequest(BaseModel):
    """Request model for initializing a debate session."""
    topic: str = Field(..., description="Topic of the debate")
    custom_personas: List[PersonaConfig] = Field(default_factory=list, description="Custom personas for agents")
    auto_agent_count: int = Field(default=0, description="Number of agents to auto-generate if custom_personas is empty")
    max_rounds: Optional[int] = Field(default=10, ge=1, le=50, description="Maximum number of debate rounds")
    context: str = Field(default="", description="Additional context for persona generation")
    corpus: Optional[List[str]] = Field(default=None, description="Optional user history statements for style detection and few-shot examples")


class InteractRequest(BaseModel):
    """Request model for agent interaction."""
    session_id: str = Field(..., description="Session identifier for the debate")
    agent_id: UUID = Field(..., description="Agent identifier")
    history_context: str = Field(..., description="String summary of what others said")


class VoteResponse(BaseModel):
    """Response model for agent vote."""
    agent_id: UUID = Field(..., description="Agent identifier")
    verdict: Union[int, str] = Field(..., description="The agent's verdict/vote")
    reasoning: str = Field(..., description="Reasoning behind the verdict")


class AgentReasoning(BaseModel):
    """Response model for a single agent's reasoning in a round."""
    agent_id: UUID = Field(..., description="Agent identifier")
    statement: str = Field(..., description="Agent's position statement")
    reasoning: str = Field(..., description="Detailed reasoning behind the statement")
    timestamp: float = Field(..., description="Timestamp when reasoning was generated")
    word_count: int = Field(..., description="Word count of the combined statement and reasoning")


class AgentVote(BaseModel):
    """Response model for a single agent's vote in a round."""
    agent_id: UUID = Field(..., description="Agent identifier")
    verdict: Union[int, str] = Field(..., description="The agent's verdict/vote")
    reasoning: str = Field(..., description="Reasoning behind the verdict")
    timestamp: float = Field(..., description="Timestamp when vote was cast")
    word_count: int = Field(..., description="Word count of the reasoning")


class RoundReasoningResponse(BaseModel):
    """Response model for a complete reasoning round."""
    round_number: int = Field(..., description="Current round number")
    reasonings: List[AgentReasoning] = Field(..., description="All agents' reasoning for this round")
    timestamp: float = Field(..., description="Timestamp when round was completed")
    total_word_count: int = Field(..., description="Total word count across all reasonings")


class RoundVotingResponse(BaseModel):
    """Response model for a complete voting round."""
    round_number: int = Field(..., description="Current round number")
    votes: List[AgentVote] = Field(..., description="All agents' votes for this round")
    timestamp: float = Field(..., description="Timestamp when round was completed")
    total_word_count: int = Field(..., description="Total word count across all vote reasonings")
    statistics: Dict[str, Any] = Field(..., description="Round statistics including consensus level, distributions, etc.")


class InitDebateResponse(BaseModel):
    """Response model for debate initialization."""
    session_id: str
    agents: List[AgentMetadata]
    topic: str
    max_rounds: int = Field(default=10, description="Maximum number of debate rounds")


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


# ===========================
# Web Opinion Extractor Models (Agent-to-API Contract)
# ===========================

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
    use_llm: bool = Field(default=True, description="Whether to use LLM for opinion extraction and analysis")
    use_cot: Union[CoTMode, str, int] = Field(
        default=CoTMode.NO_CHAIN,
        description="Chain of Thought mode: 'chain_online' (0), 'chain_local' (1), 'no_chain' (2), or CoTMode enum. Default: no_chain (only when use_llm=True)"
    )
    custom_few_shots: Optional[str] = Field(default=None, description="Optional custom few-shot examples (only when use_llm=True)")

    @field_validator('use_cot', mode='before')
    @classmethod
    def validate_use_cot(cls, v):
        """Convert various input formats to CoTMode enum."""
        if isinstance(v, CoTMode):
            return v
        try:
            return CoTMode.from_code_or_value(v)
        except ValueError as e:
            raise ValueError(f"Invalid use_cot value: {e}")

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
    opinion_type: EnumResponse
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
        description="Whether to use MBFC persistence for prior probability"
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
        description="Whether to use MBFC persistence for prior probability"
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
    # Convert opinion_type to EnumResponse (fact=0, opinion=1)
    opinion_type_enum = EnumResponse(
        value=opinion.opinion_type,
        code=0 if opinion.opinion_type == "fact" else 1
    )

    return AtomicOpinionResponse(
        text=opinion.text,
        opinion_type=opinion_type_enum,
        bias_probabilities=_convert_bias_distribution(opinion.bias_probabilities),
        original_sentence=opinion.original_sentence,
        confidence=opinion.confidence,
        reasoning=opinion.reasoning
    )


def _convert_enum_to_response(enum_value: Any, enum_class: Any) -> EnumResponse:
    """Convert an enum value to EnumResponse with both string and numeric values."""
    if hasattr(enum_value, 'value'):
        # Handle str(Enum) cases where value is the string
        string_value = enum_value.value if isinstance(enum_value.value, str) else str(enum_value.value)
    else:
        # Handle raw string values that should be enums
        string_value = str(enum_value)

    # Get the numeric code from the enum class (definition order index)
    numeric_code = 0  # Default fallback
    try:
        if hasattr(enum_class, '__members__'):
            # Find the enum member that matches the value and get its definition order
            enum_members = list(enum_class)
            for i, member in enumerate(enum_members):
                if member.value == string_value:
                    numeric_code = i
                    break
    except Exception:
        # Fallback for any issues
        pass

    return EnumResponse(value=string_value, code=numeric_code)


def _convert_privacy_type_enum(privacy_type: str) -> EnumResponse:
    """Convert privacy type string to EnumResponse."""
    try:
        enum_value = PrivacyType(privacy_type)
        return _convert_enum_to_response(enum_value, PrivacyType)
    except ValueError:
        # Fallback for unknown values
        return EnumResponse(value=privacy_type, code=0)


def _convert_privacy_severity_enum(severity: str) -> EnumResponse:
    """Convert privacy severity string to EnumResponse."""
    try:
        enum_value = PrivacySeverity(severity)
        return _convert_enum_to_response(enum_value, PrivacySeverity)
    except ValueError:
        # Fallback for unknown values
        return EnumResponse(value=severity, code=0)


def _convert_claim_relationship_enum(relationship: str) -> EnumResponse:
    """Convert claim relationship string to EnumResponse."""
    # Mapping: agree=0, disagree=1, missing=2
    relationship_map = {"agree": 0, "disagree": 1, "missing": 2}
    code = relationship_map.get(relationship, 2)  # Default to missing (2)
    return EnumResponse(value=relationship, code=code)


def _convert_conflict_verdict_enum(verdict: str) -> EnumResponse:
    """Convert conflict verdict string to EnumResponse."""
    try:
        enum_value = ConflictType(verdict)
        return _convert_enum_to_response(enum_value, ConflictType)
    except ValueError:
        # Fallback for unknown values
        return EnumResponse(value=verdict, code=0)


def _convert_agent_type_enum(agent_type: str) -> EnumResponse:
    """Convert agent type string to EnumResponse."""
    try:
        enum_value = AgentType(agent_type)
        return _convert_enum_to_response(enum_value, AgentType)
    except ValueError:
        # Fallback for unknown values
        return EnumResponse(value=agent_type, code=0)


class CheckConsistencyRequest(BaseModel):
    """Request model for checking consistency between website summary and full content."""
    summary: str = Field(..., description="Website's summary/description/abstract text")
    url: str = Field(..., description="Source URL containing the full content to check against")
    enable_deep_analysis: bool = Field(default=True, description="Whether to enable deep analysis using LLM")
    similarity_threshold: float = Field(default=0.2, description="Minimum similarity score (0.0-1.0) required for LLM analysis. Lower values increase accuracy but use more tokens.")
    use_cot_atomization: Union[CoTMode, str, int] = Field(
        default=CoTMode.NO_CHAIN,
        description="Chain of Thought mode for atomization: 'chain_online' (0), 'chain_local' (1), 'no_chain' (2), or CoTMode enum. Default: no_chain"
    )
    use_cot_audit: Union[CoTMode, str, int] = Field(
        default=CoTMode.NO_CHAIN,
        description="Chain of Thought mode for auditing: 'chain_online' (0), 'chain_local' (1), 'no_chain' (2), or CoTMode enum. Default: no_chain"
    )

    @field_validator('use_cot_atomization', mode='before')
    @classmethod
    def validate_use_cot_atomization(cls, v):
        """Convert various input formats to CoTMode enum."""
        if isinstance(v, CoTMode):
            return v
        try:
            return CoTMode.from_code_or_value(v)
        except ValueError as e:
            raise ValueError(f"Invalid use_cot_atomization value: {e}")

    @field_validator('use_cot_audit', mode='before')
    @classmethod
    def validate_use_cot_audit(cls, v):
        """Convert various input formats to CoTMode enum."""
        if isinstance(v, CoTMode):
            return v
        try:
            return CoTMode.from_code_or_value(v)
        except ValueError as e:
            raise ValueError(f"Invalid use_cot_audit value: {e}")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "This article discusses how plastic pollution affects marine ecosystems worldwide.",
                    "url": "https://example.com/plastic-pollution-impact",
                    "enable_deep_analysis": True,
                    "use_cot_atomization": "no_chain",
                    "use_cot_audit": "no_chain"
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
    conversation_records: List[Dict[str, Any]] = Field(..., description="List of user messages with 'user' key containing the message content, and optionally 'account_id'")
    
    cot_mode: Optional[CoTMode] = Field(
        default=None,
        description="Chain of Thought mode: 'chain_online' (LLM handles task decomposition), 'chain_local' (System handles decomposition), 'no_chain' (Direct prompt). If not provided, uses use_cot or execution_mode."
    )
    
    use_cot: Optional[Union[CoTMode, str, int]] = Field(
        default=None,
        description="Legacy Chain of Thought mode: 'chain_online' (0), 'chain_local' (1), 'no_chain' (2), or CoTMode enum. Deprecated, use cot_mode instead."
    )
    
    use_few_shots: bool = Field(
        default=True,
        description="Whether to use few-shot examples in the prompt (default: True)"
    )

    @field_validator('use_cot', mode='before')
    @classmethod
    def validate_use_cot(cls, v):
        """Convert various input formats to CoTMode enum."""
        if v is None:
            return None
        if isinstance(v, CoTMode):
            return v
        try:
            return CoTMode.from_code_or_value(v)
        except ValueError as e:
            raise ValueError(f"Invalid use_cot value: {e}")

    @field_validator('conversation_records')
    @classmethod
    def validate_conversation_records(cls, v):
        """Validate that each record contains 'user' key and optionally 'account_id'."""
        for i, record in enumerate(v):
            if not isinstance(record, dict):
                raise ValueError(f"Record {i} must be a dictionary")
            if 'user' not in record:
                raise ValueError(f"Record {i} must contain 'user' key")
            # Allow additional fields like account_id
        return v

    account_id: Optional[str] = Field(
        default=None,
        description="Unique account identifier for grouping conversations"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "conversation_records": [
                        {"user": "My email is john@example.com and SSN is 123-45-6789"},
                        {"user": "The password for my bank account is 'secret123'"},
                        {"user": "Call me at 555-123-4567 if you need anything"}
                    ],
                    "account_id": "user123",
                    "cot_mode": "chain_local",
                    "use_few_shots": True
                }
            ]
        }
    }


class PrivacyLeakItem(BaseModel):
    """Individual privacy leak item."""
    privacy_type: EnumResponse = Field(..., description="Type of privacy information detected with both string value and numeric code")
    severity: EnumResponse = Field(..., description="Severity level of this privacy leak with both string value and numeric code")
    severity_level: Optional[str] = Field(default=None, description="L1-L4 grading level (L1=Low, L2=Medium, L3=High, L4=Critical)")
    category: Optional[str] = Field(default=None, description="Privacy category: IDENTITY, FINANCIAL, MEDICAL, TECHNICAL")
    reasoning: str = Field(..., description="Explanation specific to this privacy leak")
    detected_items: List[str] = Field(..., description="List of specific privacy items detected for this leak")
    confidence: float = Field(default=0.5, description="Confidence score for this privacy leak detection (0.0 to 1.0)")
    region: Optional[str] = Field(default=None, description="Spatial region: HEADER, BODY, FOOTER, SIDEBAR")


class PrivacyDetectionResponse(BaseModel):
    """Response model for privacy leak detection."""
    detection_id: str = Field(..., description="Unique identifier for this privacy detection result")
    privacy_detected: bool = Field(..., description="Whether any privacy-sensitive information was detected")
    privacy_leaks: List[PrivacyLeakItem] = Field(default_factory=list, description="List of detected privacy leaks")
    overall_severity: EnumResponse = Field(..., description="Overall severity level across all detected leaks with both string value and numeric code")
    overall_severity_level: Optional[str] = Field(default=None, description="Overall L1-L4 grading level (L1=Low, L2=Medium, L3=High, L4=Critical)")
    overall_score: float = Field(..., description="Overall confidence score (0.0-1.0) as average of all leak confidences")
    conversation_length: int = Field(..., description="Number of conversation records analyzed")
    messages_analyzed: int = Field(..., description="Number of messages that were analyzed (should equal conversation_length)")
    analyzed_at: str = Field(..., description="Timestamp of analysis")
    agent_version: str = Field(..., description="Version of the privacy detection agent")
    error: Optional[str] = Field(default=None, description="Error message if analysis failed")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "detection_id": "550e8400-e29b-41d4-a716-446655440000",
                    "privacy_detected": True,
                    "privacy_leaks": [
                        {
                            "privacy_type": {"value": "contact_info", "code": 1},
                            "severity": {"value": "medium", "code": 2},
                            "severity_level": "L2",
                            "category": "IDENTITY",
                            "reasoning": "Email address detected in user message",
                            "detected_items": ["john@example.com"],
                            "confidence": 0.95,
                            "region": None
                        }
                    ],
                    "overall_severity": {"value": "medium", "code": 2},
                    "overall_severity_level": "L2",
                    "overall_score": 0.95,
                    "conversation_length": 3,
                    "messages_analyzed": 3,
                    "analyzed_at": "2024-01-01T12:00:00Z",
                    "agent_version": "3.0.0",
                    "error": None
                }
            ]
        }
    }






