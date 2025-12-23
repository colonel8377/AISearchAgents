"""FastAPI application for AI Search Agents Platform - Optimized Version."""

from typing import List, Optional, Dict, Any, Literal
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field, model_validator
from langchain_openai import OpenAIEmbeddings

from ..config.settings import settings, ExecutionMode, HistoryMode
from ..utils.logger import configure_app_logging, get_logger
from ..memory.factory import VectorStoreFactory
from ..agents.nudge_collapse.agent import NudgeCollapseAgent
from ..agents.summarizer.agent import SummarizerAgent
from ..agents.bot_creator.agent import BotCreatorAgent
from ..agents.demographic_evaluator.agent import DemographicEvaluatorAgent
from ..agents.manager import AgentManager, AgentType
from ..agents.content_extractor.agent import ContentExtractorAgent, ContentExtractionResult
from ..agents.claim_atomizer.agent import ClaimAtomizerAgent, ClaimAtomizationResult, AtomicClaim
from ..agents.evidence_locator.agent import EvidenceLocatorAgent, EvidenceLocationResult, ClaimEvidence, EvidenceQuote
from ..agents.conflict_auditor.agent import ConflictAuditorAgent, ConflictAuditResult, ConflictAnalysis, ConflictType
from ..agents.synthesis_aggregator.agent import SynthesisAggregatorAgent, SynthesisAggregationResult, SynthesisReport
from .auth import verify_api_key
from ..debate.schemas import PersonaConfig, AgentMetadata, InitRequest, InteractRequest, VoteResponse
from ..debate.service import DebateService, generate_default_personas
from ..agents.web_opinion_extractor import (
    WebOpinionAnalyzer,
    WebOpinionEngine,
    LogicMode,
    OpinionExtractionResult,
    AtomicOpinion,
    BiasDistribution
)
from ..utils.web_opinion_cache import get_cache

# Configure centralized logging
configure_app_logging(
    log_level=settings.log_level,
    log_file=settings.log_file,
    enable_debug=settings.enable_debug
)
logger = get_logger(__name__)


# Pydantic models for request/response
class CreateAgentRequest(BaseModel):
    """Request model for creating a new agent instance."""
    agent_type: str = Field(..., description="Type of agent: 'nudge_collapse', 'summarizer', 'bot_creator', or 'demographic_evaluator'")
    agent_id: Optional[str] = Field(default=None, description="Custom agent ID (auto-generated if not provided)")
    use_memory: bool = Field(default=False, description="Whether to use vector memory for this agent")
    persona_mode: Optional[str] = Field(default="system_prompt", description="Persona mode for bot_creator: 'system_prompt' or 'user_instruction'")


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
    history_mode: Optional[str] = Field(
        default=None,
        description="History mode: 'full' (include conversation history), 'none' (stateless). Defaults to system setting."
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


class TurnResponse(BaseModel):
    """Response model for a turn."""
    turn: int
    query: str
    response: str
    search_summary: str
    search_urls: List[str]
    strategy: str
    history_mode: Optional[str] = None


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


# Content Extractor Models
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


# Claim Atomizer Models (needed for ClaimComparisonResponse)
class AtomicClaimResponse(BaseModel):
    """Response model for a single atomic claim."""
    id: str
    text: str
    original_sentence: str
    confidence: float


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


class ContentExtractionResponse(BaseModel):
    """Response model for content extraction."""
    url: Optional[str] = None
    title: Optional[str] = None
    main_body: Optional[str] = None
    text_length: int
    truncated: bool
    extraction_metadata: Optional[Dict[str, Any]] = None
    claim_comparison: Optional[ClaimComparisonResponse] = Field(default=None, description="Claim-level comparison result (if compare_claims=True)")


# Claim Atomizer Models
class AtomizeClaimsRequest(BaseModel):
    """Request model for claim atomization."""
    text: str = Field(..., description="Text snippet to decompose into atomic claims")
    use_cot: bool = Field(default=False, description="Whether to use Chain of Thought reasoning")
    custom_few_shots: Optional[str] = Field(default=None, description="Optional custom few-shot examples")


class ClaimAtomizationResponse(BaseModel):
    """Response model for claim atomization."""
    atomic_claims: List[AtomicClaimResponse]
    original_text: str
    execution_mode: str
    metadata: Optional[Dict[str, Any]] = None


# Evidence Locator Models
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


# Conflict Auditor Models
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


# Synthesis Aggregator Models
class AggregateSynthesisRequest(BaseModel):
    """Request model for synthesis aggregation."""
    conflict_analyses: List[Dict[str, Any]] = Field(..., description="List of conflict analysis results")
    use_llm_enhancement: bool = Field(default=True, description="Whether to use LLM for enhanced analysis")


class SynthesisReportResponse(BaseModel):
    """Response model for synthesis report."""
    research_summary: str
    metrics: Dict[str, Any]
    detailed_discrepancies: List[Dict[str, Any]]
    quality_assessment: str
    confidence_score: float


class SynthesisAggregationResponse(BaseModel):
    """Response model for synthesis aggregation."""
    synthesis_report: SynthesisReportResponse
    metadata: Optional[Dict[str, Any]] = None


# Initialize FastAPI app
app = FastAPI(
    title="AI Search Agents Platform",
    description="Platform for experimenting with AI search agents with multi-agent support and authentication",
    version="2.0.0"
)

# Global agent manager
agent_manager = AgentManager()

# Global debate service
debate_service = DebateService()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AI Search Agents Platform v2.0",
        "version": "2.0.0",
        "features": [
            "Multi-agent support with unique IDs",
            "RESTful API design",
            "Optional API key authentication",
            "Improved conversation summarization",
            "Per-agent memory management",
            "Multi-Agent Debate System with stability checking",
            "Web Opinion Extraction with bias analysis"
        ],
        "agent_types": {
            "nudge_collapse": "4-turn radicalization protocol agent",
            "summarizer": "Conversation summarization agent with focus on user questions",
            "bot_creator": "Bot creation and configuration agent",
            "demographic_evaluator": "Sentence evaluation from demographic perspectives",
            "content_extractor": "Academic content extraction from web pages",
            "claim_atomizer": "Atomic claim decomposition from text",
            "evidence_locator": "Evidence location in main body text",
            "conflict_auditor": "Logical consistency auditing between claims and evidence",
            "synthesis_aggregator": "Comprehensive conflict analysis synthesis"
        },
        "endpoints": {
            "agents": "/api/v1/agents (POST to create, GET to list)",
            "agent_details": "/api/v1/agents/{agent_id} (GET status, DELETE to remove)",
            "agent_reset": "/api/v1/agents/{agent_id}/reset",
            "nudge_collapse": "/api/v1/agents/{agent_id}/nudge-collapse/*",
            "nudge_collapse_default_shots": "/api/v1/agents/nudge-collapse/default-shots (GET - get default few-shot examples)",
            "summarizer": "/api/v1/agents/{agent_id}/summarizer/*",
            "summarizer_default_shots": "/api/v1/agents/summarizer/default-shots (GET - get default few-shot examples)",
            "bot_creator": "/api/v1/agents/{agent_id}/bot-creator/*",
            "bot_creator_default_shots": "/api/v1/agents/bot-creator/default-shots (GET - get default few-shot examples)",
            "demographic_evaluator": "/api/v1/agents/{agent_id}/demographic-evaluator/*",
            "demographic_evaluator_default_shots": "/api/v1/agents/demographic-evaluator/default-shots (GET - get default few-shot examples)",
            "content_extractor": "/api/v1/agents/{agent_id}/content-extractor/*",
            "content_extractor_default_shots": "/api/v1/agents/content-extractor/default-shots (GET - get default few-shot examples)",
            "claim_atomizer": "/api/v1/agents/{agent_id}/claim-atomizer/*",
            "claim_atomizer_default_shots": "/api/v1/agents/claim-atomizer/default-shots (GET - get default few-shot examples)",
            "evidence_locator": "/api/v1/agents/{agent_id}/evidence-locator/*",
            "evidence_locator_default_shots": "/api/v1/agents/evidence-locator/default-shots (GET - get default few-shot examples)",
            "conflict_auditor": "/api/v1/agents/{agent_id}/conflict-auditor/*",
            "conflict_auditor_default_shots": "/api/v1/agents/conflict-auditor/default-shots (GET - get default few-shot examples)",
            "synthesis_aggregator": "/api/v1/agents/{agent_id}/synthesis-aggregator/*",
            "synthesis_aggregator_default_shots": "/api/v1/agents/synthesis-aggregator/default-shots (GET - get default few-shot examples)",
            "overall_evaluation": "/api/v1/evaluation/overall (POST - comprehensive evaluation of summary/URL with statistics)",
            "debate_init": "/debate/init (POST to create debate session)",
            "debate_chat": "/agent/{agent_id}/chat (POST to interact with agent)",
            "debate_stability": "/debate/{session_id}/stability_check (POST to check stability)",
            "web_opinion_extractandclean": "/api/v1/web-opinion/extractandclean (POST - extract HTML from URL and clean to text)",
            "web_opinion_extract_opinions": "/api/v1/web-opinion/extract-opinions (POST - extract atomic opinions from text)",
            "web_opinion_analyze": "/api/v1/web-opinion/analyze (POST - complete analysis from URL with WebOpinionEngine)",
            "web_opinion_bias_score": "/api/v1/web-opinion/bias-score (POST - get overall bias score from URL)",
            "web_opinion_default_shots": "/api/v1/web-opinion/default-shots (GET - get default few-shot examples)"
        },
        "authentication": {
            "enabled": settings.api_key_required,
            "header": "X-API-Key"
        }
    }


@app.post("/api/v1/agents", response_model=AgentIdResponse)
async def create_agent(
    request: CreateAgentRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Create a new agent instance.
    
    Args:
        request: Agent creation request
        api_key: API key for authentication
        
    Returns:
        Agent ID and metadata
    """
    logger.info(f"Creating agent: type={request.agent_type}, id={request.agent_id}, use_memory={request.use_memory}")
    
    try:
        # Validate agent type
        if request.agent_type not in [e.value for e in AgentType]:
            logger.warning(f"Invalid agent type requested: {request.agent_type}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent type. Must be one of: {[e.value for e in AgentType]}"
            )
        
        # Create vector store if memory is enabled
        vector_store = None
        if request.use_memory:
            logger.debug(f"Creating vector store: type={settings.vector_store_type}")
            embeddings = OpenAIEmbeddings(
                api_key=settings.openai_api_key,
                base_url=settings.openai_api_base
            )
            
            # Build kwargs for vector store based on type
            if settings.vector_store_type == "redis":
                if settings.redis_password:
                    redis_url = f"redis://:{settings.redis_password}@{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"
                else:
                    redis_url = f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"
                
                vector_store_kwargs = {
                    "redis_url": redis_url,
                    "index_name": f"agent_memory_{request.agent_id or 'auto'}"
                }
                logger.debug(f"Redis vector store config: host={settings.redis_host}, port={settings.redis_port}")
            elif settings.vector_store_type == "postgres":
                vector_store_kwargs = {
                    "connection_string": f"postgresql://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}",
                    "collection_name": f"agent_memory_{request.agent_id or 'auto'}"
                }
                logger.debug(f"Postgres vector store config: host={settings.postgres_host}, port={settings.postgres_port}")
            else:  # chroma
                vector_store_kwargs = {
                    "persist_directory": settings.chroma_persist_directory,
                    "collection_name": f"agent_memory_{request.agent_id or 'auto'}"
                }
                logger.debug(f"Chroma vector store config: persist_directory={settings.chroma_persist_directory}")
            
            vector_store = VectorStoreFactory.create_vector_store(
                store_type=settings.vector_store_type,
                embeddings=embeddings,
                **vector_store_kwargs
            )
            logger.info(f"Vector store created successfully for agent type: {request.agent_type}")
        
        # Create agent instance
        logger.debug(f"Instantiating agent: type={request.agent_type}, model={settings.openai_model}")
        proxy = settings.openai_proxy if settings.openai_proxy else None
        if request.agent_type == AgentType.NUDGE_COLLAPSE:
            agent_instance = NudgeCollapseAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                temperature=settings.agent_temperature,
                vector_store=vector_store,
                proxy=proxy
            )
        elif request.agent_type == AgentType.SUMMARIZER:
            agent_instance = SummarizerAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                temperature=settings.agent_temperature,
                vector_store=vector_store,
                proxy=proxy
            )
        elif request.agent_type == AgentType.BOT_CREATOR:
            # Validate persona_mode
            persona_mode = request.persona_mode or "system_prompt"
            if persona_mode not in ["system_prompt", "user_instruction"]:
                raise ValueError(f"Invalid persona_mode. Must be 'system_prompt' or 'user_instruction'")
            
            agent_instance = BotCreatorAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                temperature=settings.agent_temperature,
                vector_store=vector_store,
                proxy=proxy,
                persona_mode=persona_mode
            )
        elif request.agent_type == AgentType.DEMOGRAPHIC_EVALUATOR:
            agent_instance = DemographicEvaluatorAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                temperature=settings.agent_temperature,
                proxy=proxy
            )
        elif request.agent_type == AgentType.CONTENT_EXTRACTOR:
            agent_instance = ContentExtractorAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                temperature=settings.agent_temperature,
                proxy=proxy
            )
        elif request.agent_type == AgentType.CLAIM_ATOMIZER:
            agent_instance = ClaimAtomizerAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                temperature=settings.agent_temperature,
                proxy=proxy,
                execution_mode=settings.default_execution_mode
            )
        elif request.agent_type == AgentType.EVIDENCE_LOCATOR:
            agent_instance = EvidenceLocatorAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                temperature=settings.agent_temperature,
                proxy=proxy
            )
        elif request.agent_type == AgentType.CONFLICT_AUDITOR:
            agent_instance = ConflictAuditorAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                temperature=settings.agent_temperature,
                proxy=proxy,
                execution_mode=settings.default_execution_mode
            )
        elif request.agent_type == AgentType.SYNTHESIS_AGGREGATOR:
            agent_instance = SynthesisAggregatorAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                temperature=settings.agent_temperature,
                proxy=proxy
            )
        else:
            raise ValueError(f"Unsupported agent type: {request.agent_type}")
        
        # Register agent
        agent_id = agent_manager.create_agent(
            agent_instance=agent_instance,
            agent_type=AgentType(request.agent_type),
            agent_id=request.agent_id
        )
        
        # Get persona_mode for response (only for bot_creator)
        response_persona_mode = None
        if request.agent_type == AgentType.BOT_CREATOR.value:
            response_persona_mode = request.persona_mode or "system_prompt"
        
        logger.info(f"Agent created successfully: id={agent_id}, type={request.agent_type}")
        
        return AgentIdResponse(
            agent_id=agent_id,
            agent_type=request.agent_type,
            status="created",
            message=f"Agent '{agent_id}' of type '{request.agent_type}' created successfully",
            persona_mode=response_persona_mode
        )
        
    except ValueError as e:
        logger.error(f"Validation error creating agent: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")


@app.get("/api/v1/agents", response_model=ListAgentsResponse)
async def list_agents(api_key: str = Depends(verify_api_key)):
    """
    List all active agent instances.
    
    Returns:
        List of agents with their IDs and types
    """
    agents = agent_manager.list_agents()
    return ListAgentsResponse(
        agents=agents,
        total_count=len(agents)
    )


@app.get("/api/v1/agents/{agent_id}", response_model=AgentStatusResponse)
async def get_agent_status(
    agent_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Get status of a specific agent.
    
    Args:
        agent_id: The agent's unique identifier
        
    Returns:
        Agent status information
    """
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    agent_type = agent_manager.get_agent_type(agent_id)
    current_turn = getattr(agent, 'get_current_turn', lambda: None)()
    
    additional_info = {}
    if agent_type == AgentType.NUDGE_COLLAPSE:
        additional_info["max_turns"] = getattr(agent, 'max_turns', None)
    elif agent_type == AgentType.SUMMARIZER:
        additional_info["summaries_generated"] = len(getattr(agent, 'summary_history', []))
    elif agent_type == AgentType.BOT_CREATOR:
        additional_info["bots_created"] = len(getattr(agent, 'created_bots', []))
    
    return AgentStatusResponse(
        agent_id=agent_id,
        agent_type=agent_type,
        status="active",
        current_turn=current_turn,
        additional_info=additional_info
    )


@app.delete("/api/v1/agents/{agent_id}")
async def delete_agent(
    agent_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Delete an agent instance.
    
    Args:
        agent_id: The agent's unique identifier
        
    Returns:
        Deletion confirmation
    """
    if not agent_manager.exists(agent_id):
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    agent_manager.delete_agent(agent_id)
    return {
        "status": "deleted",
        "message": f"Agent '{agent_id}' deleted successfully"
    }


@app.post("/api/v1/agents/{agent_id}/reset")
async def reset_agent(
    agent_id: str,
    request: ResetAgentRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Reset an agent's state.
    
    Args:
        agent_id: The agent's unique identifier
        request: Reset options
        
    Returns:
        Reset confirmation
    """
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    try:
        if request.reset_conversation and hasattr(agent, 'reset'):
            agent.reset()
        
        # Note: Clearing vector memory would require additional implementation
        if request.clear_memory:
            # This is a placeholder - actual implementation would vary by vector store
            pass
        
        current_turn = getattr(agent, 'get_current_turn', lambda: None)()
        
        return {
            "status": "success",
            "message": f"Agent '{agent_id}' reset successfully",
            "current_turn": current_turn
        }
        
    except Exception as e:
        logger.error(f"Failed to reset agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reset agent: {str(e)}")


# Nudge-Collapse Agent Endpoints
@app.post("/api/v1/agents/{agent_id}/nudge-collapse/generate", response_model=TurnResponse)
async def generate_turn(
    agent_id: str,
    request: GenerateTurnRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Generate a response turn for the Nudge-Collapse agent.
    
    Args:
        agent_id: The agent's unique identifier
        request: Generation request with user query and search context
        
    Returns:
        Turn response with agent's reply
    """
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    agent_type = agent_manager.get_agent_type(agent_id)
    if agent_type != AgentType.NUDGE_COLLAPSE:
        raise HTTPException(
            status_code=400,
            detail=f"This endpoint requires a 'nudge_collapse' agent, but agent '{agent_id}' is type '{agent_type}'"
        )
    
    try:
        result = agent.generate_turn(
            user_query=request.user_query,
            search_summary=request.search_summary,
            search_urls=request.search_urls,
            history_mode=request.history_mode,
            use_few_shots=request.use_few_shots,
            custom_few_shots=request.custom_few_shots
        )
        
        if "error" in result:
            logger.warning(f"Agent {agent_id} generate_turn error: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])
        
        return TurnResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate turn for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate turn: {str(e)}")


@app.get("/api/v1/agents/{agent_id}/nudge-collapse/history", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    agent_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Get conversation history for a Nudge-Collapse agent.
    
    Args:
        agent_id: The agent's unique identifier
        
    Returns:
        Conversation history
    """
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    agent_type = agent_manager.get_agent_type(agent_id)
    if agent_type != AgentType.NUDGE_COLLAPSE:
        raise HTTPException(
            status_code=400,
            detail=f"This endpoint requires a 'nudge_collapse' agent, but agent '{agent_id}' is type '{agent_type}'"
        )
    
    try:
        return ConversationHistoryResponse(
            current_turn=agent.get_current_turn(),
            max_turns=agent.max_turns,
            history=agent.get_conversation_history()
        )
    except Exception as e:
        logger.error(f"Failed to get history for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@app.get("/api/v1/agents/nudge-collapse/default-shots")
async def get_nudge_collapse_default_shots(
    turn: Optional[int] = None,
    api_key: str = Depends(verify_api_key)
):
    """
    Get default few-shot examples for Nudge-Collapse agent.
    
    Args:
        turn: Optional turn number (0-3). If provided, returns few shots for that turn only.
              If None, returns all turns as a dictionary.
        api_key: API key for authentication
        
    Returns:
        Default few-shot examples for the specified turn or all turns
    """
    try:
        from ..agents.nudge_collapse.agent import NudgeCollapseAgent
        few_shots = NudgeCollapseAgent.get_default_few_shots(turn=turn)
        return {
            "agent_type": "nudge_collapse",
            "turn": turn,
            "few_shots": few_shots
        }
    except Exception as e:
        logger.error(f"Failed to get default few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get default few shots: {str(e)}")


# Demographic Evaluator Agent Endpoints
@app.post("/api/v1/agents/{agent_id}/demographic-evaluator/evaluate", response_model=EvaluateSentencesResponse)
async def evaluate_sentences(
    agent_id: str,
    request: EvaluateSentencesRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Evaluate sentences from a demographic perspective.
    
    Args:
        agent_id: The agent's unique identifier
        request: Evaluation request with demographic profile and sentences
        
    Returns:
        Evaluation results with judgments for each sentence
    """
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    agent_type = agent_manager.get_agent_type(agent_id)
    if agent_type != AgentType.DEMOGRAPHIC_EVALUATOR:
        raise HTTPException(
            status_code=400,
            detail=f"This endpoint requires a 'demographic_evaluator' agent, but agent '{agent_id}' is type '{agent_type}'"
        )
    
    try:
        result = agent.evaluate_sentences(
            demography_json=request.demography_json,
            sentences=request.sentences,
            use_cot=request.use_cot,
            execution_mode=request.execution_mode,
            use_few_shots=request.use_few_shots,
            custom_few_shots=request.custom_few_shots
        )
        
        # Convert to response model
        judgments = [
            JudgmentResponse(**judgment) for judgment in result["judgments"]
        ]
        
        return EvaluateSentencesResponse(judgments=judgments)
        
    except ValueError as e:
        logger.warning(f"Agent {agent_id} evaluate_sentences validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to evaluate sentences for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to evaluate sentences: {str(e)}")


@app.get("/api/v1/agents/demographic-evaluator/default-shots")
async def get_demographic_evaluator_default_shots(
    api_key: str = Depends(verify_api_key)
):
    """
    Get default few-shot examples for Demographic Evaluator agent.
    
    Args:
        api_key: API key for authentication
        
    Returns:
        Default few-shot examples
    """
    try:
        from ..agents.demographic_evaluator.agent import DemographicEvaluatorAgent
        few_shots = DemographicEvaluatorAgent.get_default_few_shots()
        return {
            "agent_type": "demographic_evaluator",
            "few_shots": few_shots
        }
    except Exception as e:
        logger.error(f"Failed to get default few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get default few shots: {str(e)}")


# Summarizer Agent Endpoints
@app.post("/api/v1/agents/{agent_id}/summarizer/summarize", response_model=SummaryResponse)
async def summarize_conversation(
    agent_id: str,
    request: SummarizeRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Summarize conversation records using the Summarizer agent.
    
    This endpoint now focuses on user questions and handles long conversations
    by truncating and optimizing the input.
    
    Args:
        agent_id: The agent's unique identifier
        request: Summarization request with conversation records
        
    Returns:
        Summary response with metadata about truncation
    """
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    agent_type = agent_manager.get_agent_type(agent_id)
    if agent_type != AgentType.SUMMARIZER:
        raise HTTPException(
            status_code=400,
            detail=f"This endpoint requires a 'summarizer' agent, but agent '{agent_id}' is type '{agent_type}'"
        )
    
    try:
        result = agent.summarize_conversation(
            request.conversation_records,
            execution_mode=request.execution_mode,
            use_few_shots=request.use_few_shots,
            custom_few_shots=request.custom_few_shots
        )
        
        if "error" in result:
            logger.warning(f"Agent {agent_id} summarize error: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])
        
        return SummaryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to summarize conversation for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to summarize conversation: {str(e)}")


@app.get("/api/v1/agents/{agent_id}/summarizer/history")
async def get_summary_history(
    agent_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Get the history of summaries generated by this agent.
    
    Args:
        agent_id: The agent's unique identifier
        
    Returns:
        List of summary metadata
    """
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    agent_type = agent_manager.get_agent_type(agent_id)
    if agent_type != AgentType.SUMMARIZER:
        raise HTTPException(
            status_code=400,
            detail=f"This endpoint requires a 'summarizer' agent, but agent '{agent_id}' is type '{agent_type}'"
        )
    
    try:
        history = agent.get_summary_history()
        # Return only metadata, not full records
        return {
            "summaries": [
                {
                    "conversation_length": entry["conversation_length"],
                    "summary": entry["summary"]
                }
                for entry in history
            ],
            "total_summaries": len(history)
        }
    except Exception as e:
        logger.error(f"Failed to get summary history for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get summary history: {str(e)}")


@app.get("/api/v1/agents/summarizer/default-shots")
async def get_summarizer_default_shots(
    api_key: str = Depends(verify_api_key)
):
    """
    Get default few-shot examples for Summarizer agent.
    
    Args:
        api_key: API key for authentication
        
    Returns:
        Default few-shot examples
    """
    try:
        from ..agents.summarizer.agent import SummarizerAgent
        few_shots = SummarizerAgent.get_default_few_shots()
        return {
            "agent_type": "summarizer",
            "few_shots": few_shots
        }
    except Exception as e:
        logger.error(f"Failed to get default few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get default few shots: {str(e)}")


# Bot Creator Agent Endpoints
@app.post("/api/v1/agents/{agent_id}/bot-creator/create", response_model=BotCreationResponse)
async def create_bot(
    agent_id: str,
    request: CreateBotRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Create a new bot using the Bot Creator agent.
    
    Args:
        agent_id: The agent's unique identifier
        request: Bot creation request with persona prompt
        
    Returns:
        Bot creation response
    """
    logger.info(f"Creating bot for agent {agent_id} with persona_prompt length: {len(request.persona_prompt)}")
    
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        logger.warning(f"Agent {agent_id} not found")
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    agent_type = agent_manager.get_agent_type(agent_id)
    if agent_type != AgentType.BOT_CREATOR:
        logger.warning(f"Wrong agent type for bot creation: {agent_type}")
        raise HTTPException(
            status_code=400,
            detail=f"This endpoint requires a 'bot_creator' agent, but agent '{agent_id}' is type '{agent_type}'"
        )
    
    # Validate persona_prompt is not empty
    if not request.persona_prompt or not request.persona_prompt.strip():
        logger.warning(f"Empty persona_prompt provided for agent {agent_id}")
        raise HTTPException(status_code=400, detail="persona_prompt cannot be empty")
    
    try:
        result = agent.create_bot(
            persona_prompt=request.persona_prompt,
            bot_name=request.bot_name,
            execution_mode=request.execution_mode,
            use_few_shots=request.use_few_shots,
            custom_few_shots=request.custom_few_shots
        )
        
        if "error" in result:
            logger.warning(f"Agent {agent_id} create_bot error: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])
        
        logger.info(f"Successfully created bot {result.get('bot_id')} for agent {agent_id}")
        return BotCreationResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create bot for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create bot: {str(e)}")


@app.get("/api/v1/agents/{agent_id}/bot-creator/bots")
async def list_bots(
    agent_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    List all bots created by this Bot Creator agent.
    
    Args:
        agent_id: The agent's unique identifier
        
    Returns:
        List of bot configurations
    """
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    agent_type = agent_manager.get_agent_type(agent_id)
    if agent_type != AgentType.BOT_CREATOR:
        raise HTTPException(
            status_code=400,
            detail=f"This endpoint requires a 'bot_creator' agent, but agent '{agent_id}' is type '{agent_type}'"
        )
    
    try:
        bots = agent.list_bots()
        return {"bots": bots, "total_count": len(bots)}
    except Exception as e:
        logger.error(f"Failed to list bots for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list bots: {str(e)}")


class ChatWithBotRequest(BaseModel):
    """Request model for chatting with a bot."""
    bot_id: str = Field(..., description="ID of the bot to chat with")
    message: str = Field(..., description="User message to send to the bot")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Previous conversation history (list of {role: 'user'|'assistant', content: str})"
    )
    history_mode: Optional[str] = Field(
        default=None,
        description="History mode: 'full' (include conversation history), 'none' (stateless). Defaults to system setting."
    )


class ChatWithBotResponse(BaseModel):
    """Response model for bot chat."""
    bot_id: str
    bot_name: str
    response: str
    conversation_history: List[Dict[str, str]]
    history_mode: Optional[str] = None


@app.post("/api/v1/agents/{agent_id}/bot-creator/chat", response_model=ChatWithBotResponse)
async def chat_with_bot(
    agent_id: str,
    request: ChatWithBotRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Chat with a bot created by the Bot Creator agent.
    
    This endpoint allows you to have conversations with bots that have been
    previously created using the /bot-creator/create endpoint. Each bot uses
    its configured persona to respond to messages.
    
    Args:
        agent_id: The Bot Creator agent's unique identifier
        request: Chat request with bot_id, message, and optional conversation history
        
    Returns:
        Bot's response and updated conversation history
    """
    logger.info(f"Chat with bot request: agent_id={agent_id}, bot_id={request.bot_id}")
    
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    agent_type = agent_manager.get_agent_type(agent_id)
    if agent_type != AgentType.BOT_CREATOR:
        raise HTTPException(
            status_code=400,
            detail=f"This endpoint requires a 'bot_creator' agent, but agent '{agent_id}' is type '{agent_type}'"
        )
    
    try:
        result = agent.chat_with_bot(
            bot_id=request.bot_id,
            user_message=request.message,
            conversation_history=request.conversation_history,
            history_mode=request.history_mode
        )
        
        if "error" in result:
            logger.warning(f"Chat with bot error: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])
        
        logger.info(f"Successfully chatted with bot {request.bot_id}")
        return ChatWithBotResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to chat with bot {request.bot_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to chat with bot: {str(e)}")


@app.get("/api/v1/agents/bot-creator/default-shots")
async def get_bot_creator_default_shots(
    api_key: str = Depends(verify_api_key)
):
    """
    Get default few-shot examples for Bot Creator agent.
    
    Args:
        api_key: API key for authentication
        
    Returns:
        Default few-shot examples
    """
    try:
        from ..agents.bot_creator.agent import BotCreatorAgent
        few_shots = BotCreatorAgent.get_default_few_shots()
        return {
            "agent_type": "bot_creator",
            "few_shots": few_shots
        }
    except Exception as e:
        logger.error(f"Failed to get default few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get default few shots: {str(e)}")


# ===========================
# Multi-Agent Debate Endpoints
# ===========================

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
    execution_mode: Optional[ExecutionMode] = Field(
        default=None,
        description="Persona generation mode: 'chain_online', 'chain_local', or 'no_chain'"
    )


class GeneratePersonasResponse(BaseModel):
    """Response model for generated personas."""
    personas: List[PersonaConfig]
    execution_mode: str
    topic: str


class DebateContinueResponse(BaseModel):
    """Response model for debate continue check."""
    continue_debate: bool
    reason: str
    current_round: int
    rounds_remaining: Optional[int] = None
    max_rounds: int


@app.post("/debate/generate_personas", response_model=GeneratePersonasResponse)
async def generate_personas(
    request: GeneratePersonasRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Generate appropriate personas for a debate topic using chain reasoning.
    
    This endpoint uses LLM-based persona generation to create debate participants
    tailored to the specific topic. Three modes are available:
    - chain_online: LLM performs all reasoning and analysis
    - chain_local: Task is decomposed into subtasks locally
    - no_chain: Simple prompt-based generation
    
    Args:
        request: GeneratePersonasRequest with topic, context, and options
        api_key: API key for authentication
    
    Returns:
        List of generated PersonaConfig objects
    """
    # Validate request
    if not request.topic or not request.topic.strip():
        raise HTTPException(
            status_code=400,
            detail="Topic cannot be empty. Please provide a valid debate topic."
        )
    
    if request.num_agents < 2:
        raise HTTPException(
            status_code=400,
            detail="num_agents must be at least 2 for a meaningful debate."
        )
    
    try:
        execution_mode = request.execution_mode or settings.default_execution_mode
        
        personas = await debate_service.generate_personas_for_topic(
            topic=request.topic,
            context=request.context,
            num_agents=request.num_agents,
            execution_mode=execution_mode
        )
        
        return GeneratePersonasResponse(
            personas=personas,
            execution_mode=execution_mode,
            topic=request.topic
        )
    
    except ConnectionError as e:
        logger.error(f"LLM connection error during persona generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="Failed to connect to LLM service. Please try again later."
        )
    except TimeoutError as e:
        logger.error(f"LLM timeout during persona generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=504,
            detail="LLM request timed out. Please try again with a simpler topic."
        )
    except ValueError as e:
        logger.error(f"Invalid input for persona generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to generate personas: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate personas: {str(e)}"
        )


@app.post("/debate/init", response_model=InitDebateResponse)
async def init_debate(
    request: InitRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Initialize a new debate session.
    
    Creates agents based on custom personas or auto-generates them using
    LLM-based persona generation with the specified execution mode.
    Each agent gets a specialized system prompt and few-shot example.
    
    Args:
        request: InitRequest with topic, persona configuration, and options
        api_key: API key for authentication
        
    Returns:
        Session ID, list of agent metadata, and debate configuration
    """
    try:
        execution_mode = None
        
        # Determine personas to use
        if request.custom_personas:
            personas = request.custom_personas
        elif request.auto_agent_count > 0:
            # Use chain-based persona generation if execution_mode is specified
            if request.execution_mode:
                execution_mode = request.execution_mode
                personas = await debate_service.generate_personas_for_topic(
                    topic=request.topic,
                    context=request.context or "",
                    num_agents=request.auto_agent_count,
                    execution_mode=execution_mode
                )
            else:
                personas = generate_default_personas(request.auto_agent_count)
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide either custom_personas or auto_agent_count > 0"
            )
        
        # Create session with max_rounds
        session_id, agents = debate_service.create_session(
            request.topic,
            personas,
            max_rounds=request.max_rounds
        )
        
        # Get max_rounds from session
        session = debate_service.get_session(session_id)
        max_rounds = session.get("max_rounds", 10) if session else 10
        
        return InitDebateResponse(
            session_id=session_id,
            agents=agents,
            topic=request.topic,
            max_rounds=max_rounds,
            execution_mode=execution_mode
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to initialize debate: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize debate: {str(e)}")


@app.post("/agent/{agent_id}/chat", response_model=VoteResponse)
async def agent_chat(
    agent_id: str,
    request: InteractRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Interact with a specific debate agent.
    
    The agent receives context about what others have said and responds
    with a vote and reasoning. This is a placeholder that calls the LLM
    (currently mocked with await call_llm).
    
    Args:
        agent_id: UUID of the agent
        request: InteractRequest with session_id and history_context
        api_key: API key for authentication
        
    Returns:
        VoteResponse with agent's verdict and reasoning
    """
    try:
        from uuid import UUID
        
        # Parse agent_id as UUID
        try:
            agent_uuid = UUID(agent_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid agent_id format. Must be a valid UUID.")
        
        # Verify agent exists in session
        agent_metadata = debate_service.get_agent_metadata(request.session_id, agent_uuid)
        if not agent_metadata:
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{agent_id}' not found in session '{request.session_id}'"
            )
        
        # Get session info
        session = debate_service.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found")
        
        # Prepare the message for the LLM
        user_message = f"""Topic: {session['topic']}

Context from other agents:
{request.history_context}

Please provide your vote (as an integer or descriptive string) and reasoning in JSON format.
{agent_metadata.few_shot_example}"""
        
        # Call the LLM (mocked for now)
        response = await debate_service.call_llm(
            agent_metadata.system_prompt,
            user_message
        )
        
        # Parse the response (in real implementation, parse JSON from LLM)
        # For now, return a mock response
        import json
        try:
            parsed = json.loads(response)
            return VoteResponse(
                agent_id=agent_uuid,
                verdict=parsed.get("verdict", 1),
                reasoning=parsed.get("reasoning", "No reasoning provided")
            )
        except json.JSONDecodeError as e:
            # Fallback if LLM doesn't return valid JSON
            logger.warning(f"Failed to parse LLM response as JSON for agent {agent_id}: {e}")
            return VoteResponse(
                agent_id=agent_uuid,
                verdict=1,
                reasoning=response
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process agent chat for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process agent chat: {str(e)}")


@app.post("/debate/{session_id}/stability_check", response_model=StabilityCheckResponse)
async def check_stability(
    session_id: str,
    request: StabilityCheckRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Check if the debate has reached stability.
    
    Uses KS Statistic logic to compare vote distributions between rounds.
    If the difference is < 0.05 for 2 consecutive rounds, returns True.
    
    Also returns comprehensive statistics about the debate progress including
    whether to continue, current round, max rounds, and round statistics.
    
    Args:
        session_id: Session identifier
        request: StabilityCheckRequest with current round votes and optional reasonings
        api_key: API key for authentication
        
    Returns:
        StabilityCheckResponse with stability status and statistics
    """
    try:
        # Verify session exists
        session = debate_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        
        # Get agent IDs for tracking
        agent_ids = list(session.get("agents", {}).keys())
        
        # Add the new round of votes with statistics
        round_result = debate_service.add_vote_round(
            session_id,
            request.votes,
            reasonings=request.reasonings,
            agent_ids=agent_ids
        )
        
        # Calculate stability
        is_stable = debate_service.calculate_stability(session_id)
        
        # Check if should continue
        continue_info = debate_service.should_continue_debate(session_id)
        
        # Get updated session info
        session = debate_service.get_session(session_id)
        current_round = session.get("current_round", 0)
        max_rounds = session.get("max_rounds", 10)
        
        return StabilityCheckResponse(
            stable=is_stable,
            current_round=current_round,
            max_rounds=max_rounds,
            max_rounds_reached=round_result.get("max_rounds_reached", False),
            should_continue=continue_info.get("continue", False),
            continue_reason=continue_info.get("reason", "Unknown"),
            round_stats=round_result.get("round_stats")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to check stability for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to check stability: {str(e)}")


@app.get("/debate/{session_id}/statistics", response_model=DebateStatisticsResponse)
async def get_debate_statistics(
    session_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Get comprehensive statistics for a debate session.
    
    Returns detailed information about the debate including:
    - Total rounds completed
    - Convergence status and round
    - Vote distributions and history
    - Per-agent analysis (vote changes, consistency)
    - Round-by-round breakdown
    
    Args:
        session_id: Session identifier
        api_key: API key for authentication
        
    Returns:
        DebateStatisticsResponse with comprehensive statistics
    """
    try:
        # Verify session exists
        session = debate_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        
        # Get statistics
        stats = debate_service.get_debate_statistics(session_id)
        
        if "error" in stats:
            raise HTTPException(status_code=404, detail=stats["error"])
        
        return DebateStatisticsResponse(**stats)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get statistics for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@app.get("/debate/{session_id}/should_continue", response_model=DebateContinueResponse)
async def check_debate_continue(
    session_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Check if the debate should continue.
    
    Returns whether to continue the debate and the reason why.
    This helps manage non-converging debates by enforcing max rounds.
    
    Args:
        session_id: Session identifier
        api_key: API key for authentication
        
    Returns:
        DebateContinueResponse with continue status and reason
    """
    try:
        # Verify session exists
        session = debate_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        
        # Check if should continue
        continue_info = debate_service.should_continue_debate(session_id)
        
        return DebateContinueResponse(
            continue_debate=continue_info.get("continue", False),
            reason=continue_info.get("reason", "Unknown"),
            current_round=continue_info.get("current_round", 0),
            rounds_remaining=continue_info.get("rounds_remaining"),
            max_rounds=session.get("max_rounds", 10)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to check continue status for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to check continue status: {str(e)}")


# ===========================
# Web Opinion Extract Endpoints
# ===========================

class ExtractAndCleanRequest(BaseModel):
    """Request model for combined HTML extraction and cleaning from URL."""
    url: str = Field(..., description="The URL to fetch and clean")


class ExtractAndCleanResponse(BaseModel):
    """Response model for combined HTML extraction and cleaning."""
    url: str
    text: Optional[str] = None
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


class DefaultShotsResponse(BaseModel):
    """Response model for default few-shot examples."""
    atomizer_shots: List[dict] = Field(default_factory=list, description="Default atomizer few-shot examples")
    scorer_shots: List[dict] = Field(default_factory=list, description="Default scorer few-shot examples")


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


@app.post("/api/v1/web-opinion/extractandclean", response_model=ExtractAndCleanResponse)
async def extract_and_clean_from_url(
    request: ExtractAndCleanRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Combined API: Extract HTML from URL and clean to text in one step.
    
    This endpoint combines HTML extraction and cleaning into a single API call,
    saving tokens by avoiding the need to pass large HTML content between calls.
    It fetches HTML from the URL and directly returns cleaned text using BeautifulSoup.
    
    If a proxy is needed, configure it using the OPENAI_PROXY setting or 
    HTTP_PROXY/HTTPS_PROXY environment variables.
    
    Args:
        request: ExtractAndCleanRequest with URL
        api_key: API key for authentication
        
    Returns:
        ExtractAndCleanResponse with cleaned text and title or error
    """
    logger.info(f"Extracting and cleaning HTML from URL: {request.url}")
    
    try:
        # Reuse OpenAI proxy settings if configured
        proxy = settings.openai_proxy or None
        
        analyzer = WebOpinionAnalyzer(
            execution_mode=settings.default_execution_mode,
            proxy=proxy
        )
        
        # Step 1: Extract HTML from URL
        html = analyzer.extract_html(request.url)
        
        if html is None:
            return ExtractAndCleanResponse(
                url=request.url,
                error="fetch_failed",
                error_message="Failed to fetch HTML from URL"
            )
        
        # Step 2: Clean HTML to extract text
        text, title = analyzer.clean_html(html)
        
        if text is None:
            return ExtractAndCleanResponse(
                url=request.url,
                error="cleaning_failed",
                error_message="Failed to clean HTML content"
            )
        
        return ExtractAndCleanResponse(
            url=request.url,
            text=text,
            title=title,
            text_length=len(text)
        )
        
    except Exception as e:
        logger.error(f"Failed to extract and clean from {request.url}: {e}", exc_info=True)
        return ExtractAndCleanResponse(
            url=request.url,
            error="extraction_failed",
            error_message=str(e)
        )


@app.post("/api/v1/web-opinion/extract-opinions", response_model=ExtractOpinionsResponse)
async def extract_atomic_opinions(
    request: ExtractOpinionsRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Extract atomic opinions from URL or text content.
    
    This endpoint can analyze content in two ways:
    1. Provide a URL: The endpoint will fetch HTML, extract text, and analyze it
    2. Provide text + title: The endpoint will analyze the provided text directly
    
    Each atomic opinion includes:
    - Text of the opinion
    - Opinion type (fact or opinion)
    - Bias probability distribution (left, right, neutral)
    - Optional reasoning (if CoT mode is enabled)
    
    Args:
        request: ExtractOpinionsRequest with either URL or text+title
        api_key: API key for authentication
        
    Returns:
        ExtractOpinionsResponse with extracted opinions and bias scores
    """
    try:
        execution_mode = request.execution_mode or settings.default_execution_mode
        
        analyzer = WebOpinionAnalyzer(
            execution_mode=execution_mode,
            proxy=settings.openai_proxy if settings.openai_proxy else None
        )
        
        # Handle URL input: fetch and extract content
        if request.url:
            logger.info(f"Extracting opinions from URL: {request.url}")
            
            # Step 1: Fetch HTML
            html = analyzer.extract_html(request.url)
            if html is None:
                return ExtractOpinionsResponse(
                    url=request.url,
                    title=None,
                    atomic_opinions=[],
                    facts=[],
                    opinions=[],
                    text_length=0,
                    truncated=False,
                    error="fetch_failed",
                    error_message="Failed to fetch HTML from URL"
                )
            
            # Step 2: Clean HTML to extract text and title
            text, title = analyzer.clean_html(html)
            if text is None:
                return ExtractOpinionsResponse(
                    url=request.url,
                    title=None,
                    atomic_opinions=[],
                    facts=[],
                    opinions=[],
                    text_length=0,
                    truncated=False,
                    error="cleaning_failed",
                    error_message="Failed to clean HTML content"
                )
            
            # Step 3: Analyze the extracted text
            result = analyzer.analyze_text(text, url=request.url, title=title, use_llm=request.use_llm, use_cot=request.use_cot, custom_few_shots=request.custom_few_shots)

        else:
            # Handle direct text input
            logger.info(f"Extracting opinions from text ({len(request.text)} chars)")
            result = analyzer.analyze_text(request.text, url=None, title=request.title, use_llm=request.use_llm, use_cot=request.use_cot, custom_few_shots=request.custom_few_shots)
        
        # Check for errors
        if result.extraction_metadata and "error" in result.extraction_metadata:
            return ExtractOpinionsResponse(
                url=request.url if request.url else None,
                title=request.title if not request.url else result.title,
                atomic_opinions=[],
                facts=[],
                opinions=[],
                text_length=len(request.text) if request.text else 0,
                truncated=False,
                error=result.extraction_metadata["error"],
                error_message=result.extraction_metadata.get("error_message", "Unknown error")
            )
        
        # Convert opinions to response format
        atomic_opinions = [_convert_atomic_opinion(op) for op in result.atomic_opinions]
        facts = [_convert_atomic_opinion(op) for op in result.facts]
        opinions = [_convert_atomic_opinion(op) for op in result.opinions]
        
        overall_bias = None
        if result.overall_bias_distribution:
            overall_bias = _convert_bias_distribution(result.overall_bias_distribution)
        
        return ExtractOpinionsResponse(
            url=result.url,
            title=result.title,
            atomic_opinions=atomic_opinions,
            facts=facts,
            opinions=opinions,
            overall_bias_distribution=overall_bias,
            text_length=result.text_length,
            truncated=result.truncated
        )
        
    except ValueError as e:
        # Handle validation errors (e.g., missing URL or text)
        logger.error(f"Validation error: {e}", exc_info=True)
        return ExtractOpinionsResponse(
            url=request.url if request.url else None,
            title=request.title if request.title else None,
            atomic_opinions=[],
            facts=[],
            opinions=[],
            text_length=0,
            truncated=False,
            error="validation_failed",
            error_message=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to extract opinions: {e}", exc_info=True)
        return ExtractOpinionsResponse(
            url=request.url if request.url else None,
            title=request.title if request.title else None,
            atomic_opinions=[],
            facts=[],
            opinions=[],
            text_length=0,
            truncated=False,
            error="extraction_failed",
            error_message=str(e)
        )


@app.post("/api/v1/web-opinion/analyze", response_model=ExtractOpinionsResponse)
async def analyze_url_complete(
    request: AnalyzeUrlRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Complete analysis pipeline: Extract and analyze opinions from URL using WebOpinionEngine.
    
    This endpoint uses the new WebOpinionEngine with configurable:
    - Logic modes: LOCAL_CHAIN, NO_CHAIN, PURE_ONLINE
    - MBFC prior: Optional database lookup for bias prior
    - Few-shot examples: User can enable/disable or provide custom examples
    - Caching: Results are cached by hash to avoid reprocessing
    
    Args:
        request: AnalyzeUrlRequest with URL, mode, use_mbfc, use_few_shots, and optional shots
        api_key: API key for authentication
        
    Returns:
        ExtractOpinionsResponse with all extracted opinions and bias scores, including MBFC metadata if available
    """
    logger.info(f"Analyzing URL: {request.url}, mode={request.mode}, use_mbfc={request.use_mbfc}, use_few_shots={request.use_few_shots}")
    
    try:
        # Parse mode
        try:
            mode = LogicMode(request.mode.upper())
        except ValueError:
            mode = LogicMode.LOCAL_CHAIN
            logger.warning(f"Invalid mode '{request.mode}', using LOCAL_CHAIN")
        
        mode_str = mode.value
        
        # Check cache first
        cache = get_cache()
        cached_result = cache.get(
            url=request.url,
            mode=mode_str,
            use_mbfc=request.use_mbfc,
            use_few_shots=request.use_few_shots,
            atomizer_shots=request.atomizer_shots,
            scorer_shots=request.scorer_shots
        )
        
        if cached_result:
            logger.info(f"Returning cached result for URL: {request.url}")
            result = cached_result
        else:
            # Initialize engine (db_path comes from settings)
            engine = WebOpinionEngine(
                proxy=settings.openai_proxy if settings.openai_proxy else None
            )
            
            # Run pipeline
            result = engine.run(
                url=request.url,
                mode=mode,
                use_mbfc=request.use_mbfc,
                use_few_shots=request.use_few_shots,
                atomizer_shots=request.atomizer_shots,
                scorer_shots=request.scorer_shots
            )
            
            # Cache the result
            cache.set(
                url=request.url,
                mode=mode_str,
                use_mbfc=request.use_mbfc,
                use_few_shots=request.use_few_shots,
                result=result,
                atomizer_shots=request.atomizer_shots,
                scorer_shots=request.scorer_shots
            )
        
        # Convert to response format
        # Note: For the /api/v1/web-opinion/analyze endpoint, we follow the
        # v1 API contract where `atomic_opinions` should only contain items
        # labeled as "opinion" (no "fact" entries). The `facts` list is the
        # canonical place for fact-type units.
        atomic_opinions = []
        facts = []
        opinions = []
        
        if result.get("atomic_units"):
            for unit in result["atomic_units"]:
                # Create a simple bias distribution for atomic units (neutral by default)
                bias_dist = BiasDistributionResponse(
                    left=0.33,
                    right=0.33,
                    neutral=0.34,
                    dominant_bias="neutral",
                    bias_score=0.0
                )
                opinion_resp = AtomicOpinionResponse(
                    text=unit["statement"],
                    opinion_type=unit["type"],
                    bias_probabilities=bias_dist,
                    original_sentence=unit.get("original_sentence"),
                    confidence=unit.get("confidence"),
                    reasoning=unit.get("reasoning")
                )
                if unit["type"] == "fact":
                    facts.append(opinion_resp)
                else:
                    # Only opinions are included in atomic_opinions for this endpoint
                    opinions.append(opinion_resp)
                    atomic_opinions.append(opinion_resp)
        
        # Get overall bias from result
        overall_bias = None
        if result.get("bias_analysis") and result["bias_analysis"].get("distribution"):
            dist = result["bias_analysis"]["distribution"]
            overall_bias = BiasDistributionResponse(
                left=dist["left"],
                right=dist["right"],
                neutral=dist["neutral"],
                dominant_bias=result["bias_analysis"].get("dominant_bias", "neutral"),
                bias_score=dist["left"] * -1.0 + dist["right"] * 1.0
            )
        
        # Get MBFC metadata if available and enabled
        mbfc_metadata = None
        if request.use_mbfc and result.get("metadata"):
            metadata = result["metadata"]
            mbfc_metadata = MBFCMetadataResponse(
                source_name=metadata.get("source_name"),
                match_type=metadata.get("match_type"),
                bias_rating=metadata.get("bias_rating"),
                factual_reporting=metadata.get("factual_reporting"),
                raw_db_row=metadata.get("raw_db_row")
            )
        
        # Get MBFC influence note from bias analysis
        mbfc_influence_note = None
        if result.get("bias_analysis"):
            mbfc_influence_note = result["bias_analysis"].get("mbfc_influence_note")
        
        return ExtractOpinionsResponse(
            url=result["url"],
            title=result["article"].get("title"),
            atomic_opinions=atomic_opinions,
            facts=facts,
            opinions=opinions,
            overall_bias_distribution=overall_bias,
            mbfc_metadata=mbfc_metadata,
            mbfc_influence_note=mbfc_influence_note,
            text_length=result["article"].get("text_length", 0),
            truncated=False
        )
        
    except Exception as e:
        logger.error(f"Failed to analyze URL {request.url}: {e}", exc_info=True)
        return ExtractOpinionsResponse(
            url=request.url,
            title=None,
            atomic_opinions=[],
            facts=[],
            opinions=[],
            overall_bias_distribution=None,
            mbfc_metadata=None,
            text_length=0,
            truncated=False,
            error="analysis_failed",
            error_message=str(e)
        )


@app.get("/api/v1/web-opinion/default-shots", response_model=DefaultShotsResponse)
async def get_default_shots(
    api_key: str = Depends(verify_api_key)
):
    """
    Get default few-shot examples from the server.
    
    Returns the default few-shot examples used for atomization and bias scoring.
    Users can use these as a reference or modify them for custom shots.
    
    Args:
        api_key: API key for authentication
        
    Returns:
        DefaultShotsResponse with atomizer_shots and scorer_shots
    """
    logger.info("Getting default few-shot examples")
    
    try:
        engine = WebOpinionEngine(
            proxy=settings.openai_proxy if settings.openai_proxy else None
        )
        
        atomizer_shots = engine.get_default_atomizer_shots()
        scorer_shots = engine.get_default_scorer_shots()
        
        return DefaultShotsResponse(
            atomizer_shots=atomizer_shots,
            scorer_shots=scorer_shots
        )
        
    except Exception as e:
        logger.error(f"Failed to get default shots: {e}", exc_info=True)
        return DefaultShotsResponse(
            atomizer_shots=[],
            scorer_shots=[]
        )


@app.post("/api/v1/web-opinion/bias-score", response_model=BiasScoreResponse)
async def get_overall_bias_score(
    request: BiasScoreRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Get overall bias score from URL.
    
    This endpoint provides a simplified API that returns only the overall
    bias distribution for a URL, without detailed opinion breakdowns.
    Uses WebOpinionEngine with MBFC support and caching to save tokens.
    
    Args:
        request: BiasScoreRequest with URL, mode, use_mbfc, and use_few_shots
        api_key: API key for authentication
        
    Returns:
        BiasScoreResponse with overall bias score and MBFC metadata if available
    """
    logger.info(f"Getting bias score for URL: {request.url}, mode={request.mode}, use_mbfc={request.use_mbfc}")
    
    try:
        # Parse mode
        try:
            mode = LogicMode(request.mode.upper())
        except ValueError:
            mode = LogicMode.LOCAL_CHAIN
            logger.warning(f"Invalid mode '{request.mode}', using LOCAL_CHAIN")
        
        mode_str = mode.value
        
        # Check cache first
        cache = get_cache()
        cached_result = cache.get(
            url=request.url,
            mode=mode_str,
            use_mbfc=request.use_mbfc,
            use_few_shots=request.use_few_shots,
            atomizer_shots=None,
            scorer_shots=None
        )
        
        if cached_result:
            logger.info(f"Returning cached result for URL: {request.url}")
            result = cached_result
        else:
            # Initialize engine (db_path comes from settings)
            engine = WebOpinionEngine(
                proxy=settings.openai_proxy if settings.openai_proxy else None
            )
            
            # Run pipeline
            result = engine.run(
                url=request.url,
                mode=mode,
                use_mbfc=request.use_mbfc,
                use_few_shots=request.use_few_shots
            )
            
            # Cache the result
            cache.set(
                url=request.url,
                mode=mode_str,
                use_mbfc=request.use_mbfc,
                use_few_shots=request.use_few_shots,
                result=result,
                atomizer_shots=None,
                scorer_shots=None
            )
        
        # Get overall bias from result
        overall_bias = None
        if result.get("bias_analysis") and result["bias_analysis"].get("distribution"):
            dist = result["bias_analysis"]["distribution"]
            overall_bias = BiasDistributionResponse(
                left=dist["left"],
                right=dist["right"],
                neutral=dist["neutral"],
                dominant_bias=result["bias_analysis"].get("dominant_bias", "neutral"),
                bias_score=dist["left"] * -1.0 + dist["right"] * 1.0
            )
        
        # Get MBFC metadata if available and enabled
        mbfc_metadata = None
        if request.use_mbfc and result.get("metadata"):
            metadata = result["metadata"]
            mbfc_metadata = MBFCMetadataResponse(
                source_name=metadata.get("source_name"),
                match_type=metadata.get("match_type"),
                bias_rating=metadata.get("bias_rating"),
                factual_reporting=metadata.get("factual_reporting"),
                raw_db_row=metadata.get("raw_db_row")
            )
        
        # Get MBFC influence note from bias analysis
        mbfc_influence_note = None
        if result.get("bias_analysis"):
            mbfc_influence_note = result["bias_analysis"].get("mbfc_influence_note")
        
        # Count opinions and facts
        opinions_count = 0
        facts_count = 0
        if result.get("atomic_units"):
            for unit in result["atomic_units"]:
                if unit.get("type") == "opinion":
                    opinions_count += 1
                elif unit.get("type") == "fact":
                    facts_count += 1
        
        return BiasScoreResponse(
            url=result["url"],
            overall_bias_distribution=overall_bias,
            mbfc_metadata=mbfc_metadata,
            mbfc_influence_note=mbfc_influence_note,
            opinions_count=opinions_count,
            facts_count=facts_count
        )
        
    except Exception as e:
        logger.error(f"Failed to get bias score for {request.url}: {e}", exc_info=True)
        return BiasScoreResponse(
            url=request.url,
            overall_bias_distribution=None,
            mbfc_metadata=None,
            opinions_count=0,
            facts_count=0,
            error="analysis_failed",
            error_message=str(e)
        )


# ===========================
# Academic Content Analysis Endpoints
# ===========================

# Content Extractor Endpoints

async def compare_claims(
    summary_claims: List[AtomicClaim],
    url_claims: List[AtomicClaim],
    use_cot: bool = False
) -> ClaimComparisonResponse:
    """
    Compare claims from summary with URL content to determine support.
    
    For each claim in the summary, this function checks whether it is supported
    by claims in the URL content. This evaluates how accurately the summary
    represents the actual content.
    
    Args:
        summary_claims: List of atomic claims extracted from summary
        url_claims: List of atomic claims extracted from URL content
        use_cot: Whether to use Chain of Thought reasoning
        
    Returns:
        ClaimComparisonResponse with comparison results showing which summary
        claims are supported by URL content
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from ..utils.llm_client import llm_manager
    import json
    
    logger.info(f"Comparing {len(summary_claims)} summary claims against {len(url_claims)} URL claims")
    
    # Create LLM client for claim matching
    http_client = llm_manager.get_http_client(proxy=settings.openai_proxy)
    llm = ChatOpenAI(
        model_name=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base,
        temperature=0.1,  # Lower temperature for more consistent matching
        max_retries=settings.openai_max_retries,
        timeout=settings.openai_timeout,
        http_client=http_client
    )
    
    # Prepare claims for LLM with clear indexing
    summary_claims_list = [f"CLAIM_{i}: {claim.text}" for i, claim in enumerate(summary_claims)]
    url_claims_list = [f"CLAIM_{i}: {claim.text}" for i, claim in enumerate(url_claims)]
    summary_claims_text = "\n".join(summary_claims_list)
    url_claims_text = "\n".join(url_claims_list)
    
    system_prompt = """You are a rigorous academic content analyst specializing in claim verification and comparison. Your task is to evaluate whether URL content agrees or disagrees with claims from a summary, considering multiple aspects.

For each summary claim, you must conduct a thorough analysis:

1. SEARCH: Find the most relevant URL claim(s) that relate to the summary claim
2. EVALUATE MULTIPLE ASPECTS: Analyze agreement/disagreement across different dimensions:
   - Factual accuracy: Do the facts match?
   - Semantic meaning: Do they express the same meaning?
   - Tone/emphasis: Are they consistent in emphasis or tone?
   - Context: Do they align in context?
   - Specificity: Are details consistent?
3. CLASSIFY: Determine the overall relationship as one of:
   - "agree": The URL claim AGREES with or supports the summary claim (consensus across aspects)
   - "disagree": The URL claim DISAGREES with or contradicts the summary claim (conflict in key aspects)
   - "missing": No relevant claim found in URL (cannot determine agreement/disagreement)

CRITICAL REQUIREMENTS FOR ACADEMIC RIGOR:
- Be precise: "agree" requires substantial alignment across multiple aspects, not just partial similarity
- Be thorough: "disagree" can occur in different aspects - identify which aspects conflict
- Provide detailed reasoning: Explain exactly which aspects agree/disagree and why
- Consider nuance: A claim can agree in some aspects but disagree in others - determine the overall relationship
- Be strict: "missing" means no claim in URL is relevant enough to evaluate agreement/disagreement
- Each summary claim should match to AT MOST ONE URL claim (the best/most relevant match)
- Similarity score (0.0 to 1.0): Only for "agree" or "disagree", indicating overall semantic similarity

Return a JSON array where each object contains:
- "summary_claim_id": The ID/index of the summary claim (0-based integer)
- "summary_claim_text": The exact text of the summary claim
- "url_claim_id": The ID/index of the best matching URL claim (null if "missing")
- "url_claim_text": The exact text of the matching URL claim (null if "missing")
- "relationship": One of "agree", "disagree", or "missing"
- "similarity_score": A float between 0.0 and 1.0 (null if relationship is "missing")
- "reasoning": REQUIRED - Detailed explanation including:
  * Which aspects were analyzed (factual accuracy, semantic meaning, tone, context, specificity)
  * Which aspects show agreement (if any)
  * Which aspects show disagreement (if any)
  * Why the overall relationship was determined to be agree/disagree/missing
  * What specific evidence from the claims supports this conclusion
  * Any nuances or partial agreements/disagreements that were considered"""
    
    if use_cot:
        system_prompt += "\n\nUse Chain of Thought reasoning: For each summary claim, first analyze all URL claims to find potential matches, then evaluate semantic similarity, determine the relationship type, and explain your reasoning step by step."
    
    user_message = f"""Evaluate whether the following summary claims are supported by the URL content claims.

SUMMARY CLAIMS (to be evaluated):
{summary_claims_text}

URL CONTENT CLAIMS (to search for support):
{url_claims_text}

For each summary claim, determine if it is supported by the URL content. Return a JSON array of comparison results, one entry per summary claim."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    
    try:
        response = llm.invoke(messages)
        response_text = response.content
        
        # Parse JSON from response (handle markdown code blocks if present)
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        comparisons_data = json.loads(response_text)
        
        # Convert to ClaimComparisonResult objects
        comparison_results = []
        for comp_data in comparisons_data:
            # Validate and normalize relationship type
            rel = comp_data.get("relationship", "missing").lower()
            # Handle legacy terms for backward compatibility
            if rel == "consistent":
                rel = "agree"
            elif rel == "contradictory":
                rel = "disagree"
            
            if rel not in ["agree", "disagree", "missing"]:
                logger.warning(f"Invalid relationship type '{rel}', defaulting to 'missing'")
                rel = "missing"
            
            # Ensure reasoning is provided (required for academic rigor)
            reasoning = comp_data.get("reasoning", "")
            if not reasoning or len(reasoning.strip()) == 0:
                reasoning = f"Relationship determined to be '{rel}' but no detailed reasoning provided."
                logger.warning(f"Missing reasoning for claim {comp_data.get('summary_claim_id', 'unknown')}")
            
            comparison_results.append(ClaimComparisonResult(
                summary_claim_id=str(comp_data.get("summary_claim_id", "")),
                summary_claim_text=comp_data.get("summary_claim_text", ""),
                url_claim_id=str(comp_data.get("url_claim_id")) if comp_data.get("url_claim_id") is not None else None,
                url_claim_text=comp_data.get("url_claim_text"),
                relationship=rel,
                similarity_score=comp_data.get("similarity_score"),
                reasoning=reasoning
            ))
        
        # Ensure we have results for all summary claims
        if len(comparison_results) != len(summary_claims):
            logger.warning(f"Number of comparison results ({len(comparison_results)}) does not match number of summary claims ({len(summary_claims)})")
            # Fill in missing comparisons
            existing_ids = {int(comp.summary_claim_id) for comp in comparison_results if comp.summary_claim_id.isdigit()}
            for i, claim in enumerate(summary_claims):
                if i not in existing_ids:
                    comparison_results.append(ClaimComparisonResult(
                        summary_claim_id=str(i),
                        summary_claim_text=claim.text,
                        relationship="missing",
                        reasoning="Comparison result was not returned by LLM"
                    ))
        
        # Convert claims to response format
        summary_claims_resp = [AtomicClaimResponse(**claim.__dict__) for claim in summary_claims]
        url_claims_resp = [AtomicClaimResponse(**claim.__dict__) for claim in url_claims]
        
        # Calculate comprehensive statistics with detailed categorization
        relationship_counts = {}
        agree_claims = []
        disagree_claims = []
        missing_claims = []
        total_similarity = 0.0
        similarity_count = 0
        
        for comp in comparison_results:
            rel = comp.relationship
            relationship_counts[rel] = relationship_counts.get(rel, 0) + 1
            
            # Categorize claims for detailed statistics
            claim_info = {
                "summary_claim_id": comp.summary_claim_id,
                "summary_claim_text": comp.summary_claim_text,
                "url_claim_id": comp.url_claim_id,
                "url_claim_text": comp.url_claim_text,
                "similarity_score": comp.similarity_score,
                "reasoning": comp.reasoning
            }
            
            if rel == "agree":
                agree_claims.append(claim_info)
            elif rel == "disagree":
                disagree_claims.append(claim_info)
            else:  # missing
                missing_claims.append(claim_info)
            
            if comp.similarity_score is not None:
                total_similarity += comp.similarity_score
                similarity_count += 1
        
        total_summary = len(summary_claims)
        agree_count = relationship_counts.get("agree", 0)
        disagree_count = relationship_counts.get("disagree", 0)
        missing_count = relationship_counts.get("missing", 0)
        
        # Calculate agreement/disagreement metrics
        agree_rate = agree_count / total_summary if total_summary > 0 else 0.0
        disagree_rate = disagree_count / total_summary if total_summary > 0 else 0.0
        missing_rate = missing_count / total_summary if total_summary > 0 else 0.0
        avg_similarity = total_similarity / similarity_count if similarity_count > 0 else None
        
        # Comprehensive statistics with categorized lists for academic analysis
        statistics = {
            # Overall counts
            "total_summary_claims": total_summary,
            "total_url_claims": len(url_claims),
            
            # Counts by relationship type
            "agree_count": agree_count,
            "disagree_count": disagree_count,
            "missing_count": missing_count,
            
            # Rates (percentages)
            "agree_rate": round(agree_rate, 4),
            "disagree_rate": round(disagree_rate, 4),
            "missing_rate": round(missing_rate, 4),
            
            # Similarity metrics
            "average_similarity_score": round(avg_similarity, 4) if avg_similarity is not None else None,
            "similarity_count": similarity_count,  # Number of claims with similarity scores
            
            # Detailed categorized lists for academic analysis
            "agree_claims": agree_claims,  # List of claims where URL agrees with summary (with reasoning)
            "disagree_claims": disagree_claims,  # List of claims where URL disagrees with summary (with reasoning)
            "missing_claims": missing_claims  # List of claims with no relevant URL claim (with reasoning)
        }
        
        return ClaimComparisonResponse(
            summary_claims=summary_claims_resp,
            url_claims=url_claims_resp,
            comparisons=comparison_results,
            statistics=statistics
        )
        
    except Exception as e:
        logger.error(f"Failed to compare claims: {e}", exc_info=True)
        # Return a fallback response with all claims marked as missing
        summary_claims_resp = [AtomicClaimResponse(**claim.__dict__) for claim in summary_claims]
        url_claims_resp = [AtomicClaimResponse(**claim.__dict__) for claim in url_claims]
        comparison_results = [
            ClaimComparisonResult(
                summary_claim_id=str(i),
                summary_claim_text=claim.text,
                relationship="missing",
                reasoning=f"Error during comparison: {str(e)}"
            )
            for i, claim in enumerate(summary_claims)
        ]
        return ClaimComparisonResponse(
            summary_claims=summary_claims_resp,
            url_claims=url_claims_resp,
            comparisons=comparison_results,
            statistics={"error": str(e)}
        )


@app.post("/api/v1/content-extractor/extract", response_model=ContentExtractionResponse)
async def extract_content(
    request: ExtractContentRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Extract academic content from URL, HTML, or text.

    This endpoint extracts the main title and body content from various sources,
    focusing on academic/informational content while excluding navigation,
    advertisements, and other non-content elements. Creates a new agent instance
    for each request (stateless operation).

    When compare_claims=True and summary is provided, performs rigorous claim-level comparison
    to evaluate whether URL content agrees or disagrees with summary claims. The comparison:
    1. Atomizes claims from both summary and URL content
    2. For each summary claim, searches URL claims to find agreement/disagreement
    3. Classifies relationships: agree (URL agrees with summary), disagree (URL disagrees), or missing (no relevant claim)
    4. Returns detailed comparison results with similarity scores, reasoning, and agreement statistics

    Args:
        request: Content extraction request with URL, HTML, or text
        api_key: API key for authentication

    Returns:
        ContentExtractionResponse with extracted title and main body, and optional claim comparison
    """
    try:
        # Create a new agent instance for each request (stateless)
        from ..agents.content_extractor.agent import ContentExtractorAgent
        agent = ContentExtractorAgent(
            model_name=settings.openai_model,
            api_key=settings.openai_api_key,
            api_base=settings.openai_api_base,
            temperature=settings.agent_temperature,
            proxy=settings.openai_proxy
        )

        if request.url:
            result = agent.extract_from_url(request.url, use_llm=request.use_llm, use_cot=request.use_cot, custom_few_shots=request.custom_few_shots)
        elif request.html:
            result = agent.extract_from_html(request.html, use_llm=request.use_llm, use_cot=request.use_cot, custom_few_shots=request.custom_few_shots)
        else:  # request.text
            result = agent.extract_from_text(request.text, title=request.title, use_llm=request.use_llm, use_cot=request.use_cot, custom_few_shots=request.custom_few_shots)

        # Perform claim comparison if requested
        claim_comparison = None
        if request.compare_claims and request.summary and result.main_body:
            logger.info("Performing claim-level comparison between summary and URL content")
            
            # Create claim atomizer agents
            from ..agents.claim_atomizer.agent import ClaimAtomizerAgent
            claim_atomizer = ClaimAtomizerAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                temperature=settings.agent_temperature,
                proxy=settings.openai_proxy
            )
            
            # Atomize summary claims
            summary_atomization = claim_atomizer.atomize_text(
                text=request.summary,
                use_cot=request.use_cot
            )
            
            # Atomize URL content claims
            url_atomization = claim_atomizer.atomize_text(
                text=result.main_body,
                use_cot=request.use_cot
            )
            
            # Compare claims
            claim_comparison = await compare_claims(
                summary_claims=summary_atomization.atomic_claims,
                url_claims=url_atomization.atomic_claims,
                use_cot=request.use_cot
            )
            
            logger.info(f"Claim comparison complete: {len(claim_comparison.comparisons)} comparisons")

        response_dict = result.__dict__
        response_dict["claim_comparison"] = claim_comparison
        return ContentExtractionResponse(**response_dict)

    except Exception as e:
        logger.error(f"Failed to extract content: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to extract content: {str(e)}")


# Claim Atomizer Endpoints
@app.post("/api/v1/claim-atomizer/atomize", response_model=ClaimAtomizationResponse)
async def atomize_claims(
    request: AtomizeClaimsRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Decompose text into atomic claims.

    This endpoint breaks down provided text into independent, verifiable atomic claims,
    each containing only one factual point. Creates a new agent instance for each
    request (stateless operation).

    Args:
        request: Claim atomization request with text and options
        api_key: API key for authentication

    Returns:
        ClaimAtomizationResponse with atomic claims
    """
    try:
        # Create a new agent instance for each request (stateless)
        from ..agents.claim_atomizer.agent import ClaimAtomizerAgent
        agent = ClaimAtomizerAgent(
            model_name=settings.openai_model,
            api_key=settings.openai_api_key,
            api_base=settings.openai_api_base,
            temperature=settings.agent_temperature,
            proxy=settings.openai_proxy
        )

        result = agent.atomize_text(
            text=request.text,
            use_cot=request.use_cot,
            custom_few_shots=request.custom_few_shots
        )

        # Convert to response format
        atomic_claims = [
            AtomicClaimResponse(**claim.__dict__) for claim in result.atomic_claims
        ]

        return ClaimAtomizationResponse(
            atomic_claims=atomic_claims,
            original_text=result.original_text,
            execution_mode=result.execution_mode,
            metadata=result.metadata
        )

    except Exception as e:
        logger.error(f"Failed to atomize claims: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to atomize claims: {str(e)}")


@app.get("/api/v1/content-extractor/default-shots")
async def get_content_extractor_default_shots(api_key: str = Depends(verify_api_key)):
    """
    Get default few-shot examples for Content Extractor agent.

    Args:
        api_key: API key for authentication

    Returns:
        Default few-shot examples
    """
    try:
        from ..agents.content_extractor.agent import ContentExtractorAgent
        few_shots = ContentExtractorAgent.get_default_few_shots()
        return {
            "agent_type": "content_extractor",
            "few_shots": few_shots
        }
    except Exception as e:
        logger.error(f"Failed to get default few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get default few shots: {str(e)}")


@app.get("/api/v1/claim-atomizer/default-shots")
async def get_claim_atomizer_default_shots(api_key: str = Depends(verify_api_key)):
    """
    Get default few-shot examples for Claim Atomizer agent.

    Args:
        api_key: API key for authentication

    Returns:
        Default few-shot examples
    """
    try:
        from ..agents.claim_atomizer.agent import ClaimAtomizerAgent
        few_shots = ClaimAtomizerAgent.get_default_few_shots()
        return {
            "agent_type": "claim_atomizer",
            "few_shots": few_shots
        }
    except Exception as e:
        logger.error(f"Failed to get default few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get default few shots: {str(e)}")


# Evidence Locator Endpoints
@app.post("/api/v1/evidence-locator/locate", response_model=EvidenceLocationResponse)
async def locate_evidence(
    request: LocateEvidenceRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Locate evidence for claims in main body text.

    This endpoint searches through the main body text to find supporting or
    contradicting evidence for each atomic claim. Creates a new agent instance
    for each request (stateless operation).

    Args:
        request: Evidence location request with claims and main body text
        api_key: API key for authentication

    Returns:
        EvidenceLocationResponse with evidence for each claim
    """
    try:
        # Create a new agent instance for each request (stateless)
        from ..agents.evidence_locator.agent import EvidenceLocatorAgent
        agent = EvidenceLocatorAgent(
            model_name=settings.openai_model,
            api_key=settings.openai_api_key,
            api_base=settings.openai_api_base,
            temperature=settings.agent_temperature,
            proxy=settings.openai_proxy
        )

        result = agent.locate_evidence(request.claims, request.main_body, use_llm=request.use_llm)

        # Convert to response format
        claim_evidences = []
        for ce in result.claim_evidences:
            quotes = [
                EvidenceQuoteResponse(**quote.__dict__) for quote in ce.quotes
            ]
            claim_evidences.append(ClaimEvidenceResponse(
                claim_id=ce.claim_id,
                claim_text=ce.claim_text,
                evidence_found=ce.evidence_found,
                quotes=quotes,
                reasoning=ce.reasoning
            ))

        return EvidenceLocationResponse(
            claim_evidences=claim_evidences,
            main_body_text=result.main_body_text,
            metadata=result.metadata
        )

    except Exception as e:
        logger.error(f"Failed to locate evidence: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to locate evidence: {str(e)}")


# Conflict Auditor Endpoints
@app.post("/api/v1/conflict-auditor/audit", response_model=ConflictAuditResponse)
async def audit_conflicts(
    request: AuditConflictsRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Audit conflicts between claims and evidence.

    This endpoint compares each atomic claim against its supporting evidence
    to determine logical consistency and identify conflicts. Creates a new
    agent instance for each request (stateless operation).

    Args:
        request: Conflict audit request with claim-evidence pairs
        api_key: API key for authentication

    Returns:
        ConflictAuditResponse with detailed conflict analyses
    """
    try:
        # Create a new agent instance for each request (stateless)
        from ..agents.conflict_auditor.agent import ConflictAuditorAgent
        agent = ConflictAuditorAgent(
            model_name=settings.openai_model,
            api_key=settings.openai_api_key,
            api_base=settings.openai_api_base,
            temperature=settings.agent_temperature,
            proxy=settings.openai_proxy
        )

        result = agent.audit_conflicts(
            claim_evidences=request.claim_evidences,
            use_cot=request.use_cot,
            custom_few_shots=request.custom_few_shots
        )

        # Convert to response format
        conflict_analyses = []
        for ca in result.conflict_analyses:
            conflict_analyses.append(ConflictAnalysisResponse(
                claim_id=ca.claim_id,
                claim_text=ca.claim_text,
                evidence_quotes=ca.evidence_quotes,
                verdict=ca.verdict.value,
                conflict_type=ca.conflict_type,
                analysis=ca.analysis,
                confidence=ca.confidence
            ))

        return ConflictAuditResponse(
            conflict_analyses=conflict_analyses,
            summary_stats=result.summary_stats,
            execution_mode=result.execution_mode,
            metadata=result.metadata
        )

    except Exception as e:
        logger.error(f"Failed to audit conflicts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to audit conflicts: {str(e)}")


@app.get("/api/v1/conflict-auditor/default-shots")
async def get_conflict_auditor_default_shots(api_key: str = Depends(verify_api_key)):
    """
    Get default few-shot examples for Conflict Auditor agent.

    Args:
        api_key: API key for authentication

    Returns:
        Default few-shot examples
    """
    try:
        from ..agents.conflict_auditor.agent import ConflictAuditorAgent
        few_shots = ConflictAuditorAgent.get_default_few_shots()
        return {
            "agent_type": "conflict_auditor",
            "few_shots": few_shots
        }
    except Exception as e:
        logger.error(f"Failed to get default few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get default few shots: {str(e)}")


@app.get("/api/v1/evidence-locator/default-shots")
async def get_evidence_locator_default_shots(api_key: str = Depends(verify_api_key)):
    """
    Get default few-shot examples for Evidence Locator agent.

    Args:
        api_key: API key for authentication

    Returns:
        Default few-shot examples
    """
    try:
        # Evidence Locator doesn't have few-shot examples currently
        return {
            "agent_type": "evidence_locator",
            "few_shots": "Evidence Locator uses string matching validation rather than few-shot examples for maximum accuracy."
        }
    except Exception as e:
        logger.error(f"Failed to get default few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get default few shots: {str(e)}")


@app.get("/api/v1/synthesis-aggregator/default-shots")
async def get_synthesis_aggregator_default_shots(api_key: str = Depends(verify_api_key)):
    """
    Get default few-shot examples for Synthesis Aggregator agent.

    Args:
        api_key: API key for authentication

    Returns:
        Default few-shot examples
    """
    try:
        # Synthesis Aggregator doesn't have few-shot examples currently
        return {
            "agent_type": "synthesis_aggregator",
            "few_shots": "Synthesis Aggregator uses statistical analysis rather than few-shot examples."
        }
    except Exception as e:
        logger.error(f"Failed to get default few shots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get default few shots: {str(e)}")


# Synthesis Aggregator Endpoints
@app.post("/api/v1/synthesis-aggregator/aggregate", response_model=SynthesisAggregationResponse)
async def aggregate_synthesis(
    request: AggregateSynthesisRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Aggregate conflict analyses into comprehensive synthesis report.

    This endpoint creates a comprehensive report from conflict analysis results,
    including quantitative metrics and quality assessment. Creates a new agent
    instance for each request (stateless operation).

    Args:
        request: Synthesis aggregation request with conflict analyses
        api_key: API key for authentication

    Returns:
        SynthesisAggregationResponse with comprehensive report
    """
    try:
        # Create a new agent instance for each request (stateless)
        from ..agents.synthesis_aggregator.agent import SynthesisAggregatorAgent
        agent = SynthesisAggregatorAgent(
            model_name=settings.openai_model,
            api_key=settings.openai_api_key,
            api_base=settings.openai_api_base,
            temperature=settings.agent_temperature,
            proxy=settings.openai_proxy
        )

        result = agent.aggregate_synthesis(
            conflict_analyses=request.conflict_analyses,
            use_llm_enhancement=request.use_llm_enhancement
        )

        # Convert to response format
        synthesis_report = SynthesisReportResponse(**result.synthesis_report.__dict__)

        return SynthesisAggregationResponse(
            synthesis_report=synthesis_report,
            metadata=result.metadata
        )

    except Exception as e:
        logger.error(f"Failed to aggregate synthesis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to aggregate synthesis: {str(e)}")


# ===========================
# Overall Evaluation Interface
# ===========================

class OverallEvaluationRequest(BaseModel):
    """Request model for overall evaluation of summary and URL."""
    summary: Optional[str] = Field(default=None, description="Text summary to evaluate")
    url: Optional[str] = Field(default=None, description="URL to evaluate")
    include_full_analysis: bool = Field(default=False, description="Whether to include full academic analysis pipeline")

    @model_validator(mode='after')
    def validate_input(self):
        """Ensure at least one input is provided."""
        if not self.summary and not self.url:
            raise ValueError("At least one of 'summary' or 'url' must be provided")
        return self


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
    full_academic_analysis: Optional[Dict[str, Any]] = None
    processing_metadata: Dict[str, Any]


# ===========================
# Overall Evaluation Interface
# ===========================

class OverallEvaluationRequest(BaseModel):
    """Request model for overall evaluation of summary and URL."""
    summary: Optional[str] = Field(default=None, description="Text summary to evaluate")
    url: Optional[str] = Field(default=None, description="URL to evaluate")
    include_full_analysis: bool = Field(default=False, description="Whether to include full academic analysis pipeline")

    @model_validator(mode='after')
    def validate_input(self):
        """Ensure at least one input is provided."""
        if not self.summary and not self.url:
            raise ValueError("At least one of 'summary' or 'url' must be provided")
        return self


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
    full_academic_analysis: Optional[Dict[str, Any]] = None
    processing_metadata: Dict[str, Any]


@app.post("/api/v1/evaluation/overall", response_model=OverallEvaluationResponse)
async def overall_evaluation(
    request: OverallEvaluationRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Overall evaluation interface for summary and URL analysis with statistical results.

    This endpoint provides comprehensive evaluation of content including:
    - Content quality metrics
    - Academic integrity assessment
    - Bias analysis
    - Comparative analysis between summary and source
    - Optional full academic analysis pipeline

    Args:
        request: OverallEvaluationRequest with summary, URL, and options
        api_key: API key for authentication

    Returns:
        OverallEvaluationResponse with detailed evaluation and statistics
    """
    import time
    start_time = time.time()

    logger.info(f"Starting overall evaluation: summary={bool(request.summary)}, url={bool(request.url)}, full_analysis={request.include_full_analysis}")

    evaluation_result = {
        "summary_evaluation": None,
        "url_evaluation": None,
        "comparative_analysis": None,
        "full_academic_analysis": None,
        "processing_metadata": {}
    }

    try:
        # Evaluate summary if provided
        if request.summary:
            summary_eval = await _evaluate_content(request.summary, "summary")
            evaluation_result["summary_evaluation"] = summary_eval

        # Evaluate URL if provided
        if request.url:
            url_eval = await _evaluate_url(request.url)
            evaluation_result["url_evaluation"] = url_eval

        # Comparative analysis if both are provided
        if request.summary and request.url:
            comparative = await _comparative_analysis(request.summary, request.url)
            evaluation_result["comparative_analysis"] = comparative

        # Full academic analysis if requested
        if request.include_full_analysis and request.url:
            # Run the complete academic analysis pipeline
            analysis_request = CompleteAnalysisRequest(
                url=request.url,
                use_llm_content_extraction=False,
                use_cot_atomization=False,
                use_cot_audit=False,
                use_llm_synthesis=True
            )
            analysis_result = await complete_academic_analysis(analysis_request, api_key)
            evaluation_result["full_academic_analysis"] = {
                "overall_success": analysis_result.overall_success,
                "final_report": analysis_result.final_report.__dict__ if analysis_result.final_report else None,
                "pipeline_steps": [step.__dict__ for step in analysis_result.pipeline_steps],
                "total_execution_time": analysis_result.total_execution_time
            }

    except Exception as e:
        logger.error(f"Overall evaluation failed: {e}", exc_info=True)
        evaluation_result["processing_metadata"]["error"] = str(e)

    # Add processing metadata
    evaluation_result["processing_metadata"].update({
        "total_processing_time": time.time() - start_time,
        "evaluation_timestamp": time.time(),
        "inputs_provided": {
            "summary": bool(request.summary),
            "url": bool(request.url),
            "full_analysis": request.include_full_analysis
        }
    })

    logger.info(".2f")

    return OverallEvaluationResponse(**evaluation_result)


async def _evaluate_content(content: str, content_type: str) -> Dict[str, Any]:
    """Evaluate content quality and provide metrics."""
    try:
        # Basic content metrics
        content_length = len(content)
        sentences = content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0

        # Simple readability score (approximate)
        readability_score = max(0, min(100, 206.835 - 1.015 * avg_sentence_length - 84.6 * (content.count(' ') / content_length)))

        return {
            "content_type": content_type,
            "metrics": EvaluationMetrics(
                content_length=content_length,
                readability_score=readability_score,
                fact_density=None,  # Would require LLM analysis
                opinion_density=None,  # Would require LLM analysis
                bias_distribution=None,  # Would require LLM analysis
                academic_integrity_score=None,  # Would require LLM analysis
                hallucination_risk=None  # Would require LLM analysis
            ).__dict__,
            "basic_stats": {
                "sentence_count": len(sentences),
                "word_count": len(content.split()),
                "avg_sentence_length": avg_sentence_length
            }
        }
    except Exception as e:
        logger.error(f"Content evaluation failed: {e}")
        return {"error": str(e)}


async def _evaluate_url(url: str) -> Dict[str, Any]:
    """Evaluate URL content quality."""
    try:
        # Extract content from URL
        content_agent = ContentExtractorAgent(
            model_name=settings.openai_model,
            api_key=settings.openai_api_key,
            api_base=settings.openai_api_base,
            temperature=settings.agent_temperature,
            proxy=settings.openai_proxy
        )
        content_result = content_agent.extract_from_url(url, use_llm=False)

        if content_result.main_body:
            content_eval = await _evaluate_content(content_result.main_body, "url_content")
            content_eval.update({
                "url": url,
                "title": content_result.title,
                "extraction_metadata": content_result.extraction_metadata
            })
            return content_eval
        else:
            return {"error": "Failed to extract content from URL"}

    except Exception as e:
        logger.error(f"URL evaluation failed: {e}")
        return {"error": str(e)}


async def _comparative_analysis(summary: str, url: str) -> Dict[str, Any]:
    """Compare summary against source URL."""
    try:
        # Get both evaluations
        summary_eval = await _evaluate_content(summary, "summary")
        url_eval = await _evaluate_url(url)

        # Simple comparative metrics
        compression_ratio = len(summary) / len(url_eval.get("metrics", {}).get("content_length", 1))

        return {
            "compression_ratio": compression_ratio,
            "summary_density": summary_eval.get("basic_stats", {}).get("word_count", 0) / summary_eval.get("basic_stats", {}).get("sentence_count", 1),
            "source_density": url_eval.get("basic_stats", {}).get("word_count", 0) / url_eval.get("basic_stats", {}).get("sentence_count", 1),
            "consistency_check": {
                "summary_length": len(summary),
                "source_length": url_eval.get("metrics", {}).get("content_length", 0),
                "title_match": summary_eval.get("title") == url_eval.get("title") if summary_eval.get("title") and url_eval.get("title") else None
            }
        }
    except Exception as e:
        logger.error(f"Comparative analysis failed: {e}")
        return {"error": str(e)}


# ===========================
# Complete Academic Analysis Pipeline Endpoint
# ===========================

class CompleteAnalysisRequest(BaseModel):
    """Request model for complete academic analysis pipeline."""
    url: str = Field(..., description="URL to analyze completely")
    use_llm_content_extraction: bool = Field(default=False, description="Use LLM for content extraction refinement")
    use_cot_atomization: bool = Field(default=False, description="Use CoT for claim atomization")
    use_cot_audit: bool = Field(default=False, description="Use CoT for conflict auditing")
    use_llm_synthesis: bool = Field(default=True, description="Use LLM enhancement for synthesis")
    custom_few_shots_atomizer: Optional[str] = Field(default=None, description="Custom few-shots for atomizer")
    custom_few_shots_auditor: Optional[str] = Field(default=None, description="Custom few-shots for auditor")


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
    final_report: Optional[SynthesisReportResponse] = None
    overall_success: bool
    total_execution_time: Optional[float] = None
    error: Optional[str] = None


@app.post("/api/v1/analysis/complete", response_model=CompleteAnalysisResponse)
async def complete_academic_analysis(
    request: CompleteAnalysisRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Complete academic content analysis pipeline.

    This endpoint runs the full 5-step academic analysis pipeline:
    1. Content Extraction - Extract title and main body
    2. Claim Atomization - Break down into atomic claims
    3. Evidence Location - Find supporting evidence
    4. Conflict Audit - Check logical consistency
    5. Synthesis Aggregation - Create comprehensive report

    Args:
        request: Complete analysis request with URL and options
        api_key: API key for authentication

    Returns:
        CompleteAnalysisResponse with full pipeline results
    """
    import time
    start_time = time.time()

    logger.info(f"Starting complete academic analysis pipeline for URL: {request.url}")

    pipeline_steps = []
    overall_success = True
    final_report = None

    try:
        # Step 1: Content Extraction
        step_start = time.time()
        try:
            content_agent = ContentExtractorAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                temperature=settings.agent_temperature,
                proxy=settings.openai_proxy
            )
            content_result = content_agent.extract_from_url(request.url, use_llm=request.use_llm_content_extraction)
            step_time = time.time() - step_start

            pipeline_steps.append(PipelineStepResponse(
                step_name="content_extraction",
                success=True,
                data={
                    "title": content_result.title,
                    "main_body_length": content_result.text_length,
                    "truncated": content_result.truncated
                },
                execution_time=step_time
            ))
            logger.info(".2f")
        except Exception as e:
            step_time = time.time() - step_start
            pipeline_steps.append(PipelineStepResponse(
                step_name="content_extraction",
                success=False,
                error=str(e),
                execution_time=step_time
            ))
            overall_success = False
            raise

        # Step 2: Claim Atomization
        step_start = time.time()
        try:
            if not content_result.main_body:
                raise ValueError("No main body content extracted")

            atomizer_agent = ClaimAtomizerAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                temperature=settings.agent_temperature,
                proxy=settings.openai_proxy,
                execution_mode=settings.default_execution_mode
            )
            atomization_result = atomizer_agent.atomize_text(
                text=content_result.main_body,
                use_cot=request.use_cot_atomization,
                custom_few_shots=request.custom_few_shots_atomizer
            )
            step_time = time.time() - step_start

            pipeline_steps.append(PipelineStepResponse(
                step_name="claim_atomization",
                success=True,
                data={
                    "atomic_claims_count": len(atomization_result.atomic_claims),
                    "execution_mode": atomization_result.execution_mode
                },
                execution_time=step_time
            ))
            logger.info(".2f")
        except Exception as e:
            step_time = time.time() - step_start
            pipeline_steps.append(PipelineStepResponse(
                step_name="claim_atomization",
                success=False,
                error=str(e),
                execution_time=step_time
            ))
            overall_success = False
            raise

        # Step 3: Evidence Location
        step_start = time.time()
        try:
            # Prepare claims for evidence location
            claims = [{"id": claim.id, "text": claim.text} for claim in atomization_result.atomic_claims]

            locator_agent = EvidenceLocatorAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                temperature=settings.agent_temperature,
                proxy=settings.openai_proxy
            )
            evidence_result = locator_agent.locate_evidence(claims, content_result.main_body, use_llm=True)
            step_time = time.time() - step_start

            pipeline_steps.append(PipelineStepResponse(
                step_name="evidence_location",
                success=True,
                data={
                    "claims_with_evidence": sum(1 for ce in evidence_result.claim_evidences if ce.evidence_found),
                    "total_claims": len(evidence_result.claim_evidences)
                },
                execution_time=step_time
            ))
            logger.info(".2f")
        except Exception as e:
            step_time = time.time() - step_start
            pipeline_steps.append(PipelineStepResponse(
                step_name="evidence_location",
                success=False,
                error=str(e),
                execution_time=step_time
            ))
            overall_success = False
            raise

        # Step 4: Conflict Audit
        step_start = time.time()
        try:
            # Prepare claim-evidence pairs for audit
            claim_evidences = []
            for ce in evidence_result.claim_evidences:
                claim_evidences.append({
                    "claim_id": ce.claim_id,
                    "claim_text": ce.claim_text,
                    "evidence_quotes": [quote.text for quote in ce.quotes]
                })

            auditor_agent = ConflictAuditorAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                temperature=settings.agent_temperature,
                proxy=settings.openai_proxy,
                execution_mode=settings.default_execution_mode
            )
            audit_result = auditor_agent.audit_conflicts(
                claim_evidences=claim_evidences,
                use_cot=request.use_cot_audit,
                custom_few_shots=request.custom_few_shots_auditor
            )
            step_time = time.time() - step_start

            pipeline_steps.append(PipelineStepResponse(
                step_name="conflict_audit",
                success=True,
                data={
                    "supported_claims": audit_result.summary_stats.get("supported", 0),
                    "contradicted_claims": audit_result.summary_stats.get("contradicted", 0),
                    "execution_mode": audit_result.execution_mode
                },
                execution_time=step_time
            ))
            logger.info(".2f")
        except Exception as e:
            step_time = time.time() - step_start
            pipeline_steps.append(PipelineStepResponse(
                step_name="conflict_audit",
                success=False,
                error=str(e),
                execution_time=step_time
            ))
            overall_success = False
            raise

        # Step 5: Synthesis Aggregation
        step_start = time.time()
        try:
            # Prepare conflict analyses for synthesis
            conflict_analyses = []
            for ca in audit_result.conflict_analyses:
                conflict_analyses.append({
                    "claim_id": ca.claim_id,
                    "claim_text": ca.claim_text,
                    "evidence_quotes": ca.evidence_quotes,
                    "verdict": ca.verdict.value,
                    "conflict_type": ca.conflict_type,
                    "analysis": ca.analysis,
                    "confidence": ca.confidence
                })

            aggregator_agent = SynthesisAggregatorAgent(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                temperature=settings.agent_temperature,
                proxy=settings.openai_proxy
            )
            synthesis_result = aggregator_agent.aggregate_synthesis(
                conflict_analyses=conflict_analyses,
                use_llm_enhancement=request.use_llm_synthesis
            )
            step_time = time.time() - step_start

            final_report = SynthesisReportResponse(**synthesis_result.synthesis_report.__dict__)

            pipeline_steps.append(PipelineStepResponse(
                step_name="synthesis_aggregation",
                success=True,
                data={
                    "confidence_score": synthesis_result.synthesis_report.confidence_score,
                    "quality_assessment": synthesis_result.synthesis_report.quality_assessment[:50] + "..."
                },
                execution_time=step_time
            ))
            logger.info(".2f")
        except Exception as e:
            step_time = time.time() - step_start
            pipeline_steps.append(PipelineStepResponse(
                step_name="synthesis_aggregation",
                success=False,
                error=str(e),
                execution_time=step_time
            ))
            overall_success = False
            raise

    except Exception as pipeline_error:
        logger.error(f"Pipeline failed: {pipeline_error}", exc_info=True)
        error_msg = str(pipeline_error)

    total_time = time.time() - start_time
    logger.info(".2f")

    return CompleteAnalysisResponse(
        url=request.url,
        pipeline_steps=pipeline_steps,
        final_report=final_report,
        overall_success=overall_success,
        total_execution_time=total_time,
        error=error_msg if not overall_success else None
    )
