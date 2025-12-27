"""Debate API routes."""

from fastapi import APIRouter, Depends, HTTPException

from ..common import get_api_key
from ..schemas import (
    GeneratePersonasRequest, GeneratePersonasResponse, InitDebateResponse,
    DebateStatisticsResponse, DebateContinueResponse, BiasDistributionResponse, AtomicOpinionResponse,
    RoundReasoningResponse, RoundVotingResponse
)
from ...agents.web_opinion_extractor import AtomicOpinion, BiasDistribution
from ...config.settings import settings
from ...debate.schemas import InitRequest
from ...debate.service import DebateService, generate_default_personas
from ...utils.logger import get_logger

debate_service = DebateService()

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/debate", tags=["Debate"])

@router.post("/generate-personas", response_model=GeneratePersonasResponse)
async def generate_personas(
    request: GeneratePersonasRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)

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

        _api_key: Authentication dependency (value not used, required for auth check)

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

        personas, detected_style = await debate_service.generate_personas_for_topic(

            topic=request.topic,

            context=request.context,

            num_agents=request.num_agents,

            corpus=request.corpus

        )

        return GeneratePersonasResponse(

            personas=personas,

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

@router.post("/init", response_model=InitDebateResponse)
async def init_debate(
    request: InitRequest,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)

):

    """

    Initialize a new debate session.

    Creates agents based on custom personas or auto-generates them using

    LLM-based persona generation with the specified execution mode.

    Each agent gets a specialized system prompt and few-shot example.

    Args:

        request: InitRequest with topic, persona configuration, and options

        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:

        Session ID, list of agent metadata, and debate configuration

    """

    try:

        # Determine personas to use

        if request.custom_personas:

            personas = request.custom_personas

        elif request.auto_agent_count > 0:

            personas, detected_style = await debate_service.generate_personas_for_topic(

                topic=request.topic,

                context=request.context or "",

                num_agents=request.auto_agent_count,

                    corpus=request.corpus

                )

        else:

            raise HTTPException(

                status_code=400,

                detail="Must provide either custom_personas or auto_agent_count > 0"

            )

        # Create session with max_rounds

        session_id, agents = debate_service.create_session(

            request.topic,

            personas,

            max_rounds=request.max_rounds,

            corpus=request.corpus,

            detected_user_style=detected_style if 'detected_style' in locals() else None

        )

        # Get max_rounds from session

        session = debate_service.get_session(session_id)

        max_rounds = session.get("max_rounds", 10) if session else 10

        return InitDebateResponse(

            session_id=session_id,

            agents=agents,

            topic=request.topic,

            max_rounds=max_rounds

        )

    except HTTPException:

        raise

    except Exception as e:

        logger.error(f"Failed to initialize debate: {e}", exc_info=True)

        raise HTTPException(status_code=500, detail=f"Failed to initialize debate: {str(e)}")

@router.get("/{session_id}/statistics", response_model=DebateStatisticsResponse)
async def get_debate_statistics(
    session_id: str,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)

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

        _api_key: Authentication dependency (value not used, required for auth check)

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

@router.get("/{session_id}/should-continue", response_model=DebateContinueResponse)
async def check_debate_continue(
    session_id: str,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)

):

    """

    Check if the debate should continue.

    Returns whether to continue the debate and the reason why.

    This helps manage non-converging debates by enforcing max rounds.

    Args:

        session_id: Session identifier

        _api_key: Authentication dependency (value not used, required for auth check)

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


@router.post("/{session_id}/round-reasoning", response_model=RoundReasoningResponse)
async def execute_round_reasoning(
    session_id: str,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)

):

    """

    Execute a reasoning round where all agents provide statements and reasoning.

    All agents reason independently based on previous debate history but cannot

    see the current round's statements from other agents.

    Args:

        session_id: Session identifier

        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:

        RoundReasoningResponse with all agents' reasoning for this round

    """

    try:

        # Verify session exists

        session = debate_service.get_session(session_id)

        if not session:

            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

        # Check if we should continue debate

        continue_info = debate_service.should_continue_debate(session_id)

        if not continue_info.get("continue", False):

            raise HTTPException(

                status_code=400,

                detail=f"Debate cannot continue: {continue_info.get('reason', 'Unknown reason')}"

            )

        # Execute reasoning round

        result = await debate_service.execute_round_reasoning(session_id)

        if "error" in result:

            raise HTTPException(status_code=500, detail=result["error"])

        return RoundReasoningResponse(**result)

    except HTTPException:

        raise

    except Exception as e:

        logger.error(f"Failed to execute reasoning round for session {session_id}: {e}", exc_info=True)

        raise HTTPException(status_code=500, detail=f"Failed to execute reasoning round: {str(e)}")


@router.post("/{session_id}/round-voting", response_model=RoundVotingResponse)
async def execute_round_voting(
    session_id: str,
    _api_key: str = Depends(get_api_key)  # Authentication dependency (value not used)

):

    """

    Execute a voting round where all agents vote based on current round's reasoning.

    All agents review the current round's statements and reasoning from all other

    agents and cast their votes.

    Args:

        session_id: Session identifier

        _api_key: Authentication dependency (value not used, required for auth check)

    Returns:

        RoundVotingResponse with all agents' votes for this round

    """

    try:

        # Verify session exists

        session = debate_service.get_session(session_id)

        if not session:

            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

        # Check if reasoning phase was completed

        if session.get("current_phase") != "reasoning":

            raise HTTPException(

                status_code=400,

                detail="Voting round requires completed reasoning round first"

            )

        # Execute voting round

        result = await debate_service.execute_round_voting(session_id)

        if "error" in result:

            raise HTTPException(status_code=500, detail=result["error"])

        return RoundVotingResponse(**result)

    except HTTPException:

        raise

    except Exception as e:

        logger.error(f"Failed to execute voting round for session {session_id}: {e}", exc_info=True)

        raise HTTPException(status_code=500, detail=f"Failed to execute voting round: {str(e)}")
