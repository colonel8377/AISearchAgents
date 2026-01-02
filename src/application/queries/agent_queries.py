"""
Agent Query classes and handlers for CQRS pattern.

Queries represent read operations that retrieve data without modifying state.
Query handlers process these queries and can be optimized independently of commands.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from ...shared.constant.enums import AgentType


# Query Classes
@dataclass
class GetAgentQuery:
    """Query to get a specific agent."""
    agent_id: str


@dataclass
class ListAgentsQuery:
    """Query to list agents with optional filtering."""
    filter_type: Optional[AgentType] = None
    limit: Optional[int] = None


@dataclass
class GetAgentStatusQuery:
    """Query to get agent status."""
    agent_id: str


# Query Results
@dataclass
class AgentInfo:
    """Agent information for queries."""
    agent_id: str
    agent_type: AgentType
    status: str
    message: Optional[str] = None
    persona_mode: Optional[str] = None
    current_turn: Optional[int] = None
    additional_info: Optional[Dict[str, Any]] = None


@dataclass
class AgentListResult:
    """Result of listing agents."""
    agents: List[Dict[str, Any]]
    total_count: int


@dataclass
class AgentQueryResult:
    """Result of agent query execution."""
    success: bool
    data: Optional[Any] = None
    error_message: Optional[str] = None


class AgentQueryHandler:
    """
    Handles agent read operations (Queries) in CQRS pattern.

    Queries are read operations that retrieve data without modifying state.
    They can be optimized independently from commands (caching, read replicas, etc.).
    """

    def __init__(self, business_logic):
        """
        Initialize query handler.

        Args:
            business_logic: Agent business logic instance
        """
        self.business_logic = business_logic

    async def handle_get_agent(self, query: GetAgentQuery) -> AgentQueryResult:
        """
        Handle get agent query.

        Args:
            query: Get agent query

        Returns:
            Query result with agent info or error
        """
        try:
            # For get_agent, we might want to return basic info
            # Since we don't have a direct get_agent_logic method, we'll use status
            result = self.business_logic.get_agent_status_logic(query.agent_id)

            agent_info = AgentInfo(
                agent_id=result["agent_id"],
                agent_type=result["agent_type"],
                status=result["status"],
                current_turn=result.get("current_turn"),
                additional_info=result.get("additional_info")
            )

            return AgentQueryResult(success=True, data=agent_info)

        except Exception as e:
            return AgentQueryResult(
                success=False,
                error_message=str(e)
            )

    async def handle_list_agents(self, query: ListAgentsQuery) -> AgentQueryResult:
        """
        Handle list agents query.

        Args:
            query: List agents query

        Returns:
            Query result with agent list
        """
        try:
            result = self.business_logic.list_agents_logic()

            # Apply filters if specified
            agents = result["agents"]
            if query.filter_type:
                agents = [a for a in agents if a.get("agent_type") == query.filter_type.value]

            # Apply limit if specified
            if query.limit:
                agents = agents[:query.limit]

            list_result = AgentListResult(
                agents=agents,
                total_count=len(agents)
            )

            return AgentQueryResult(success=True, data=list_result)

        except Exception as e:
            return AgentQueryResult(
                success=False,
                error_message=str(e)
            )

    async def handle_get_agent_status(self, query: GetAgentStatusQuery) -> AgentQueryResult:
        """
        Handle get agent status query.

        Args:
            query: Get agent status query

        Returns:
            Query result with agent status info
        """
        try:
            result = self.business_logic.get_agent_status_logic(query.agent_id)

            agent_info = AgentInfo(
                agent_id=result["agent_id"],
                agent_type=result["agent_type"],
                status=result["status"],
                current_turn=result.get("current_turn"),
                additional_info=result.get("additional_info")
            )

            return AgentQueryResult(success=True, data=agent_info)

        except Exception as e:
            return AgentQueryResult(
                success=False,
                error_message=str(e)
            )