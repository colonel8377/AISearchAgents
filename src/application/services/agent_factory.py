"""Agent Factory for creating agent instances - Single Responsibility: Agent Creation."""

from typing import Optional, Any
from langchain_openai import OpenAIEmbeddings

from ...shared.constant.enums import AgentType
from ...application.agents.bot_creator.agent import BotCreatorAgent
from ...application.agents.claim_atomizer.agent import ClaimAtomizerAgent
from ...application.agents.conflict_auditor.agent import ConflictAuditorAgent
from ...application.agents.content_extractor.agent import ContentExtractorAgent
from ...application.agents.demographic_evaluator.agent import DemographicEvaluatorAgent
from ...application.agents.nudge_collapse.agent import NudgeCollapseAgent
from ...application.agents.summarizer.agent import SummarizerAgent
from ...shared.config.settings import settings
from ...infrastructure.factory import VectorStoreFacade
from ...shared.utils.logger import get_logger

logger = get_logger(__name__)


class AgentFactory:
    """
    Factory class for creating agent instances.
    
    Single Responsibility: Handle agent instantiation logic.
    Follows Factory Pattern for creating different agent types.
    """
    
    @staticmethod
    def create_vector_store(agent_id: str, use_memory: bool):
        """
        Create a vector store for agent memory if needed.
        
        Args:
            agent_id: The agent ID
            use_memory: Whether to use memory
            
        Returns:
            Vector store instance or None
        """
        if not use_memory:
            return None
        
        logger.debug(f"Creating vector store: type={settings.vector_store_type}")
        embeddings = OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base
        )
        
        if settings.vector_store_type == "redis":
            if settings.redis_password:
                redis_url = f"redis://:{settings.redis_password}@{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"
            else:
                redis_url = f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"
            
            vector_store_kwargs = {
                "redis_url": redis_url,
                "index_name": f"agent_memory_{agent_id or 'auto'}"
            }
            logger.debug(f"Redis vector store config: host={settings.redis_host}, port={settings.redis_port}")
        elif settings.vector_store_type == "postgres":
            vector_store_kwargs = {
                "connection_string": f"postgresql://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}",
                "collection_name": f"agent_memory_{agent_id or 'auto'}"
            }
            logger.debug(f"Postgres vector store config: host={settings.postgres_host}, port={settings.postgres_port}")
        else:  # chroma
            vector_store_kwargs = {
                "persist_directory": settings.chroma_persist_directory,
                "collection_name": f"agent_memory_{agent_id or 'auto'}"
            }
            logger.debug(f"Chroma vector store config: persist_directory={settings.chroma_persist_directory}")
        
        vector_store = VectorStoreFacade.create_vector_store(
            store_type=settings.vector_store_type,
            embeddings=embeddings,
            **vector_store_kwargs
        )
        logger.info(f"Vector store created successfully")
        
        return vector_store
    
    @staticmethod
    def create_agent(
        agent_type: AgentType,
        agent_id: Optional[str] = None,
        use_memory: bool = False,
        persona_mode: Optional[str] = None
    ) -> Any:
        """
        Create an agent instance of the specified type.
        
        Args:
            agent_type: Type of agent to create
            agent_id: Optional agent ID for vector store naming
            use_memory: Whether to enable memory for the agent
            persona_mode: Optional persona mode for BotCreatorAgent
            
        Returns:
            Agent instance
            
        Raises:
            ValueError: If agent type is unsupported or invalid parameters provided
        """
        logger.debug(f"Creating agent: type={agent_type.value}, model={settings.openai_model}")
        
        # Create vector store if needed
        vector_store = AgentFactory.create_vector_store(agent_id or "auto", use_memory)
        
        # Common agent parameters
        common_params = {
            "model_name": settings.openai_model,
            "api_key": settings.openai_api_key,
            "api_base": settings.openai_api_base,
            "temperature": settings.agent_temperature,
        }
        
        # Create agent based on type
        if agent_type == AgentType.NUDGE_COLLAPSE:
            return NudgeCollapseAgent(
                vector_store=vector_store,
                **common_params
            )
        elif agent_type == AgentType.SUMMARIZER:
            return SummarizerAgent(
                vector_store=vector_store,
                **common_params
            )
        elif agent_type == AgentType.BOT_CREATOR:
            if persona_mode and persona_mode not in ["system_prompt", "user_instruction"]:
                raise ValueError(f"Invalid persona_mode. Must be 'system_prompt' or 'user_instruction'")
            
            return BotCreatorAgent(
                vector_store=vector_store,
                persona_mode=persona_mode or "system_prompt",
                **common_params
            )
        elif agent_type == AgentType.DEMOGRAPHIC_EVALUATOR:
            return DemographicEvaluatorAgent(**common_params)
        elif agent_type == AgentType.CONTENT_EXTRACTOR:
            return ContentExtractorAgent(**common_params)
        elif agent_type == AgentType.CLAIM_ATOMIZER:
            return ClaimAtomizerAgent(**common_params)
        elif agent_type == AgentType.CONFLICT_AUDITOR:
            return ConflictAuditorAgent(**common_params)
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")

