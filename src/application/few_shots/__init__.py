"""Prompt templates and few-shot examples for all agents."""

from typing import Dict, List, Any
from ...shared.utils.logger import get_logger

logger = get_logger(__name__)


def reset_all_agent_few_shots() -> Dict[str, Any]:
    """
    Reset all agent custom few shots to their default values.
    
    This function iterates through all agents that support custom few shots
    and resets them by calling their set_custom_few_shots(None) method.
    
    Returns:
        Dict with:
        - success: bool - Whether all agents were reset successfully
        - errors: List[str] - List of error messages (if any)
    """
    result = {
        "success": True,
        "errors": []
    }
    
    def add_error(msg: str):
        logger.error(msg, exc_info=True)
        result["errors"].append(msg)
        result["success"] = False
    
    try:
        from ...application.agents.nudge_collapse.agent import NudgeCollapseAgent
        from ...application.agents.summarizer.agent import SummarizerAgent
        from ...application.agents.bot_creator.agent import BotCreatorAgent
        from ...application.agents.demographic_evaluator.agent import DemographicEvaluatorAgent
        from ...application.agents.content_extractor.agent import ContentExtractorAgent
        from ...application.agents.claim_atomizer.agent import ClaimAtomizerAgent
        from ...application.agents.conflict_auditor.agent import ConflictAuditorAgent
        from ...application.agents.web_opinion_extractor.agent import WebOpinionAnalyzer
        from ...application.agents.privacy_detector.agent import PrivacyDetectorAgent
        
        agents_to_reset = [
            ("NudgeCollapseAgent", NudgeCollapseAgent),
            ("SummarizerAgent", SummarizerAgent),
            ("BotCreatorAgent", BotCreatorAgent),
            ("DemographicEvaluatorAgent", DemographicEvaluatorAgent),
            ("ContentExtractorAgent", ContentExtractorAgent),
            ("ClaimAtomizerAgent", ClaimAtomizerAgent),
            ("ConflictAuditorAgent", ConflictAuditorAgent),
            ("WebOpinionAnalyzer", WebOpinionAnalyzer),
            ("PrivacyDetectorAgent", PrivacyDetectorAgent),
        ]
        
        for agent_name, agent_class in agents_to_reset:
            try:
                if hasattr(agent_class, 'set_custom_few_shots'):
                    agent_class.set_custom_few_shots(None)
                    logger.debug(f"{agent_name} custom few shots reset")
            except Exception as e:
                logger.warning(f"Failed to reset {agent_name} custom shots: {e}")
                add_error(f"Failed to reset {agent_name} custom few shots: {e}")
        
        logger.info("All agent custom few shots reset")
    except Exception as e:
        add_error(f"Failed to reset agent custom few shots: {e}")
    
    return result
