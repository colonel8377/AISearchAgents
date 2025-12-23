"""Synthesis Aggregator Agent for summarizing conflict analysis results.

This agent aggregates all conflict analyses into a comprehensive report with
quantitative metrics and quality assessment.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from ...config.settings import settings
from ...utils.llm_client import llm_manager
from ...utils.logger import get_logger
from ...utils.smart_memory import SmartMemory

logger = get_logger(__name__)


@dataclass
class SynthesisReport:
    """Comprehensive synthesis report."""
    research_summary: str
    metrics: Dict[str, Any]
    detailed_discrepancies: List[Dict[str, Any]]
    quality_assessment: str
    confidence_score: float


@dataclass
class SynthesisAggregationResult:
    """Result of synthesis aggregation."""
    synthesis_report: SynthesisReport
    metadata: Optional[Dict[str, Any]] = None


class SynthesisAggregatorAgent:
    """
    Synthesis Aggregator Agent that creates comprehensive reports from
    conflict analysis results.

    Features:
    - Quantitative conflict metrics
    - Quality assessment and anomaly detection
    - Comprehensive discrepancy mapping
    - Confidence scoring
    """

    SYSTEM_PROMPT = """You are a Statistical Research Auditor.

Task: Synthesize the findings from the previous steps into a structured, quantitative conflict report.

Instructions:
1. Conflict Counting: Quantify the total number of claims, the number of confirmed conflicts, and the hallucination rate (Neutral/Total).
2. Anomaly Detection: Flag any Verdicts from Step 4 that seem logically weak or based on insufficient evidence.
3. Source Integrity: Verify that every "Contradicted" verdict is backed by an exact quote verified in Step 1.

Final Output Structure:
- RESEARCH SUMMARY: [Brief overview of consistency]
- METRICS:
  - Total Claims: [N]
  - Conflicted Claims: [Count]
  - Evidence-Missing Claims: [Count]
- DETAILED DISCREPANCIES: [A table or list mapping Claim -> Evidence -> Logic]"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.1,
        proxy: Optional[str] = None
    ):
        """
        Initialize the SynthesisAggregatorAgent.

        Args:
            model_name: Name of the LLM model to use (defaults to settings.openai_model)
            api_key: OpenAI API key or compatible API key
            api_base: Base URL for the API
            temperature: Temperature for LLM responses
            proxy: Optional HTTP proxy for API requests
        """
        model_name = model_name or settings.openai_model
        logger.info(f"Initializing SynthesisAggregatorAgent: model={model_name}, temperature={temperature}")

        # Use shared HTTP client for LLM
        http_client = llm_manager.get_http_client(proxy=proxy)

        self.llm = ChatOpenAI(
            model_name=model_name,
            api_key=api_key or settings.openai_api_key,
            base_url=api_base or settings.openai_api_base,
            temperature=temperature,
            max_retries=settings.openai_max_retries,
            timeout=settings.openai_timeout,
            http_client=http_client
        )

        # Initialize smart memory if enabled
        self.smart_memory = SmartMemory(llm=self.llm, vector_store=None) if settings.smart_memory_enabled else None

        logger.debug("SynthesisAggregatorAgent initialized successfully")

    def _create_synthesis_report(self, conflict_analyses: List[Dict[str, Any]]) -> SynthesisReport:
        """
        Create a synthesis report from conflict analyses.

        Args:
            conflict_analyses: List of conflict analysis results

        Returns:
            SynthesisReport with comprehensive analysis
        """
        logger.debug(f"Creating synthesis report from {len(conflict_analyses)} conflict analyses")

        # Calculate basic metrics
        total_claims = len(conflict_analyses)
        supported_claims = sum(1 for ca in conflict_analyses if ca['verdict'] == 'supported')
        contradicted_claims = sum(1 for ca in conflict_analyses if ca['verdict'] == 'contradicted')
        neutral_missing_claims = sum(1 for ca in conflict_analyses if ca['verdict'] == 'neutral_missing')

        hallucination_rate = neutral_missing_claims / total_claims if total_claims > 0 else 0.0

        # Create detailed discrepancies list
        detailed_discrepancies = []
        for ca in conflict_analyses:
            if ca['verdict'] == 'contradicted':
                discrepancy = {
                    "claim_id": ca['claim_id'],
                    "claim_text": ca['claim_text'],
                    "evidence_quotes": ca['evidence_quotes'],
                    "conflict_type": ca['conflict_type'],
                    "analysis": ca['analysis'],
                    "confidence": ca['confidence']
                }
                detailed_discrepancies.append(discrepancy)

        # Assess overall quality
        if contradicted_claims == 0:
            quality_assessment = "HIGH: No factual conflicts detected. Claims are well-supported by evidence."
            confidence_score = 0.9
        elif contradicted_claims / total_claims <= 0.1:
            quality_assessment = "MEDIUM: Minor conflicts detected but overall consistency is good."
            confidence_score = 0.7
        elif contradicted_claims / total_claims <= 0.3:
            quality_assessment = "LOW: Significant conflicts detected. Source reliability is questionable."
            confidence_score = 0.4
        else:
            quality_assessment = "CRITICAL: Major factual inconsistencies. Source contains substantial inaccuracies."
            confidence_score = 0.1

        # Create research summary
        if contradicted_claims == 0:
            research_summary = f"The analysis found {total_claims} atomic claims, all of which are supported by evidence in the source material. No factual conflicts were detected."
        else:
            research_summary = f"The analysis found {total_claims} atomic claims with {contradicted_claims} conflicts and {neutral_missing_claims} claims lacking evidence. Hallucination rate: {hallucination_rate:.1%}."

        metrics = {
            "total_claims": total_claims,
            "supported_claims": supported_claims,
            "conflicted_claims": contradicted_claims,
            "evidence_missing_claims": neutral_missing_claims,
            "hallucination_rate": hallucination_rate,
            "conflict_types": {}
        }

        # Count conflict types
        for ca in conflict_analyses:
            if ca['verdict'] == 'contradicted':
                conflict_type = ca['conflict_type']
                metrics["conflict_types"][conflict_type] = metrics["conflict_types"].get(conflict_type, 0) + 1

        return SynthesisReport(
            research_summary=research_summary,
            metrics=metrics,
            detailed_discrepancies=detailed_discrepancies,
            quality_assessment=quality_assessment,
            confidence_score=confidence_score
        )

    def _enhance_report_with_llm(self, basic_report: SynthesisReport, conflict_analyses: List[Dict[str, Any]]) -> SynthesisReport:
        """
        Use LLM to enhance the synthesis report with more detailed analysis.

        Args:
            basic_report: Basic synthesis report from statistical analysis
            conflict_analyses: Raw conflict analysis data

        Returns:
            Enhanced SynthesisReport with LLM insights
        """
        logger.debug("Enhancing synthesis report with LLM analysis")

        # Prepare input for LLM
        summary_text = f"""
BASIC ANALYSIS:
{basic_report.research_summary}

METRICS:
- Total Claims: {basic_report.metrics['total_claims']}
- Supported Claims: {basic_report.metrics['supported_claims']}
- Conflicted Claims: {basic_report.metrics['conflicted_claims']}
- Evidence Missing: {basic_report.metrics['evidence_missing_claims']}
- Hallucination Rate: {basic_report.metrics['hallucination_rate']:.1%}

CONFLICT DETAILS:
"""

        for i, ca in enumerate(conflict_analyses[:10]):  # Limit to first 10 for context
            summary_text += f"\nClaim {ca['claim_id']}: {ca['claim_text'][:100]}..."
            summary_text += f"\nVerdict: {ca['verdict']}"
            if ca['verdict'] == 'contradicted':
                summary_text += f"\nConflict Type: {ca['conflict_type']}"
                summary_text += f"\nAnalysis: {ca['analysis'][:200]}..."
            summary_text += "\n"

        user_message = f"""Please enhance this synthesis report with more detailed analysis and insights:

{summary_text}

Provide an enhanced research summary that includes:
1. Overall assessment of source reliability
2. Patterns in the conflicts (if any)
3. Recommendations for using this source
4. Any anomalies or unusual patterns in the data

Focus on academic/research context and provide actionable insights."""

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=user_message)
        ]

        try:
            response = self.llm.invoke(messages)
            enhanced_summary = response.content.strip()

            # Create enhanced report
            enhanced_report = SynthesisReport(
                research_summary=enhanced_summary,
                metrics=basic_report.metrics,
                detailed_discrepancies=basic_report.detailed_discrepancies,
                quality_assessment=basic_report.quality_assessment,
                confidence_score=basic_report.confidence_score
            )

            logger.info("Successfully enhanced synthesis report with LLM")
            return enhanced_report

        except Exception as e:
            logger.error(f"LLM enhancement failed: {e}, using basic report")
            return basic_report

    def aggregate_synthesis(
        self,
        conflict_analyses: List[Dict[str, Any]],
        use_llm_enhancement: bool = True
    ) -> SynthesisAggregationResult:
        """
        Aggregate conflict analyses into a comprehensive synthesis report.

        Args:
            conflict_analyses: List of conflict analysis results
            use_llm_enhancement: Whether to use LLM for enhanced analysis

        Returns:
            SynthesisAggregationResult with comprehensive report
        """
        logger.info(f"Aggregating synthesis from {len(conflict_analyses)} conflict analyses")

        # Create basic statistical report
        basic_report = self._create_synthesis_report(conflict_analyses)

        # Enhance with LLM if requested
        if use_llm_enhancement:
            final_report = self._enhance_report_with_llm(basic_report, conflict_analyses)
        else:
            final_report = basic_report

        result = SynthesisAggregationResult(
            synthesis_report=final_report,
            metadata={
                "model": self.llm.model_name if use_llm_enhancement else None,
                "temperature": self.llm.temperature if use_llm_enhancement else None,
                "use_llm_enhancement": use_llm_enhancement,
                "total_analyses": len(conflict_analyses),
                "conflicted_claims": final_report.metrics['conflicted_claims']
            }
        )

        logger.info(f"Synthesis aggregation complete: {final_report.metrics['conflicted_claims']} conflicts identified, confidence: {final_report.confidence_score:.2f}")

        return result

    def reset(self) -> None:
        """Reset the agent to initial state."""
        logger.info("Resetting SynthesisAggregatorAgent state")
        # No state to reset for this agent
        logger.debug("Agent reset complete")
