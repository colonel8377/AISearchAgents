"""Conflict Auditor Agent for checking logical consistency between claims and evidence.

This agent compares each atomic claim against its supporting evidence to determine
if they are consistent, contradictory, or if evidence is missing.
"""

import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from ...config.settings import settings, ExecutionMode
from ...utils.llm_client import llm_manager
from ...utils.logger import get_logger
from ...utils.smart_memory import SmartMemory

logger = get_logger(__name__)


class ConflictType(Enum):
    """Types of conflicts between claims and evidence."""
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NEUTRAL_MISSING = "neutral_missing"
    NUMERICAL_DISCREPANCY = "numerical_discrepancy"
    TEMPORAL_ERROR = "temporal_error"
    DIRECTIONAL_CONTRADICTION = "directional_contradiction"
    SCOPE_DISTORTION = "scope_distortion"


@dataclass
class ConflictAnalysis:
    """Analysis result for a single claim-evidence pair."""
    claim_id: str
    claim_text: str
    evidence_quotes: List[str]
    verdict: ConflictType
    conflict_type: str
    analysis: str
    confidence: float


@dataclass
class ConflictAuditResult:
    """Result of conflict auditing for multiple claims."""
    conflict_analyses: List[ConflictAnalysis]
    summary_stats: Dict[str, int]
    execution_mode: str
    metadata: Optional[Dict[str, Any]] = None


class ConflictAuditorAgent:
    """
    Conflict Auditor Agent that compares claims against evidence to detect
    logical inconsistencies and factual conflicts.

    Features:
    - Detailed conflict type classification
    - Chain of Thought reasoning support
    - Few-shot example support
    - Confidence scoring
    """

    SYSTEM_PROMPT = """You are an Academic Peer Reviewer specializing in Fact-Consistency Auditing.

Task: Compare the [Claim] against the [Extracted Evidence] and determine the relationship.

Definitions:
- SUPPORTED: The evidence explicitly confirms the claim.
- CONTRADICTED: The evidence explicitly negates or provides different data/facts than the claim.
- NEUTRAL/MISSING: The evidence is insufficient to prove or disprove the claim.

Logical Rigor:
1. Categorize the Conflict: If "Contradicted", specify if it is a Numerical discrepancy, Temporal error, Directional contradiction (e.g., increase vs decrease), or Scope distortion.
2. Reasoning: Explain the logical gap.

Output Format:
- CLAIM_ID: [ID]
- VERDICT: [Supported/Contradicted/Neutral]
- CONFLICT_TYPE: [Type]
- ANALYSIS: [Reasoning]"""

    SYSTEM_PROMPT_COT = """You are an Academic Peer Reviewer specializing in Fact-Consistency Auditing.

Task: Compare the [Claim] against the [Extracted Evidence] and determine the relationship.

Definitions:
- SUPPORTED: The evidence explicitly confirms the claim.
- CONTRADICTED: The evidence explicitly negates or provides different data/facts than the claim.
- NEUTRAL/MISSING: The evidence is insufficient to prove or disprove the claim.

Logical Rigor:
1. Categorize the Conflict: If "Contradicted", specify if it is a Numerical discrepancy, Temporal error, Directional contradiction (e.g., increase vs decrease), or Scope distortion.
2. Reasoning: Explain the logical gap.

Configuration:
- [CoT Mode]: Enabled - Explain the comparison logic step-by-step (e.g., "Claim says X, Evidence says Y; X and Y are mutually exclusive because...").

Output Format:
- CLAIM_ID: [ID]
- VERDICT: [Supported/Contradicted/Neutral]
- CONFLICT_TYPE: [Type]
- ANALYSIS: [Reasoning]"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.1,
        proxy: Optional[str] = None,
        execution_mode: Optional[ExecutionMode] = None
    ):
        """
        Initialize the ConflictAuditorAgent.

        Args:
            model_name: Name of the LLM model to use (defaults to settings.openai_model)
            api_key: OpenAI API key or compatible API key
            api_base: Base URL for the API
            temperature: Temperature for LLM responses
            proxy: Optional HTTP proxy for API requests
            execution_mode: Execution mode for CoT ('chain_online', 'chain_local', 'no_chain')
        """
        model_name = model_name or settings.openai_model
        logger.info(f"Initializing ConflictAuditorAgent: model={model_name}, temperature={temperature}, execution_mode={execution_mode}")

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

        self.execution_mode = execution_mode or settings.default_execution_mode

        # Initialize smart memory if enabled
        self.smart_memory = SmartMemory(llm=self.llm, vector_store=None) if settings.smart_memory_enabled else None

        logger.debug(f"ConflictAuditorAgent initialized successfully with execution_mode={self.execution_mode}")

    def _audit_conflicts_with_llm(
        self,
        claim_evidences: List[Dict[str, Any]],
        use_cot: bool = False,
        custom_few_shots: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to audit conflicts between claims and evidence.

        Args:
            claim_evidences: List of claim-evidence pairs
            use_cot: Whether to use Chain of Thought reasoning
            custom_few_shots: Optional custom few-shot examples

        Returns:
            List of conflict analysis results
        """
        logger.debug(f"Auditing conflicts for {len(claim_evidences)} claim-evidence pairs with CoT={use_cot}")

        # Select system prompt based on CoT mode
        if use_cot:
            system_prompt = self.SYSTEM_PROMPT_COT
        else:
            system_prompt = self.SYSTEM_PROMPT

        # Add few-shot examples if provided
        if custom_few_shots:
            system_prompt = f"{custom_few_shots}\n\n{system_prompt}"

        # Prepare input text
        claims_text = ""
        for i, ce in enumerate(claim_evidences, 1):
            claims_text += f"\nCLAIM_{ce['claim_id']}: {ce['claim_text']}\n"
            if ce['evidence_quotes']:
                quotes_text = "\n".join([f'  - "{quote}"' for quote in ce['evidence_quotes']])
                claims_text += f"EVIDENCE:\n{quotes_text}\n"
            else:
                claims_text += "EVIDENCE: NO_EVIDENCE_FOUND\n"

        user_message = f"""Please audit the following claims against their evidence:

{claims_text}

For each claim, provide analysis in the specified format."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]

        try:
            response = self.llm.invoke(messages)
            response_text = response.content

            # Parse analyses from response
            analyses = []

            for ce in claim_evidences:
                claim_id = ce['claim_id']
                analysis_pattern = rf'- CLAIM_ID:\s*{claim_id}(.*?)(?=\n- CLAIM_ID:|\n*$)'

                match = re.search(analysis_pattern, response_text, re.DOTALL)
                if match:
                    analysis_text = match.group(1).strip()

                    # Extract verdict, conflict type, and analysis
                    verdict_match = re.search(r'- VERDICT:\s*(.+?)(?=\n|$)', analysis_text)
                    conflict_type_match = re.search(r'- CONFLICT_TYPE:\s*(.+?)(?=\n|$)', analysis_text)
                    analysis_match = re.search(r'- ANALYSIS:\s*(.+?)(?=\n|$)', analysis_text)

                    verdict_str = verdict_match.group(1).strip() if verdict_match else "Neutral"
                    conflict_type_str = conflict_type_match.group(1).strip() if conflict_type_match else "Unknown"
                    analysis_str = analysis_match.group(1).strip() if analysis_match else "No analysis provided"

                    # Map verdict string to ConflictType
                    verdict_map = {
                        "supported": ConflictType.SUPPORTED,
                        "contradicted": ConflictType.CONTRADICTED,
                        "neutral": ConflictType.NEUTRAL_MISSING,
                        "missing": ConflictType.NEUTRAL_MISSING
                    }
                    verdict = verdict_map.get(verdict_str.lower(), ConflictType.NEUTRAL_MISSING)

                    analyses.append({
                        "claim_id": claim_id,
                        "claim_text": ce['claim_text'],
                        "evidence_quotes": ce['evidence_quotes'],
                        "verdict": verdict,
                        "conflict_type": conflict_type_str,
                        "analysis": analysis_str,
                        "confidence": 0.8  # Default confidence
                    })
                else:
                    # Default analysis if parsing failed
                    analyses.append({
                        "claim_id": claim_id,
                        "claim_text": ce['claim_text'],
                        "evidence_quotes": ce['evidence_quotes'],
                        "verdict": ConflictType.NEUTRAL_MISSING,
                        "conflict_type": "parsing_error",
                        "analysis": "Failed to parse LLM response",
                        "confidence": 0.5
                    })

            logger.info(f"Conflict audit complete: analyzed {len(analyses)} claims")
            return analyses

        except Exception as e:
            logger.error(f"LLM conflict audit failed: {e}", exc_info=True)
            # Return default analyses for all claims
            return [{
                "claim_id": ce['claim_id'],
                "claim_text": ce['claim_text'],
                "evidence_quotes": ce['evidence_quotes'],
                "verdict": ConflictType.NEUTRAL_MISSING,
                "conflict_type": "audit_failed",
                "analysis": f"Conflict audit failed: {str(e)}",
                "confidence": 0.0
            } for ce in claim_evidences]

    def audit_conflicts(
        self,
        claim_evidences: List[Dict[str, Any]],
        use_cot: bool = False,
        custom_few_shots: Optional[str] = None
    ) -> ConflictAuditResult:
        """
        Audit conflicts between claims and their evidence.

        Args:
            claim_evidences: List of claim-evidence pairs with keys: claim_id, claim_text, evidence_quotes
            use_cot: Whether to use Chain of Thought reasoning
            custom_few_shots: Optional custom few-shot examples

        Returns:
            ConflictAuditResult with detailed analyses
        """
        logger.info(f"Auditing conflicts for {len(claim_evidences)} claim-evidence pairs with CoT={use_cot}")

        # Perform conflict auditing
        analysis_data = self._audit_conflicts_with_llm(
            claim_evidences,
            use_cot=use_cot,
            custom_few_shots=custom_few_shots
        )

        # Convert to ConflictAnalysis objects
        conflict_analyses = []
        for data in analysis_data:
            analysis = ConflictAnalysis(
                claim_id=data['claim_id'],
                claim_text=data['claim_text'],
                evidence_quotes=data['evidence_quotes'],
                verdict=data['verdict'],
                conflict_type=data['conflict_type'],
                analysis=data['analysis'],
                confidence=data['confidence']
            )
            conflict_analyses.append(analysis)

        # Calculate summary statistics
        verdict_counts = {}
        for analysis in conflict_analyses:
            verdict_name = analysis.verdict.value
            verdict_counts[verdict_name] = verdict_counts.get(verdict_name, 0) + 1

        summary_stats = {
            "total_claims": len(conflict_analyses),
            "supported": verdict_counts.get(ConflictType.SUPPORTED.value, 0),
            "contradicted": verdict_counts.get(ConflictType.CONTRADICTED.value, 0),
            "neutral_missing": verdict_counts.get(ConflictType.NEUTRAL_MISSING.value, 0),
            "conflict_types": {}
        }

        # Count conflict types
        for analysis in conflict_analyses:
            if analysis.verdict == ConflictType.CONTRADICTED:
                conflict_type = analysis.conflict_type
                summary_stats["conflict_types"][conflict_type] = summary_stats["conflict_types"].get(conflict_type, 0) + 1

        result = ConflictAuditResult(
            conflict_analyses=conflict_analyses,
            summary_stats=summary_stats,
            execution_mode="chain_online" if use_cot else "no_chain",
            metadata={
                "model": self.llm.model_name,
                "temperature": self.llm.temperature,
                "use_cot": use_cot,
                "total_claims": len(conflict_analyses)
            }
        )

        supported_count = summary_stats["supported"]
        contradicted_count = summary_stats["contradicted"]
        logger.info(f"Conflict audit complete: {supported_count} supported, {contradicted_count} contradicted, {summary_stats['neutral_missing']} neutral/missing")

        return result

    @staticmethod
    def get_default_few_shots() -> str:
        """
        Get default few-shot examples for conflict auditing.

        Returns:
            String containing few-shot examples
        """
        return """Example 1 - Numerical Discrepancy:
CLAIM_1: The company reported $2.1 billion in revenue for Q3 2023.
EVIDENCE:
  - "The company reported $2.3 billion in revenue for Q3 2023, representing a 15% increase."

- CLAIM_ID: 1
- VERDICT: Contradicted
- CONFLICT_TYPE: Numerical discrepancy
- ANALYSIS: The claim states $2.1 billion while the evidence shows $2.3 billion. This is a direct numerical contradiction.

Example 2 - Supported Claim:
CLAIM_2: The revenue represents a 15% increase.
EVIDENCE:
  - "The company reported $2.3 billion in revenue for Q3 2023, representing a 15% increase."

- CLAIM_ID: 2
- VERDICT: Supported
- CONFLICT_TYPE: N/A
- ANALYSIS: The evidence explicitly confirms the 15% increase figure.

Example 3 - Missing Evidence:
CLAIM_3: The CEO is named John Smith.
EVIDENCE: NO_EVIDENCE_FOUND

- CLAIM_ID: 3
- VERDICT: Neutral
- CONFLICT_TYPE: Missing evidence
- ANALYSIS: No evidence was found in the text to either confirm or contradict the CEO's name."""

    def reset(self) -> None:
        """Reset the agent to initial state."""
        logger.info("Resetting ConflictAuditorAgent state")
        # No state to reset for this agent
        logger.debug("Agent reset complete")
