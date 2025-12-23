"""Claim Atomizer Agent for decomposing text into atomic claims.

This agent breaks down provided text snippets into independent, verifiable
atomic claims, each containing only one factual point.
"""

import json
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from ...config.settings import settings, ExecutionMode
from ...utils.llm_client import llm_manager
from ...utils.logger import get_logger
from ...utils.smart_memory import SmartMemory

logger = get_logger(__name__)


@dataclass
class AtomicClaim:
    """An atomic claim extracted from text."""
    id: str
    text: str
    original_sentence: str
    confidence: float


@dataclass
class ClaimAtomizationResult:
    """Result of claim atomization."""
    atomic_claims: List[AtomicClaim]
    original_text: str
    execution_mode: str
    metadata: Optional[Dict[str, Any]] = None


class ClaimAtomizerAgent:
    """
    Claim Atomizer Agent that decomposes text snippets into independent,
    verifiable atomic claims.

    Features:
    - Atomic claim decomposition (one fact per claim)
    - Chain of Thought reasoning support
    - Few-shot example support
    - Confidence scoring
    """

    SYSTEM_PROMPT = """You are a Linguistic Logic Analyst. Your task is to perform Atomic Claim Decomposition.

Task: Break down the provided [Snippet] into a list of independent, verifiable atomic claims.

Requirements:
1. Each claim must contain only ONE factual point (e.g., a specific number, a date, a causal relationship, or a specific entity's action).
2. Ensure each claim is a self-contained sentence (include the subject; do not use pronouns).
3. Neutrality: Remove rhetorical flourishes, marketing adjectives, and subjective qualifiers.

Output Format:
- CLAIM_1: [Fact]
- CLAIM_2: [Fact]
- CLAIM_3: [Fact]
..."""

    SYSTEM_PROMPT_COT = """You are a Linguistic Logic Analyst. Your task is to perform Atomic Claim Decomposition.

Task: Break down the provided [Snippet] into a list of independent, verifiable atomic claims.

Requirements:
1. Each claim must contain only ONE factual point (e.g., a specific number, a date, a causal relationship, or a specific entity's action).
2. Ensure each claim is a self-contained sentence (include the subject; do not use pronouns).
3. Neutrality: Remove rhetorical flourishes, marketing adjectives, and subjective qualifiers.

Configuration:
- [CoT Mode]: Enabled - list the logical components of each sentence before extracting claims.

Output Format:
- CLAIM_1: [Fact]
- CLAIM_2: [Fact]
- CLAIM_3: [Fact]
..."""

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
        Initialize the ClaimAtomizerAgent.

        Args:
            model_name: Name of the LLM model to use (defaults to settings.openai_model)
            api_key: OpenAI API key or compatible API key
            api_base: Base URL for the API
            temperature: Temperature for LLM responses
            proxy: Optional HTTP proxy for API requests
            execution_mode: Execution mode for CoT ('chain_online', 'chain_local', 'no_chain')
        """
        model_name = model_name or settings.openai_model
        logger.info(f"Initializing ClaimAtomizerAgent: model={model_name}, temperature={temperature}, execution_mode={execution_mode}")

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

        logger.debug(f"ClaimAtomizerAgent initialized successfully with execution_mode={self.execution_mode}")

    def _atomize_claims_with_llm(self, text: str, use_cot: bool = False, custom_few_shots: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Use LLM to decompose text into atomic claims.

        Args:
            text: Text snippet to decompose
            use_cot: Whether to use Chain of Thought reasoning
            custom_few_shots: Optional custom few-shot examples

        Returns:
            List of claim dictionaries
        """
        logger.debug(f"Atomizing claims from text ({len(text)} chars) with CoT={use_cot}")

        # Select system prompt based on CoT mode
        if use_cot:
            system_prompt = self.SYSTEM_PROMPT_COT
        else:
            system_prompt = self.SYSTEM_PROMPT

        # Add few-shot examples if provided
        if custom_few_shots:
            system_prompt = f"{custom_few_shots}\n\n{system_prompt}"

        user_message = f"""Please decompose the following text into atomic claims:

{text}

Return each claim in the format:
- CLAIM_1: [Atomic fact]
- CLAIM_2: [Atomic fact]
..."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]

        try:
            response = self.llm.invoke(messages)
            response_text = response.content

            # Parse claims from response
            claims = []
            claim_pattern = r'- CLAIM_(\d+):\s*(.+?)(?=\n- CLAIM_|\n*$)'

            for match in re.finditer(claim_pattern, response_text, re.DOTALL):
                claim_id = match.group(1)
                claim_text = match.group(2).strip()

                claims.append({
                    "id": claim_id,
                    "text": claim_text,
                    "original_sentence": text,  # In a more sophisticated implementation, we'd track which sentence each claim came from
                    "confidence": 0.8  # Default confidence
                })

            logger.info(f"Extracted {len(claims)} atomic claims")
            return claims

        except Exception as e:
            logger.error(f"LLM atomization failed: {e}", exc_info=True)
            return []

    def atomize_text(
        self,
        text: str,
        use_cot: bool = False,
        custom_few_shots: Optional[str] = None
    ) -> ClaimAtomizationResult:
        """
        Decompose text into atomic claims.

        Args:
            text: Text snippet to decompose
            use_cot: Whether to use Chain of Thought reasoning
            custom_few_shots: Optional custom few-shot examples

        Returns:
            ClaimAtomizationResult with atomic claims
        """
        logger.info(f"Atomizing text ({len(text)} chars) with CoT={use_cot}")

        # Decompose text into atomic claims
        raw_claims = self._atomize_claims_with_llm(text, use_cot=use_cot, custom_few_shots=custom_few_shots)

        # Convert to AtomicClaim objects
        atomic_claims = []
        for data in raw_claims:
            try:
                claim = AtomicClaim(
                    id=data.get("id", ""),
                    text=data.get("text", ""),
                    original_sentence=data.get("original_sentence", text),
                    confidence=data.get("confidence", 0.8)
                )
                atomic_claims.append(claim)
            except Exception as e:
                logger.warning(f"Failed to create atomic claim: {e}")
                continue

        result = ClaimAtomizationResult(
            atomic_claims=atomic_claims,
            original_text=text,
            execution_mode="chain_online" if use_cot else "no_chain",
            metadata={
                "model": self.llm.model_name,
                "temperature": self.llm.temperature,
                "use_cot": use_cot,
                "total_claims": len(atomic_claims)
            }
        )

        logger.info(f"Atomization complete: {len(atomic_claims)} atomic claims extracted")

        return result

    @staticmethod
    def get_default_few_shots() -> str:
        """
        Get default few-shot examples for claim atomization.

        Returns:
            String containing few-shot examples
        """
        return """Example 1:
Input: "The company reported $2.1 billion in revenue for Q3 2023, which represents a 15% increase from the previous year. CEO John Smith stated that this growth was driven by strong performance in the Asian market."

Output:
- CLAIM_1: The company reported $2.1 billion in revenue for Q3 2023.
- CLAIM_2: The company's Q3 2023 revenue represents a 15% increase from the previous year.
- CLAIM_3: CEO John Smith stated that this growth was driven by strong performance in the Asian market.

Example 2:
Input: "Dr. Sarah Johnson, the lead researcher, discovered a new treatment that reduces symptoms by 40% in clinical trials involving 500 patients."

Output:
- CLAIM_1: Dr. Sarah Johnson is the lead researcher.
- CLAIM_2: Dr. Sarah Johnson discovered a new treatment.
- CLAIM_3: The new treatment reduces symptoms by 40% in clinical trials.
- CLAIM_4: The clinical trials involved 500 patients."""

    def reset(self) -> None:
        """Reset the agent to initial state."""
        logger.info("Resetting ClaimAtomizerAgent state")
        # No state to reset for this agent
        logger.debug("Agent reset complete")
