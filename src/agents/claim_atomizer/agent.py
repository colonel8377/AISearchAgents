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
from ...utils.agent_cache import cached

logger = get_logger(__name__)


@dataclass
class AtomicClaim:
    """An atomic claim extracted from text."""
    id: str
    text: str
    original_sentence: str
    confidence: float
    paragraph_index: Optional[int] = None  # Index of the paragraph this claim came from
    paragraph_text: Optional[str] = None   # Full text of the paragraph this claim came from


@dataclass
class ParagraphClaims:
    """Claims from a single paragraph."""
    paragraph_index: int
    paragraph_text: str
    atomic_claims: List[AtomicClaim]

@dataclass
class ClaimAtomizationResult:
    """Result of claim atomization."""
    atomic_claims: List[AtomicClaim]  # Flat list for backward compatibility
    paragraphs: List[ParagraphClaims]  # New: claims grouped by paragraphs
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

    # Class variable to store custom few shots (persistent across instances)
    _custom_few_shots: Optional[str] = None

    SYSTEM_PROMPT = """You are a Linguistic Logic Analyst. Your task is to perform Atomic Claim Decomposition.

Task: Break down the provided [Snippet] into a list of independent, verifiable atomic claims.

Requirements:
1. Each claim must contain only ONE factual point (e.g., a specific number, a date, a causal relationship, or a specific entity\'s action).
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
1. Each claim must contain only ONE factual point (e.g., a specific number, a date, a causal relationship, or a specific entity\'s action).
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

    @cached()
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

        # Add few-shot examples
        if custom_few_shots:
            # Use explicitly provided custom few shots
            system_prompt = f"{custom_few_shots}\n\n{system_prompt}"
        elif self._custom_few_shots:
            # Use stored custom few shots
            system_prompt = f"{self._custom_few_shots}\n\n{system_prompt}"

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

                # Try to find the sentence containing this claim
                original_sentence = self._find_sentence_containing_claim(text, claim_text)
                if not original_sentence:
                    # Fallback: truncate the text if it's too long
                    original_sentence = text[:200] + "..." if len(text) > 200 else text

                claims.append({
                    "id": claim_id,
                    "text": claim_text,
                    "original_sentence": original_sentence,
                    "confidence": 0.8  # Default confidence
                })

            logger.info(f"Extracted {len(claims)} atomic claims")
            return claims

        except Exception as e:
            logger.error(f"LLM atomization failed: {e}", exc_info=True)
            return []

    def _find_sentence_containing_claim(self, text: str, claim_text: str) -> str:
        """
        Find the sentence in the text that contains the given claim.

        Args:
            text: The full text to search in
            claim_text: The claim text to find

        Returns:
            The sentence containing the claim, or empty string if not found
        """
        import re

        # Split text into sentences (simple approach)
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        # Find the sentence that contains the most words from the claim
        best_sentence = ""
        max_overlap = 0

        claim_words = set(re.findall(r'\b\w+\b', claim_text.lower()))

        for sentence in sentences:
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            overlap = len(claim_words.intersection(sentence_words))

            if overlap > max_overlap:
                max_overlap = overlap
                best_sentence = sentence.strip()

        # If we found a good match, return it (limit length)
        if best_sentence and len(best_sentence) <= 500:
            return best_sentence

        # If no good sentence found or too long, return a truncated version
        return ""

    def _split_text_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs using robust paragraph detection.

        Args:
            text: The text to split

        Returns:
            List of paragraphs
        """
        import re

        # Split by double newlines (common paragraph separator)
        paragraphs = re.split(r'\n\s*\n', text.strip())

        # Filter out empty paragraphs and clean up
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # If no paragraphs found, treat the whole text as one paragraph
        if not paragraphs:
            paragraphs = [text.strip()]

        # Filter out very short paragraphs (likely headers or separators)
        paragraphs = [p for p in paragraphs if len(p) > 20]

        return paragraphs

    def atomize_text(
        self,
        text: str,
        use_cot: bool = False,
        custom_few_shots: Optional[str] = None,
        split_into_paragraphs: bool = False
    ) -> ClaimAtomizationResult:
        """
        Decompose text into atomic claims.

        Args:
            text: Text snippet to decompose
            use_cot: Whether to use Chain of Thought reasoning
            custom_few_shots: Optional custom few-shot examples
            split_into_paragraphs: Whether to split text into paragraphs before atomization

        Returns:
            ClaimAtomizationResult with atomic claims
        """
        logger.info(f"Atomizing text ({len(text)} chars) with CoT={use_cot}, split_paragraphs={split_into_paragraphs}")

        if split_into_paragraphs:
            # Split text into paragraphs and process each separately
            paragraphs = self._split_text_into_paragraphs(text)
            logger.info(f"Split text into {len(paragraphs)} paragraphs")

            all_raw_claims = []
            claim_counter = 0

            # Process each paragraph separately
            for para_idx, paragraph in enumerate(paragraphs):
                logger.debug(f"Processing paragraph {para_idx + 1}/{len(paragraphs)} ({len(paragraph)} chars)")

                # Skip very short paragraphs
                if len(paragraph) < 50:
                    logger.debug(f"Skipping short paragraph: {paragraph[:50]}...")
                    continue

                # Decompose paragraph into atomic claims
                para_claims = self._atomize_claims_with_llm(paragraph, use_cot=use_cot, custom_few_shots=custom_few_shots)

                # Add paragraph information to each claim
                for claim_data in para_claims:
                    claim_data['paragraph_index'] = para_idx
                    claim_data['paragraph_text'] = paragraph
                    # Update claim IDs to be unique across paragraphs
                    if 'id' in claim_data:
                        claim_data['id'] = f"P{para_idx + 1}_{claim_data['id']}"
                    else:
                        claim_counter += 1
                        claim_data['id'] = f"P{para_idx + 1}_{claim_counter}"

                all_raw_claims.extend(para_claims)

            logger.info(f"Extracted {len(all_raw_claims)} atomic claims from {len(paragraphs)} paragraphs")
        else:
            # Process the entire text as one unit (for summaries)
            all_raw_claims = self._atomize_claims_with_llm(text, use_cot=use_cot, custom_few_shots=custom_few_shots)

            # Reformat IDs for summary claims to be consistent with paragraph format (P1_1, P1_2, etc.)
            for i, claim_data in enumerate(all_raw_claims, 1):
                claim_data['id'] = f"P1_{i}"

            logger.info(f"Extracted {len(all_raw_claims)} atomic claims from entire text")

        # Convert to AtomicClaim objects
        atomic_claims = []
        for data in all_raw_claims:
            try:
                claim_text = data.get("text", "")
                original_sentence = data.get("original_sentence")

                if split_into_paragraphs:
                    # When split into paragraphs, use paragraph context
                    paragraph_text = data.get("paragraph_text", text)
                    paragraph_index = data.get("paragraph_index")
                else:
                    # When not split, treat entire text as context
                    paragraph_text = text
                    paragraph_index = None

                # If no original_sentence provided, try to find it in the context
                if not original_sentence:
                    original_sentence = self._find_sentence_containing_claim(paragraph_text, claim_text)
                    if not original_sentence:
                        # Fallback: use the context text (truncated if too long)
                        original_sentence = paragraph_text[:300] + "..." if len(paragraph_text) > 300 else paragraph_text

                claim = AtomicClaim(
                    id=data.get("id", ""),
                    text=claim_text,
                    original_sentence=original_sentence,
                    confidence=data.get("confidence", 0.8),
                    paragraph_index=paragraph_index,
                    paragraph_text=paragraph_text if split_into_paragraphs else None
                )
                atomic_claims.append(claim)
            except Exception as e:
                logger.warning(f"Failed to create atomic claim: {e}")
                continue

        # Group claims by paragraphs (only if text was split into paragraphs)
        if split_into_paragraphs:
            paragraph_groups = {}
            for claim in atomic_claims:
                para_idx = claim.paragraph_index
                if para_idx is not None:
                    if para_idx not in paragraph_groups:
                        paragraph_groups[para_idx] = {
                            'paragraph_index': para_idx,
                            'paragraph_text': claim.paragraph_text or '',
                            'atomic_claims': []
                        }
                    paragraph_groups[para_idx]['atomic_claims'].append(claim)

            # Convert to ParagraphClaims objects
            paragraphs = []
            for para_idx in sorted(paragraph_groups.keys()):
                para_data = paragraph_groups[para_idx]
                paragraphs.append(ParagraphClaims(
                    paragraph_index=para_data['paragraph_index'],
                    paragraph_text=para_data['paragraph_text'],
                    atomic_claims=para_data['atomic_claims']
                ))
        else:
            paragraphs = []

        result = ClaimAtomizationResult(
            atomic_claims=atomic_claims,  # Keep flat list for backward compatibility
            paragraphs=paragraphs,        # New: claims grouped by paragraphs
            original_text=text,
            execution_mode="chain_online" if use_cot else "no_chain",
            metadata={
                "model": self.llm.model_name,
                "temperature": self.llm.temperature,
                "use_cot": use_cot,
                "total_claims": len(atomic_claims),
                "total_paragraphs": len(paragraphs)
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

    @classmethod
    def set_custom_few_shots(cls, custom_few_shots: Optional[str] = None) -> None:
        """Set custom few-shot examples for claim atomization."""
        cls._custom_few_shots = custom_few_shots
        logger.info("Custom few shots set for ClaimAtomizerAgent: {}".format(custom_few_shots is not None))

    @classmethod
    def get_custom_few_shots(cls) -> Optional[str]:
        """Get currently set custom few-shot examples."""
        return cls._custom_few_shots

    @classmethod
    def get_effective_few_shots(cls) -> str:
        """Get effective few-shot examples (custom if set, otherwise default)."""
        return cls._custom_few_shots if cls._custom_few_shots is not None else cls.get_default_few_shots()

    def reset(self) -> None:
        """Reset the agent to initial state."""
        logger.info("Resetting ClaimAtomizerAgent state")
        # No state to reset for this agent
        logger.debug("Agent reset complete")
