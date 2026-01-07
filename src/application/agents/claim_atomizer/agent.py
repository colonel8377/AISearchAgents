"""Claim Atomizer Agent for decomposing text into atomic claims.

This agent breaks down provided text snippets into independent, verifiable
atomic claims, each containing only one factual point.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from .model import AtomicClaim
from ...few_shots.claim_atomizer.few_shots import CLAIM_ATOMIZER_FEW_SHOTS
from ....infrastructure.repositories import AgentProtocol
from ....infrastructure.storage.persistence import get_database
from ....shared.constant.enums import CoTMode
from ....shared.config.settings import settings, ExecutionMode
from src.shared.llm import llm_manager
from ....shared.utils.logger import get_logger
from .utils import TextProcessor, ClaimParser, ClaimFormatter

logger = get_logger(__name__)


class ClaimAtomizerAgent(AgentProtocol):
    """
    Claim Atomizer Agent that decomposes text snippets into independent,
    verifiable atomic claims.

    Features:
    - Atomic claim decomposition (one fact per claim)
    - Chain of Thought reasoning support
    - Few-shot example support
    - Confidence scoring
    - Async LLM calls
    """

    _custom_few_shots_cache: Optional[str] = None

    SYSTEM_PROMPT = """You are a Linguistic Logic Analyst specializing in Atomic Claim Decomposition.

Task: Break down the provided text into independent, verifiable atomic claims. Each claim must be a single, concrete factual assertion.

Requirements:
1. Each claim must contain ONLY ONE factual point that can be empirically verified.
2. Make claims self-contained with clear subjects and complete information.
3. Focus on concrete facts: numbers, dates, events, relationships, properties.
4. IMPORTANT: Do NOT decompose questions, rhetorical questions, or interrogative sentences into claims. If the input contains questions, skip them entirely.

Output Format:
- CLAIM_1: [Single concrete factual claim]
- CLAIM_2: [Single concrete factual claim]
- CLAIM_3: [Single concrete factual claim]
..."""

    SYSTEM_PROMPT_COT = """You are a Linguistic Logic Analyst specializing in Atomic Claim Decomposition with Chain-of-Thought reasoning.

Task: Analyze the provided text step-by-step, then break it down into independent, verifiable atomic claims.

Step-by-Step Process:
1. IDENTIFY: List all sentences and phrases in the text
2. CLASSIFY: Determine if each part is a factual claim, question, opinion, or rhetorical element
3. FILTER: Remove questions, opinions, and non-factual content
4. DECOMPOSE: Break remaining factual content into atomic claims
5. VALIDATE: Ensure each claim contains only one verifiable fact

Requirements:
1. Each claim must contain ONLY ONE factual point that can be empirically verified.
2. Make claims self-contained with clear subjects and complete information.
3. Remove all subjective language, opinions, and rhetorical elements.
4. IMPORTANT: Skip all questions, rhetorical questions, and interrogative sentences entirely.
5. Focus on concrete facts: numbers, dates, events, relationships, properties.

Output Format:
First, show your analysis:
ANALYSIS: [Brief step-by-step reasoning]

Then list claims:
- CLAIM_1: [Single concrete factual claim]
- CLAIM_2: [Single concrete factual claim]
- CLAIM_3: [Single concrete factual claim]
..."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.1,
        execution_mode: Optional[ExecutionMode] = None
    ):
        """
        Initialize the ClaimAtomizerAgent.

        Args:
            model_name: Name of the LLM model to use (defaults to settings.openai_model)
            api_key: OpenAI API key or compatible API key
            api_base: Base URL for the API
            temperature: Temperature for LLM responses
            execution_mode: Execution mode for CoT (deprecated, kept for compatibility)
        """
        model_name = model_name or settings.openai_model
        logger.info(f"Initializing ClaimAtomizerAgent: model={model_name}, temperature={temperature}")

        http_client = llm_manager.get_http_client()

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

        # Initialize custom few shots
        self._custom_few_shots = None
        if settings.enable_persistence:
            try:
                database = get_database()
                self._custom_few_shots = database.load_custom_few_shots("claim_atomizer")
            except Exception as e:
                logger.warning(f"Failed to load custom few shots: {e}")

        logger.debug(f"ClaimAtomizerAgent initialized successfully")

    def _get_system_prompt(self, use_cot: Union[CoTMode, bool], use_few_shots: bool = True) -> str:
        """
        Get the appropriate system prompt based on CoT mode and few-shot settings.

        Args:
            use_cot: Whether to use Chain of Thought reasoning
            use_few_shots: Whether to use few-shot examples

        Returns:
            System prompt string
        """
        # Determine effective CoT mode
        if isinstance(use_cot, CoTMode):
            effective_cot = use_cot.value in ("chain_local", "chain_online")
        else:
            effective_cot = bool(use_cot)

        # Select base prompt
        base_prompt = self.SYSTEM_PROMPT_COT if effective_cot else self.SYSTEM_PROMPT

        # Add few-shot examples if enabled
        if use_few_shots:
            if self._custom_few_shots:
                return f"{self._custom_few_shots}\n\n{base_prompt}"
            else:
                # Use default few shots
                return f"{CLAIM_ATOMIZER_FEW_SHOTS}\n\n{base_prompt}"

        return base_prompt

    async def _atomize_text_with_llm(
        self,
        text: str,
        use_cot: Union[CoTMode, bool] = CoTMode.NO_CHAIN,
        use_few_shots: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to decompose text into atomic claims (async).

        Args:
            text: Text snippet to decompose
            use_cot: Whether to use Chain of Thought reasoning
            use_few_shots: Whether to use few-shot examples

        Returns:
            List of claim dictionaries
        """
        logger.debug(f"Atomizing claims from text ({len(text)} chars) with CoT={use_cot}, use_few_shots={use_few_shots}")

        system_prompt = self._get_system_prompt(use_cot, use_few_shots=use_few_shots)

        user_message = f"""Please decompose the following text into atomic claims. Remember: do NOT extract claims from questions or interrogative sentences.

Text to analyze:
{text}

Return each atomic claim in the format:
- CLAIM_1: [Single concrete atomic factual claim]
- CLAIM_2: [Single concrete atomic factual claim]
..."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]

        try:
            response = await self.llm.ainvoke(messages)
            response_text = response.content

            claims = ClaimParser.parse_claims_from_response(response_text, text)
            logger.info(f"Extracted {len(claims)} atomic claims")
            return claims

        except Exception as e:
            logger.error(f"LLM atomization failed: {e}", exc_info=True)
            return []

    async def _process_paragraphs(
        self,
        paragraphs: List[str],
        use_cot: Union[CoTMode, bool],
        use_few_shots: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process multiple paragraphs concurrently and extract claims.

        Args:
            paragraphs: List of paragraph texts
            use_cot: Chain of Thought mode
            use_few_shots: Whether to use few-shot examples

        Returns:
            List of formatted claim dictionaries with paragraph information
        """
        # Process paragraphs concurrently
        tasks = [
            self._atomize_text_with_llm(paragraph, use_cot=use_cot, use_few_shots=use_few_shots)
            for paragraph in paragraphs
        ]

        all_paragraph_results = await asyncio.gather(*tasks)

        # Format claims with paragraph information
        all_claims = []
        for para_idx, (paragraph, para_claims) in enumerate(zip(paragraphs, all_paragraph_results)):
            for claim_data in para_claims:
                formatted_claim = ClaimFormatter.format_claim_data(
                    claim_data=claim_data,
                    paragraph_index=para_idx,
                    paragraph_text=paragraph,
                    source_text=paragraph
                )
                # Update claim ID to be unique across paragraphs
                formatted_claim['id'] = f"P{para_idx + 1}_{claim_data.get('id', para_idx)}"
                all_claims.append(formatted_claim)

        return all_claims

    def _create_atomic_claims(
        self,
        claim_data_list: List[Dict[str, Any]],
        source_text: str,
        split_into_paragraphs: bool
    ) -> List[AtomicClaim]:
        """
        Convert claim data dictionaries to AtomicClaim objects.

        Args:
            claim_data_list: List of claim data dictionaries
            source_text: Original source text
            split_into_paragraphs: Whether text was split into paragraphs

        Returns:
            List of AtomicClaim objects
        """
        atomic_claims = []
        for data in claim_data_list:
            try:
                claim = AtomicClaim(
                    id=data.get("id", ""),
                    text=data.get("text", ""),
                    original_sentence=data.get("original_sentence", ""),
                    confidence=data.get("confidence", 0.8),
                    paragraph_index=data.get("paragraph_index") if split_into_paragraphs else None,
                    paragraph_text=data.get("paragraph_text") if split_into_paragraphs else None
                )
                atomic_claims.append(claim)
            except Exception as e:
                logger.warning(f"Failed to create atomic claim: {e}")
                continue

        return atomic_claims

    def _group_claims_by_paragraphs(
        self,
        atomic_claims: List[AtomicClaim]
    ) -> List[ParagraphClaims]:
        """
        Group atomic claims by their paragraph index.

        Args:
            atomic_claims: List of atomic claims

        Returns:
            List of ParagraphClaims objects
        """
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

        return paragraphs

    async def atomize_text(
        self,
        text: str,
        use_cot: Union[CoTMode, bool] = CoTMode.NO_CHAIN,
        split_into_paragraphs: bool = False,
        use_few_shots: bool = True
    ) -> ClaimAtomizationResult:
        """
        Decompose text into atomic claims (async).

        Args:
            text: Text snippet to decompose
            use_cot: Whether to use Chain of Thought reasoning
            split_into_paragraphs: Whether to split text into paragraphs before atomization
            use_few_shots: Whether to use few-shot examples

        Returns:
            ClaimAtomizationResult with atomic claims
        """
        logger.info(f"Atomizing text ({len(text)} chars) with CoT={use_cot}, split_paragraphs={split_into_paragraphs}, use_few_shots={use_few_shots}")

        # Process text based on split mode
        if split_into_paragraphs:
            paragraphs = TextProcessor.split_text_into_paragraphs(text)
            logger.info(f"Split text into {len(paragraphs)} paragraphs")

            # Filter short paragraphs
            valid_paragraphs = [p for p in paragraphs if len(p) >= 10]
            logger.debug(f"Processing {len(valid_paragraphs)} valid paragraphs (filtered {len(paragraphs) - len(valid_paragraphs)} short ones)")

            # Process paragraphs concurrently
            claim_data_list = await self._process_paragraphs(
                valid_paragraphs,
                use_cot=use_cot,
                use_few_shots=use_few_shots
            )

            logger.info(f"Extracted {len(claim_data_list)} atomic claims from {len(valid_paragraphs)} paragraphs")
        else:
            # Process entire text as one unit
            raw_claims = await self._atomize_text_with_llm(text, use_cot=use_cot, use_few_shots=use_few_shots)

            # Format claim IDs consistently
            claim_data_list = []
            for i, claim_data in enumerate(raw_claims, 1):
                formatted_claim = ClaimFormatter.format_claim_data(
                    claim_data=claim_data,
                    paragraph_index=None,
                    paragraph_text=None,
                    source_text=text
                )
                formatted_claim['id'] = f"P1_{i}"
                claim_data_list.append(formatted_claim)

            logger.info(f"Extracted {len(claim_data_list)} atomic claims from entire text")

        # Convert to AtomicClaim objects
        atomic_claims = self._create_atomic_claims(
            claim_data_list,
            source_text=text,
            split_into_paragraphs=split_into_paragraphs
        )

        # Group by paragraphs if split
        paragraphs = self._group_claims_by_paragraphs(atomic_claims) if split_into_paragraphs else []

        # Determine execution mode
        if isinstance(use_cot, CoTMode):
            execution_mode = "chain_online" if use_cot.value in ("chain_local", "chain_online") else "no_chain"
        else:
            execution_mode = "chain_online" if use_cot else "no_chain"

        result = ClaimAtomizationResult(
            atomic_claims=atomic_claims,
            paragraphs=paragraphs,
            original_text=text,
            execution_mode=execution_mode,
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
        """Get default few-shot examples for claim atomization."""
        return CLAIM_ATOMIZER_FEW_SHOTS

    @classmethod
    def set_custom_few_shots(cls, custom_few_shots: Optional[str] = None) -> None:
        """Set custom few-shot examples for claim atomization."""
        if settings.enable_persistence:
            database = get_database()
            success = database.save_custom_few_shots("claim_atomizer", custom_few_shots)
            if success:
                cls._custom_few_shots_cache = custom_few_shots
                logger.info(f"Custom few shots saved for ClaimAtomizerAgent: {custom_few_shots is not None}")
            else:
                logger.warning("Failed to save custom few shots to persistence")
        else:
            cls._custom_few_shots_cache = custom_few_shots
            logger.info(f"Custom few shots set for ClaimAtomizerAgent (no persistence): {custom_few_shots is not None}")

    @classmethod
    def get_custom_few_shots(cls) -> Optional[str]:
        """Get currently set custom few-shot examples."""
        if settings.enable_persistence:
            database = get_database()
            few_shots = database.load_custom_few_shots("claim_atomizer")
            if isinstance(few_shots, str) or few_shots is None:
                cls._custom_few_shots_cache = few_shots
            return few_shots
        else:
            return cls._custom_few_shots_cache

    @classmethod
    def get_effective_few_shots(cls) -> str:
        """Get effective few-shot examples (custom if set, otherwise default)."""
        custom_few_shots = cls.get_custom_few_shots()
        return custom_few_shots if custom_few_shots is not None else cls.get_default_few_shots()

    def reset(self) -> None:
        """Reset the agent to initial state."""
        logger.info("Resetting ClaimAtomizerAgent state")
        logger.debug("Agent reset complete")
