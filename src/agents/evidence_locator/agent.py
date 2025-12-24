"""Evidence Locator Agent for finding supporting evidence in text.

This agent searches through the main body text to find evidence that supports
or contradicts each atomic claim, extracting exact quotes with context.
"""

import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from ...config.settings import settings
from ...utils.llm_client import llm_manager
from ...utils.logger import get_logger
from ...utils.smart_memory import SmartMemory
from ...utils.agent_cache import cached

logger = get_logger(__name__)


@dataclass
class EvidenceQuote:
    """A quote with context from the main body text."""
    text: str
    location: str  # Paragraph number or section title
    start_pos: int  # Character position in original text
    end_pos: int  # Character position in original text


@dataclass
class ClaimEvidence:
    """Evidence for a specific claim."""
    claim_id: str
    claim_text: str
    evidence_found: bool
    quotes: List[EvidenceQuote]
    reasoning: str


@dataclass
class EvidenceLocationResult:
    """Result of evidence location for multiple claims."""
    claim_evidences: List[ClaimEvidence]
    main_body_text: str
    metadata: Optional[Dict[str, Any]] = None


class EvidenceLocatorAgent:
    """
    Evidence Locator Agent that searches through main body text to find
    supporting or contradicting evidence for each atomic claim.

    Features:
    - Exact quote extraction with context
    - Paragraph/section location tracking
    - String matching verification
    - Confidence scoring
    """

    SYSTEM_PROMPT = """You are a Literal Evidence Retriever.

Task: For each [Claim] provided, find the most relevant supporting or contradicting segments from the [Main Body].

Rules:
1. Direct Quotes Only: You must extract the EXACT sentence(s) from the Main Body. Do not paraphrase.
2. Contextual Inclusion: Extract the target sentence plus one sentence before and after to ensure semantic integrity.
3. Exhaustive Search: If the Main Body mentions the topic in multiple places, extract all relevant segments.
4. Grounding Check: If no relevant information exists, state "NO_EVIDENCE_FOUND".

Output Format:
- FOR [CLAIM_ID]:
  - QUOTE: "[Exact text from body]"
  - LOCATION: [Paragraph Number or Section Title]"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.1,
        proxy: Optional[str] = None
    ):
        """
        Initialize the EvidenceLocatorAgent.

        Args:
            model_name: Name of the LLM model to use (defaults to settings.openai_model)
            api_key: OpenAI API key or compatible API key
            api_base: Base URL for the API
            temperature: Temperature for LLM responses
            proxy: Optional HTTP proxy for API requests
        """
        model_name = model_name or settings.openai_model
        logger.info(f"Initializing EvidenceLocatorAgent: model={model_name}, temperature={temperature}")

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

        logger.debug("EvidenceLocatorAgent initialized successfully")

    def _split_into_paragraphs(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into paragraphs with position tracking.

        Args:
            text: The text to split

        Returns:
            List of (paragraph_text, start_pos, end_pos) tuples
        """
        paragraphs = []
        current_pos = 0

        # Split by double newlines (paragraph breaks)
        para_texts = re.split(r'\n\s*\n', text)

        for para_text in para_texts:
            para_text = para_text.strip()
            if para_text:
                start_pos = text.find(para_text, current_pos)
                end_pos = start_pos + len(para_text)
                paragraphs.append((para_text, start_pos, end_pos))
                current_pos = end_pos

        return paragraphs

    @cached()
    def _find_evidence_with_llm(self, claims: List[Dict[str, Any]], main_body: str) -> List[Dict[str, Any]]:
        """
        Use LLM to find evidence for each claim in the main body text.

        Args:
            claims: List of claim dictionaries with 'id' and 'text' keys
            main_body: The main body text to search

        Returns:
            List of evidence results
        """
        logger.debug(f"Finding evidence for {len(claims)} claims in text ({len(main_body)} chars)")

        # Prepare claims text
        claims_text = "\n".join([f"CLAIM_{claim['id']}: {claim['text']}" for claim in claims])

        user_message = f"""Please find evidence for the following claims in the main body text:

CLAIMS:
{claims_text}

MAIN BODY TEXT:
{main_body}

For each claim, provide evidence in the specified format, or state "NO_EVIDENCE_FOUND" if none exists."""

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=user_message)
        ]

        try:
            response = self.llm.invoke(messages)
            response_text = response.content

            # Parse evidence from response
            evidence_results = []

            for claim in claims:
                claim_id = claim['id']
                claim_pattern = rf'- FOR CLAIM_{claim_id}:(.*?)(?=\n- FOR CLAIM_|\n*$)'

                match = re.search(claim_pattern, response_text, re.DOTALL)
                if match:
                    evidence_text = match.group(1).strip()

                    # Extract quotes and locations
                    quotes = []
                    quote_pattern = r'- QUOTE:\s*"([^"]*)"'
                    location_pattern = r'- LOCATION:\s*(.+)'

                    quote_matches = re.findall(quote_pattern, evidence_text)
                    location_matches = re.findall(location_pattern, evidence_text)

                    for i, quote_text in enumerate(quote_matches):
                        location = location_matches[i] if i < len(location_matches) else "Unknown"

                        # Find position in original text
                        start_pos = main_body.find(quote_text.strip('"'))
                        end_pos = start_pos + len(quote_text.strip('"')) if start_pos != -1 else 0

                        if start_pos != -1:
                            quotes.append(EvidenceQuote(
                                text=quote_text.strip('"'),
                                location=location.strip(),
                                start_pos=start_pos,
                                end_pos=end_pos
                            ))

                    evidence_found = len(quotes) > 0
                    reasoning = "Evidence found in main body text" if evidence_found else "No evidence found"

                    evidence_results.append({
                        "claim_id": claim_id,
                        "claim_text": claim['text'],
                        "evidence_found": evidence_found,
                        "quotes": quotes,
                        "reasoning": reasoning
                    })
                else:
                    # No evidence found for this claim
                    evidence_results.append({
                        "claim_id": claim_id,
                        "claim_text": claim['text'],
                        "evidence_found": False,
                        "quotes": [],
                        "reasoning": "No evidence found in main body text"
                    })

            logger.info(f"Evidence location complete: found evidence for {sum(1 for r in evidence_results if r['evidence_found'])}/{len(claims)} claims")
            return evidence_results

        except Exception as e:
            logger.error(f"LLM evidence location failed: {e}", exc_info=True)
            # Return no evidence found for all claims
            return [{
                "claim_id": claim['id'],
                "claim_text": claim['text'],
                "evidence_found": False,
                "quotes": [],
                "reasoning": f"Evidence location failed: {str(e)}"
            } for claim in claims]

    def _verify_quote_in_text(self, quote: str, main_body: str) -> Tuple[bool, str]:
        """
        Verify that a quote actually exists in the main body text.

        This implements the "physical string match" verification mentioned in the requirements
        to prevent LLM hallucinations by ensuring quotes are grounded in the source text.

        Args:
            quote: The quote to verify
            main_body: The main body text

        Returns:
            Tuple of (is_valid, reason)
        """
        if not quote or not quote.strip():
            return False, "Empty quote"

        # Remove extra whitespace and normalize for comparison
        normalized_quote = re.sub(r'\s+', ' ', quote.strip())
        normalized_body = re.sub(r'\s+', ' ', main_body)

        # Exact match check
        if normalized_quote in normalized_body:
            return True, "Exact match found"

        # Fuzzy match: check if quote is contained within body (allowing for minor differences)
        # Remove punctuation for more lenient matching
        def normalize_text(text: str) -> str:
            # Remove punctuation and convert to lowercase
            text = re.sub(r'[^\w\s]', '', text.lower())
            return re.sub(r'\s+', ' ', text.strip())

        normalized_quote_clean = normalize_text(quote)
        normalized_body_clean = normalize_text(main_body)

        if normalized_quote_clean and normalized_quote_clean in normalized_body_clean:
            return True, "Fuzzy match found (punctuation differences)"

        # Check for partial matches (at least 80% of quote words found in sequence)
        quote_words = normalized_quote_clean.split()
        body_words = normalized_body_clean.split()

        if len(quote_words) == 0:
            return False, "Quote contains no meaningful words"

        # Find longest consecutive sequence match
        max_consecutive = 0
        for i in range(len(body_words) - len(quote_words) + 1):
            consecutive = 0
            for j in range(len(quote_words)):
                if body_words[i + j] == quote_words[j]:
                    consecutive += 1
                else:
                    break
            max_consecutive = max(max_consecutive, consecutive)

        match_ratio = max_consecutive / len(quote_words)
        if match_ratio >= 0.8:
            return True, f"High partial match ({match_ratio:.1%})"

        return False, f"Insufficient match (only {match_ratio:.1%} of words found in sequence)"

    def locate_evidence(self, claims: List[Dict[str, Any]], main_body: str, use_llm: bool = True) -> EvidenceLocationResult:
        """
        Locate evidence for multiple claims in the main body text.

        Args:
            claims: List of claim dictionaries with 'id' and 'text' keys
            main_body: The main body text to search
            use_llm: Whether to use LLM for evidence location (default: True)

        Returns:
            EvidenceLocationResult with evidence for each claim
        """
        logger.info(f"Locating evidence for {len(claims)} claims in main body text ({len(main_body)} chars)")

        if use_llm:
            # Use LLM to find evidence
            evidence_data = self._find_evidence_with_llm(claims, main_body)
        else:
            # Fallback: simple string matching
            evidence_data = []
            for claim in claims:
                claim_text = claim['text']
                # Simple keyword matching (very basic)
                claim_words = set(re.findall(r'\b\w+\b', claim_text.lower()))
                body_words = set(re.findall(r'\b\w+\b', main_body.lower()))

                overlap = claim_words.intersection(body_words)
                evidence_found = len(overlap) > 0

                evidence_data.append({
                    "claim_id": claim['id'],
                    "claim_text": claim['text'],
                    "evidence_found": evidence_found,
                    "quotes": [],
                    "reasoning": f"Keyword overlap: {len(overlap)} words" if evidence_found else "No keyword overlap found"
                })

        # Convert to ClaimEvidence objects with enhanced verification
        claim_evidences = []
        for data in evidence_data:
            # Verify quotes actually exist in text with detailed validation
            verified_quotes = []
            verification_details = []

            for quote in data['quotes']:
                is_valid, reason = self._verify_quote_in_text(quote.text, main_body)
                if is_valid:
                    verified_quotes.append(quote)
                    verification_details.append(f"✓ {reason}")
                else:
                    logger.warning(f"Quote verification failed for claim {data['claim_id']}: {reason}")
                    verification_details.append(f"✗ {reason}")

            # Update reasoning with verification details
            enhanced_reasoning = data['reasoning']
            if verification_details:
                enhanced_reasoning += f"\nVerification: {'; '.join(verification_details)}"

            # If LLM found quotes but verification failed, mark as no evidence
            evidence_found = len(verified_quotes) > 0

            claim_evidence = ClaimEvidence(
                claim_id=data['claim_id'],
                claim_text=data['claim_text'],
                evidence_found=evidence_found,
                quotes=verified_quotes,
                reasoning=enhanced_reasoning
            )
            claim_evidences.append(claim_evidence)

        result = EvidenceLocationResult(
            claim_evidences=claim_evidences,
            main_body_text=main_body,
            metadata={
                "model": self.llm.model_name if use_llm else None,
                "temperature": self.llm.temperature if use_llm else None,
                "use_llm": use_llm,
                "total_claims": len(claims),
                "claims_with_evidence": sum(1 for ce in claim_evidences if ce.evidence_found)
            }
        )

        logger.info(f"Evidence location complete: {sum(1 for ce in claim_evidences if ce.evidence_found)}/{len(claims)} claims have evidence")

        return result

    def reset(self) -> None:
        """Reset the agent to initial state."""
        logger.info("Resetting EvidenceLocatorAgent state")
        # No state to reset for this agent
        logger.debug("Agent reset complete")
