"""Conflict Auditor Agent for checking logical consistency between claims and evidence.

This agent compares each atomic claim against its supporting evidence to determine
if they are consistent, contradictory, or if evidence is missing.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ...utils.logger import get_logger
from ...config.settings import settings
from ...utils.agent_cache import cached
from ...memory.factory import VectorStoreFactory
from ...few_shots.conflict_auditor.few_shots import CONFLICT_AUDITOR_FEW_SHOTS
from ...storage import get_database

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
    analyses: List[ConflictAnalysis]
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

    # Class variable for caching custom few shots (optional performance optimization)
    _custom_few_shots_cache: Optional[str] = None

    SYSTEM_PROMPT = """You are an Academic Peer Reviewer specializing in Fact-Consistency Auditing.

Task: Compare the [Claim] against the [Extracted Evidence] and determine the relationship.

Definitions:
- SUPPORTED: The evidence explicitly confirms the claim.
- CONTRADICTED: The evidence explicitly negates or provides different data/facts than the claim.
- NEUTRAL/MISSING: The evidence is insufficient to prove or disprove the claim.

Logical Rigor:
1. Categorize the Conflict: If "Contradicted", specify if it is a Numerical discrepancy, Temporal error, Directional contradiction (e.g., increase vs decrease), or Scope distortion.
2. Assess Evidence Strength: Evaluate how directly the evidence supports or contradicts the claim.
3. Consider Context: Account for qualifiers, conditions, or scope limitations in both claim and evidence.
4. Provide Confidence: Rate your certainty in the assessment (0.0-1.0).

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

Chain of Thought Analysis:
1. IDENTIFY KEY ELEMENTS: Extract the core factual assertions from both the claim and evidence.
2. COMPARE FACTS: Determine if the evidence aligns with, contradicts, or is neutral toward the claim.
3. EVALUATE STRENGTH: Assess how directly and strongly the evidence supports the claim.
4. CONSIDER CONTEXT: Account for any qualifiers, conditions, or scope differences.
5. DETERMINE RELATIONSHIP: Classify as Supported, Contradicted, or Neutral based on the analysis.
6. ASSIGN CONFIDENCE: Provide a confidence score based on the clarity and strength of the relationship.

Output Format:
- CLAIM_ID: [ID]
- VERDICT: [Supported/Contradicted/Neutral]
- CONFLICT_TYPE: [Type]
- ANALYSIS: [Reasoning]"""

    SYSTEM_PROMPT_CLAIM_COMPARISON = """You are an Academic Consistency Checker specializing in comparing claims for factual alignment with source verification.

Task: Compare a SUMMARY CLAIM against multiple related URL CLAIMS and their SOURCE QUOTES to determine overall consistency.

Definitions:
- SUPPORTED: The URL claims and source quotes provide evidence that confirms or aligns with the summary claim.
- CONTRADICTED: The URL claims and source quotes provide evidence that contradicts or conflicts with the summary claim.
- NEUTRAL: The URL claims and source quotes neither clearly support nor contradict the summary claim.

Analysis Requirements:
1. Examine all provided URL claims and their corresponding source quotes
2. Cross-reference the summary claim against the actual source text (quotes)
3. Consider factual alignment, semantic meaning, and contextual implications
4. Prioritize direct quotes from the source material over interpreted claims
5. Provide probability score (0.0 = definitely contradicted, 1.0 = definitely supported)
6. Cite specific source quotes in your reasoning to prevent hallucination

Important: Base your analysis on the provided SOURCE QUOTES, not just the URL claims. The source quotes contain the actual text from the original article.

Output Format:
STATUS: [supported|contradicted|neutral]
PROBABILITY: [0.0-1.0]
REASON: [Brief explanation citing specific source quotes]"""

    SYSTEM_PROMPT_CLAIM_COMPARISON_COT = """You are an Academic Consistency Checker specializing in comparing claims for factual alignment with source verification.

Task: Compare a SUMMARY CLAIM against multiple related URL CLAIMS and their SOURCE QUOTES to determine overall consistency.

Definitions:
- SUPPORTED: The URL claims and source quotes provide evidence that confirms or aligns with the summary claim.
- CONTRADICTED: The URL claims and source quotes provide evidence that contradicts or conflicts with the summary claim.
- NEUTRAL: The URL claims and source quotes neither clearly support nor contradict the summary claim.

Chain of Thought Analysis:
1. IDENTIFY KEY ELEMENTS: Extract core factual statements from the summary claim
2. EXAMINE SOURCE MATERIAL: Carefully read all provided SOURCE QUOTES (these contain the actual article text)
3. CROSS-REFERENCE CLAIMS: Compare summary claim against each URL claim and its corresponding source quote
4. VERIFY ACCURACY: Ensure analysis is based on actual source text, not just interpreted claims
5. ASSESS OVERALL CONSISTENCY: Consider all evidence together - does the source material support the summary?
6. EVALUATE CONTEXT: Consider scope, time, perspective, and any qualifiers in the source material
7. DETERMINE RELATIONSHIP: Classify as supported/contradicted/neutral based on source evidence
8. ASSIGN PROBABILITY: Give confidence score based on strength and clarity of source alignment

Critical: Always prioritize SOURCE QUOTES over URL claims. The source quotes contain the ground truth from the original article. If source quotes don't support the summary, it should be marked as contradicted or neutral.

Output Format:
STATUS: [supported|contradicted|neutral]
PROBABILITY: [0.0-1.0]
REASON: [Detailed explanation with chain of thought, citing specific source quotes]"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.1,
        proxy: Optional[str] = None,
        execution_mode: Optional[str] = None
    ):
        """
        Initialize the ConflictAuditorAgent.

        Args:
            model_name: Name of the LLM model to use
            api_key: OpenAI API key
            api_base: Base URL for the API
            temperature: Temperature for LLM responses
            proxy: Optional HTTP proxy
            execution_mode: Execution mode for CoT
        """
        model_name = model_name or settings.openai_model
        logger.info(f"Initializing ConflictAuditorAgent: model={model_name}, temperature={temperature}")

        # Use shared HTTP client for better connection pooling and performance
        from ...utils.llm_client import llm_manager
        http_client = llm_manager.get_http_client(proxy=proxy)

        self.llm = ChatOpenAI(
            model_name=model_name,
            api_key=api_key,
            base_url=api_base,
            temperature=temperature,
            max_retries=settings.openai_max_retries,
            timeout=settings.openai_timeout,
            http_client=http_client
        )

        self.execution_mode = execution_mode or settings.default_execution_mode
        logger.debug(f"ConflictAuditorAgent initialized successfully")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception)
    )
    @cached()
    def _audit_conflicts_with_llm(self, claim_evidences: List[Dict[str, Any]], use_cot: bool = False, custom_few_shots: Optional[str] = None) -> List[Dict[str, Any]]:
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

        # Add few-shot examples
        if custom_few_shots:
            # Use explicitly provided custom few shots
            system_prompt = custom_few_shots + "\n\n" + system_prompt

        elif self._custom_few_shots:
            # Use stored custom few shots
            system_prompt = self._custom_few_shots + "\n\n" + system_prompt


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

        response = self.llm.invoke(messages)
        response_text = response.content.strip()

        # Parse the response
        analyses = []
        sections = response_text.split('CLAIM_')[1:]  # Skip the first empty part

        for section in sections:
            analysis_data = {
                "claim_id": "",
                "verdict": ConflictType.NEUTRAL_MISSING,
                "conflict_type": "unknown",
                "analysis": "Unable to parse response",
                "confidence": 0.0
            }

            lines = section.strip().split('\n')
            claim_id = lines[0].split(':')[0] if ':' in lines[0] else section[:10]

            for line in lines[1:]:
                line = line.strip()
                if line.startswith('- CLAIM_ID:'):
                    analysis_data["claim_id"] = line.split(':', 1)[1].strip()
                elif line.startswith('- VERDICT:'):
                    verdict_value = line.split(':', 1)[1].strip().lower()
                    if verdict_value == "supported":
                        analysis_data["verdict"] = ConflictType.SUPPORTED
                    elif verdict_value == "contradicted":
                        analysis_data["verdict"] = ConflictType.CONTRADICTED
                    else:
                        analysis_data["verdict"] = ConflictType.NEUTRAL_MISSING
                elif line.startswith('- CONFLICT_TYPE:'):
                    analysis_data["conflict_type"] = line.split(':', 1)[1].strip()
                elif line.startswith('- ANALYSIS:'):
                    analysis_data["analysis"] = line.split(':', 1)[1].strip()
                elif line.startswith('CONFIDENCE:') or line.startswith('- CONFIDENCE:'):
                    try:
                        analysis_data["confidence"] = float(line.split(':', 1)[1].strip())
                    except ValueError:
                        pass

            analyses.append(analysis_data)

        # Ensure we have the right number of analyses
        while len(analyses) < len(claim_evidences):
            analyses.append({
                "claim_id": f"missing_{len(analyses)}",
                "verdict": ConflictType.NEUTRAL_MISSING,
                "conflict_type": "parsing_error",
                "analysis": "Failed to parse LLM response for this claim",
                "confidence": 0.0
            })

        return analyses[:len(claim_evidences)]

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
            # Find the original claim-evidence pair
            original_ce = None
            for ce in claim_evidences:
                if str(ce.get('claim_id', '')) == str(data['claim_id']):
                    original_ce = ce
                    break

            if original_ce:
                analysis = ConflictAnalysis(
                    claim_id=original_ce['claim_id'],
                    claim_text=original_ce['claim_text'],
                    evidence_quotes=original_ce.get('evidence_quotes', []),
                    verdict=data['verdict'],
                    conflict_type=data['conflict_type'],
                    analysis=data['analysis'],
                    confidence=data['confidence']
                )
                conflict_analyses.append(analysis)

        # Calculate summary statistics
        summary_stats = {}
        for analysis in conflict_analyses:
            conflict_type = analysis.conflict_type
            summary_stats[conflict_type] = summary_stats.get(conflict_type, 0) + 1

        result = ConflictAuditResult(
            analyses=conflict_analyses,
            summary_stats=summary_stats,
            execution_mode="cot" if use_cot else "direct",
            metadata={"total_claims": len(claim_evidences)}
        )

        return result

    def compare_claims(
        self,
        summary_claims: List[Dict[str, Any]],
        url_claims: List[Dict[str, Any]],
        use_cot: bool = False,
        custom_few_shots: Optional[str] = None,
        use_embedding_similarity: bool = True,
        similarity_threshold: float = 0.3,
        max_related_claims: int = 3
    ) -> Dict[str, Any]:
        """
        Compare summary claims against URL claims to check consistency with source verification.

        Args:
            summary_claims: List of summary claims with keys: id, text, confidence, original_sentence
            url_claims: List of URL claims with keys: id, text, confidence, original_sentence
            use_cot: Whether to use Chain of Thought reasoning
            custom_few_shots: Optional custom few-shot examples
            use_embedding_similarity: Whether to use embedding-based similarity for claim matching (more accurate but requires API calls)
            similarity_threshold: Minimum similarity score required for LLM analysis (0.0-1.0). Claims below this threshold are marked as neutral.
            max_related_claims: Maximum number of related URL claims to analyze per summary claim (default: 3)

        Returns:
            Dict with comparison results for each summary claim, including source quotes for verification
        """
        logger.info(f"Comparing {len(summary_claims)} summary claims against {len(url_claims)} URL claims (embedding_similarity={use_embedding_similarity}, threshold={similarity_threshold}, max_related={max_related_claims})")

        # Select system prompt based on CoT mode
        if use_cot:
            system_prompt = self.SYSTEM_PROMPT_CLAIM_COMPARISON_COT
        else:
            system_prompt = self.SYSTEM_PROMPT_CLAIM_COMPARISON

        # Add few-shot examples
        if custom_few_shots:
            system_prompt = custom_few_shots + "\n\n" + system_prompt
        elif self._custom_few_shots:
            system_prompt = self._custom_few_shots + "\n\n" + system_prompt

        comparisons = []

        # Pre-compute embeddings if using embedding similarity
        summary_embeddings = None
        url_embeddings = None
        embedding_computed = False

        if use_embedding_similarity and summary_claims and url_claims:
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                import numpy as np

                embeddings = VectorStoreFactory.create_embeddings()

                # Get all claim texts
                all_texts = [claim['text'] for claim in summary_claims + url_claims]
                all_embeddings = embeddings.embed_documents(all_texts)

                # Split embeddings back
                summary_embeddings = np.array(all_embeddings[:len(summary_claims)])
                url_embeddings = np.array(all_embeddings[len(summary_claims):])
                embedding_computed = True

                logger.info(f"Computed embeddings for {len(all_texts)} claims")

            except Exception as e:
                logger.warning(f"Failed to compute embeddings, falling back to text similarity: {e}")
                use_embedding_similarity = False

        for i, summary_claim in enumerate(summary_claims):
            # Find the most relevant URL claims (top-k)
            related_claims = []

            if use_embedding_similarity and embedding_computed:
                # Use embedding similarity
                summary_embedding = summary_embeddings[i].reshape(1, -1)
                similarities = cosine_similarity(summary_embedding, url_embeddings)[0]

                # Get top-k similar claims
                top_indices = np.argsort(similarities)[::-1][:max_related_claims]
                for idx in top_indices:
                    if similarities[idx] >= similarity_threshold:
                        related_claims.append((url_claims[idx], similarities[idx]))

                # If no claims found above threshold, try entity-based matching as fallback
                if not related_claims and similarities.max() < similarity_threshold:
                    logger.info(f"No claims above embedding threshold {similarity_threshold}, trying entity-based matching for claim {i}")
                    entity_related_claims = self._find_related_claims_by_entities(
                        summary_claim, url_claims, similarity_threshold
                    )
                    if entity_related_claims:
                        related_claims.extend(entity_related_claims)
                        logger.info(f"Entity-based matching found {len(entity_related_claims)} related claims for claim {i}")

            else:
                # Fallback to simple text similarity
                similarity_scores = []
                for url_claim in url_claims:
                    # Simple similarity based on common words (Jaccard similarity)
                    summary_words = set(summary_claim['text'].lower().split())
                    url_words = set(url_claim['text'].lower().split())
                    if summary_words and url_words:
                        similarity = len(summary_words.intersection(url_words)) / len(summary_words.union(url_words))
                    else:
                        similarity = 0.0
                    similarity_scores.append((url_claim, similarity))

                # Sort by similarity and take top-k
                similarity_scores.sort(key=lambda x: x[1], reverse=True)
                related_claims = [(claim, score) for claim, score in similarity_scores[:max_related_claims] if score >= similarity_threshold]

            # Build related URL claims list for response
            related_url_claims = []
            source_quotes = []

            for url_claim, similarity in related_claims:
                related_url_claims.append({
                    "id": url_claim["id"],
                    "text": url_claim["text"],
                    "confidence": url_claim.get("confidence", 0.8),
                    "original_sentence": url_claim.get("original_sentence", ""),
                    "similarity_score": similarity,
                    "support_status": None,  # Will be filled after LLM analysis
                    "support_probability": None,  # Will be filled after LLM analysis
                    "support_reason": None  # Will be filled after LLM analysis
                })
                if 'original_sentence' in url_claim:
                    source_quotes.append(url_claim['original_sentence'])

            if related_claims:
                # Prepare all related URL claims and their source quotes for LLM analysis
                url_claims_text = ""
                for j, (url_claim, similarity) in enumerate(related_claims):
                    url_claims_text += f"\nURL CLAIM {j+1}: {url_claim['text']}"
                    if 'original_sentence' in url_claim:
                        url_claims_text += f"\nSOURCE QUOTE {j+1}: \"{url_claim['original_sentence']}\""

                # Use LLM to analyze the summary claim against all related URL claims
                user_message = f"""Please analyze this summary claim against the related URL claims and source quotes:

SUMMARY CLAIM: {summary_claim['text']}
{url_claims_text}

Based on the source quotes provided, determine if the summary claim is supported, contradicted, or neutral.
Consider all the URL claims and their corresponding source quotes together.

Provide your analysis in the exact format:
STATUS: [supported|contradicted|neutral]
PROBABILITY: [0.0-1.0]
REASON: [Brief explanation with reference to specific source quotes]"""

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_message)
                ]

                response = self.llm.invoke(messages)
                response_text = response.content.strip()

                # Parse the response
                status = "neutral"
                probability = 0.5
                reason = "Unable to determine"

                for line in response_text.split('\n'):
                    line = line.strip()
                    if line.startswith('STATUS:'):
                        status_value = line.split(':', 1)[1].strip().lower()
                        if status_value in ['supported', 'contradicted', 'neutral']:
                            status = status_value
                    elif line.startswith('PROBABILITY:'):
                        try:
                            prob_value = float(line.split(':', 1)[1].strip())
                            probability = max(0.0, min(1.0, prob_value))
                        except ValueError:
                            pass
                    elif line.startswith('REASON:'):
                        reason = line.split(':', 1)[1].strip()

                # Update related_url_claims with support information based on overall analysis
                for i, claim_data in enumerate(related_url_claims):
                    claim_data["support_status"] = status  # Use overall status for each related claim
                    claim_data["support_probability"] = probability
                    claim_data["support_reason"] = reason

                # Validate that we have proper source quotes for analysis
                validation_info = {
                    "has_source_quotes": len(source_quotes) > 0,
                    "all_related_claims_have_quotes": all(c.get("original_sentence") for c in related_url_claims),
                    "similarity_scores_valid": all(0.0 <= c["similarity_score"] <= 1.0 for c in related_url_claims),
                    "analysis_complete": True
                }

                comparison = {
                    "summary_claim": {
                        "id": summary_claim["id"],
                        "text": summary_claim["text"],
                        "confidence": summary_claim.get("confidence", 0.8),
                        "original_sentence": summary_claim.get("original_sentence", ""),
                        "validation": {
                            "has_text": bool(summary_claim.get("text")),
                            "has_id": bool(summary_claim.get("id"))
                        }
                    },
                    "related_url_claims": related_url_claims,
                    "source_quotes": source_quotes,
                    "status": status,
                    "probability": probability if isinstance(probability, (int, float)) and probability is not None else 0.0,
                    "reason": reason,
                    "llm_analysis_performed": True,
                    "total_related_claims": len(related_claims),
                    "validation": validation_info,
                    "statistics": {
                        "max_similarity_score": max(similarity for _, similarity in related_claims) if related_claims else 0.0,
                        "min_similarity_score": min(similarity for _, similarity in related_claims) if related_claims else 0.0,
                        "avg_similarity_score": sum(similarity for _, similarity in related_claims) / len(related_claims) if related_claims else 0.0,
                        "claims_above_threshold": len(related_claims),
                        "source_quotes_count": len(source_quotes),
                        "support_distribution": {
                            "supported_claims": len([c for c in related_url_claims if c.get("support_status") == "supported"]),
                            "contradicted_claims": len([c for c in related_url_claims if c.get("support_status") == "contradicted"]),
                            "neutral_claims": len([c for c in related_url_claims if c.get("support_status") == "neutral"])
                        }
                    }
                }
            else:
                # No related URL claims found above threshold - skip LLM analysis entirely
                comparison = {
                    "summary_claim": {
                        "id": summary_claim["id"],
                        "text": summary_claim["text"],
                        "confidence": summary_claim.get("confidence", 0.8),
                        "original_sentence": summary_claim.get("original_sentence", ""),
                        "validation": {
                            "has_text": bool(summary_claim.get("text")),
                            "has_id": bool(summary_claim.get("id"))
                        }
                    },
                    "related_url_claims": [],
                    "source_quotes": [],
                    "status": "not_supported",
                    "probability": 0.0,  # Always 0.0 for not_supported status
                    "reason": f"No URL claims found with sufficient similarity (threshold: {similarity_threshold})",
                    "llm_analysis_performed": False,
                    "total_related_claims": 0,
                    "validation": {
                        "has_source_quotes": False,
                        "all_related_claims_have_quotes": False,
                        "similarity_scores_valid": True,  # No scores to validate
                        "analysis_complete": True,  # Analysis was attempted but no matches found
                        "no_matches_found": True
                    },
                    "statistics": {
                        "max_similarity_score": 0.0,
                        "min_similarity_score": 0.0,
                        "avg_similarity_score": 0.0,
                        "claims_above_threshold": 0,
                        "source_quotes_count": 0,
                        "support_distribution": {
                            "supported_claims": 0,
                            "contradicted_claims": 0,
                            "neutral_claims": 0
                        }
                    }
                }

            comparisons.append(comparison)

        # Calculate statistics
        llm_analysis_count = sum(1 for c in comparisons if c.get("llm_analysis_performed", False))
        total_related_claims = sum(c.get("total_related_claims", 0) for c in comparisons)

        # Calculate per-summary statistics
        all_similarity_scores = []
        for c in comparisons:
            # Collect all individual similarity scores from related claims
            related_claims = c.get("related_url_claims", [])
            if related_claims:
                for claim in related_claims:
                    all_similarity_scores.append(claim.get("similarity_score", 0))

        return {
            "comparisons": comparisons,
            "summary_statistics": {
                "total_summary_claims": len(summary_claims),
                "total_url_claims": len(url_claims),
                "claims_with_related_matches": sum(1 for c in comparisons if c.get("total_related_claims", 0) > 0),
                "claims_without_matches": sum(1 for c in comparisons if c.get("total_related_claims", 0) == 0),
                "llm_analysis_performed": llm_analysis_count,
                "llm_analysis_skipped": len(comparisons) - llm_analysis_count,
                "total_related_claims_analyzed": total_related_claims,
                "average_related_claims_per_summary": total_related_claims / len(comparisons) if comparisons else 0.0,
                "overall_similarity_stats": {
                    "max_similarity_found": max(all_similarity_scores) if all_similarity_scores else 0.0,
                    "min_similarity_found": min(all_similarity_scores) if all_similarity_scores else 0.0,
                    "avg_similarity_across_matches": sum(all_similarity_scores) / len(all_similarity_scores) if all_similarity_scores else 0.0
                }
            },
            "consistency_results": {
                "supported_count": sum(1 for c in comparisons if c["status"] == "supported"),
                "contradicted_count": sum(1 for c in comparisons if c["status"] == "contradicted"),
                "neutral_count": sum(1 for c in comparisons if c["status"] == "neutral"),
                "not_supported_count": sum(1 for c in comparisons if c["status"] == "not_supported"),
                "consistency_score": sum(c.get("probability", 0.0) for c in comparisons if c["status"] == "supported") / len(comparisons) if comparisons else 0.0
            },
            "processing_config": {
                "embedding_similarity_used": use_embedding_similarity,
                "similarity_threshold": similarity_threshold,
                "max_related_claims_per_summary": max_related_claims
            }
        }

    def _find_related_claims_by_entities(self, summary_claim: Dict[str, Any], url_claims: List[Dict[str, Any]], similarity_threshold: float) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find related URL claims by entity matching when embedding similarity fails.

        Args:
            summary_claim: The summary claim to match
            url_claims: List of URL claims to search through
            similarity_threshold: Minimum similarity threshold for fallback

        Returns:
            List of (url_claim, similarity_score) tuples found by entity matching
        """
        try:
            # Extract entities from summary claim using LLM
            entity_extraction_prompt = f"""Extract key entities (people, organizations, locations, dates, numbers, specific terms) from the following claim.
Return them as a comma-separated list. If no clear entities, return "NONE".

CLAIM: {summary_claim['text']}

ENTITIES:"""

            entity_response = self.llm.invoke([SystemMessage(content="You are an entity extraction assistant."), HumanMessage(content=entity_extraction_prompt)])
            summary_entities = [e.strip() for e in entity_response.content.strip().split(',') if e.strip() and e.strip() != "NONE"]

            if not summary_entities:
                return []

            related_claims = []
            for url_claim in url_claims:
                # Extract entities from URL claim
                url_entity_prompt = f"""Extract key entities (people, organizations, locations, dates, numbers, specific terms) from the following claim.
Return them as a comma-separated list. If no clear entities, return "NONE".

CLAIM: {url_claim['text']}

ENTITIES:"""

                url_entity_response = self.llm.invoke([SystemMessage(content="You are an entity extraction assistant."), HumanMessage(content=url_entity_prompt)])
                url_entities = [e.strip() for e in url_entity_response.content.strip().split(',') if e.strip() and e.strip() != "NONE"]

                if url_entities:
                    # Calculate entity overlap similarity
                    common_entities = set(summary_entities).intersection(set(url_entities))
                    if common_entities:
                        # Use Jaccard similarity for entity overlap
                        entity_similarity = len(common_entities) / len(set(summary_entities).union(set(url_entities)))

                        # Boost similarity if there are significant entity matches
                        if entity_similarity >= 0.3:  # At least 30% entity overlap
                            boosted_similarity = min(0.9, similarity_threshold + entity_similarity * 0.5)
                            related_claims.append((url_claim, boosted_similarity))

            return related_claims

        except Exception as e:
            logger.warning(f"Entity-based matching failed: {e}")
            return []

    @staticmethod
    def get_default_few_shots() -> str:
        """
        Get default few-shot examples for conflict auditing.

        Returns:
            String containing few-shot examples
        """
        return CONFLICT_AUDITOR_FEW_SHOTS

    @classmethod
    def set_custom_few_shots(cls, custom_few_shots: Optional[str] = None) -> None:
        """
        Set custom few-shot examples for conflict auditing.

        Args:
            custom_few_shots: Custom few-shot examples string. If None, clears custom few shots.
        """
        if settings.enable_persistence:
            database = get_database()
            success = database.save_custom_few_shots("conflict_auditor", custom_few_shots)
            if success:
                cls._custom_few_shots_cache = custom_few_shots  # Update cache
                logger.info(f"Custom few shots saved for ConflictAuditorAgent: {custom_few_shots is not None}")
            else:
                logger.warning("Failed to save custom few shots to database")
        else:
            cls._custom_few_shots_cache = custom_few_shots
            logger.info(f"Custom few shots set for ConflictAuditorAgent (no persistence): {custom_few_shots is not None}")

    @classmethod
    def get_custom_few_shots(cls) -> Optional[str]:
        """
        Get currently set custom few-shot examples.

        Returns:
            Custom few-shot examples string or None if not set
        """
        if settings.enable_persistence:
            database = get_database()
            few_shots = database.load_custom_few_shots("conflict_auditor")
            # Update cache
            if isinstance(few_shots, str) or few_shots is None:
                cls._custom_few_shots_cache = few_shots
            return few_shots
        else:
            return cls._custom_few_shots_cache

    @classmethod
    def get_effective_few_shots(cls) -> str:
        """
        Get effective few-shot examples (custom if set, otherwise default).

        Returns:
            Effective few-shot examples string
        """
        custom_few_shots = cls.get_custom_few_shots()
        return custom_few_shots if custom_few_shots is not None else cls.get_default_few_shots()

    def reset(self) -> None:
        """Reset the agent to initial state."""
        logger.info("Resetting ConflictAuditorAgent state")
        # No state to reset for this agent
        logger.debug("Agent reset complete")
