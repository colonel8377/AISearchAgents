"""Utility classes and functions for consistency checking."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class ClaimInfo:
    """Structured claim information."""
    id: str
    text: str
    original_sentence: Optional[str] = None
    paragraph_index: Optional[int] = None
    confidence: float = 0.8


class ClaimFormatter:
    """Utility class for formatting claims into different representations."""

    @staticmethod
    def to_dict(claim: Any) -> Dict[str, Any]:
        """
        Convert a claim object to dictionary format.

        Args:
            claim: Claim object with attributes: id, text, original_sentence, paragraph_index, confidence

        Returns:
            Dictionary representation of the claim
        """
        return {
            "id": claim.id,
            "text": claim.text,
            "original_sentence": getattr(claim, 'original_sentence', ''),
            "paragraph_index": getattr(claim, 'paragraph_index', None),
            "confidence": getattr(claim, 'confidence', 0.8)
        }

    @staticmethod
    def to_simple_dict(claim: Any) -> Dict[str, Any]:
        """
        Convert a claim object to simple dictionary (id and text only).

        Args:
            claim: Claim object with attributes: id, text

        Returns:
            Simple dictionary with id and text
        """
        return {
            "id": claim.id,
            "text": claim.text
        }

    @staticmethod
    def format_claim_pair(
        summary_claim: Any,
        url_claim: Any,
        similarity_score: float,
        relationship: str,
        confidence: float,
        reason: str
    ) -> Dict[str, Any]:
        """
        Format a claim pair comparison result.

        Args:
            summary_claim: Summary claim object
            url_claim: URL claim object
            similarity_score: Similarity score between claims
            relationship: Relationship status (supported/contradicted/neutral/error)
            confidence: Confidence score
            reason: Reason for the relationship

        Returns:
            Formatted comparison result dictionary
        """
        return {
            "summary_claim": ClaimFormatter.to_dict(summary_claim),
            "url_claim": {
                "id": url_claim.id,
                "text": url_claim.text,
                "original_sentence": getattr(url_claim, 'original_sentence', ''),
                "paragraph_index": getattr(url_claim, 'paragraph_index', None)
            },
            "similarity_score": float(similarity_score),
            "relationship": relationship,
            "confidence": float(confidence),
            "reason": reason
        }

    @staticmethod
    def format_paragraph_info(paragraph: Any) -> Dict[str, Any]:
        """
        Format paragraph information with claims.

        Args:
            paragraph: Paragraph object with attributes: paragraph_index, paragraph_text, atomic_claims

        Returns:
            Formatted paragraph dictionary
        """
        return {
            "paragraph_index": paragraph.paragraph_index,
            "paragraph_text": paragraph.paragraph_text,
            "claim_count": len(paragraph.atomic_claims),
            "claims": [
                {
                    "id": c.id,
                    "text": c.text,
                    "original_sentence": getattr(c, 'original_sentence', ''),
                    "confidence": c.confidence
                } for c in paragraph.atomic_claims
            ]
        }


class StatisticsCalculator:
    """Utility class for calculating statistics from consistency analysis results."""

    @staticmethod
    def calculate_comparison_statistics(
        comparisons: List[Dict[str, Any]],
        similarity_threshold: float,
        summary_atomization: Any,
        url_atomization: Any
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics from claim comparisons.

        Args:
            comparisons: List of comparison results (already filtered by threshold)
            similarity_threshold: Threshold used for filtering
            summary_atomization: Summary atomization result
            url_atomization: URL atomization result

        Returns:
            Dictionary with statistics
        """
        total_summary_claims = len(summary_atomization.atomic_claims)
        total_url_claims = len(url_atomization.atomic_claims)
        total_possible_pairs = total_summary_claims * total_url_claims
        
        # Verify all comparisons meet threshold (safety check)
        valid_comparisons = [
            c for c in comparisons 
            if c.get("similarity_score", 0.0) >= similarity_threshold
        ]
        
        stats = {
            "total_summary_claims": total_summary_claims,
            "total_url_claims": total_url_claims,
            "total_possible_pairs": total_possible_pairs,
            "total_comparisons": len(valid_comparisons),
            "similarity_threshold_used": similarity_threshold,
            "pairs_above_threshold": len(valid_comparisons),
            "pairs_below_threshold": total_possible_pairs - len(valid_comparisons),
            "filtering_rate": (
                (total_possible_pairs - len(valid_comparisons)) / total_possible_pairs 
                if total_possible_pairs > 0 else 0.0
            ),
            "supported_count": sum(1 for c in valid_comparisons if c.get("relationship") == "supported"),
            "contradicted_count": sum(1 for c in valid_comparisons if c.get("relationship") == "contradicted"),
            "neutral_count": sum(1 for c in valid_comparisons if c.get("relationship") == "neutral"),
            "error_count": sum(1 for c in valid_comparisons if c.get("relationship") == "error"),
            "avg_similarity_score": (
                sum(c.get("similarity_score", 0.0) for c in valid_comparisons) / len(valid_comparisons)
                if valid_comparisons else 0.0
            ),
            "min_similarity_score": (
                min(c.get("similarity_score", 0.0) for c in valid_comparisons)
                if valid_comparisons else 0.0
            ),
            "max_similarity_score": (
                max(c.get("similarity_score", 0.0) for c in valid_comparisons)
                if valid_comparisons else 0.0
            ),
            "avg_confidence": (
                sum(c.get("confidence", 0.0) for c in valid_comparisons) / len(valid_comparisons)
                if valid_comparisons else 0.0
            )
        }
        
        return stats

