"""Utility functions for claim atomization."""

import re
from typing import List, Dict, Any, Optional

from src.shared.utils.logger import get_logger

logger = get_logger(__name__)


class TextProcessor:
    """Utility class for text processing operations."""

    @staticmethod
    def split_text_into_paragraphs(text: str) -> List[str]:
        """
        Split text into paragraphs using robust paragraph detection.

        Args:
            text: The text to split

        Returns:
            List of paragraphs
        """
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

    @staticmethod
    def find_sentence_containing_claim(text: str, claim_text: str) -> str:
        """
        Find the most appropriate sentence in the text that contains the given claim.
        Prioritizes declarative sentences over questions, titles, or list items.

        Args:
            text: The full text to search in
            claim_text: The claim text to find

        Returns:
            The most appropriate sentence containing the claim, or empty string if not found
        """
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        # Filter and score sentences
        candidate_sentences = []
        claim_words = set(re.findall(r'\b\w+\b', claim_text.lower()))

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue

            sentence_lower = sentence.lower()
            sentence_words = set(re.findall(r'\b\w+\b', sentence_lower))

            # Calculate word overlap
            overlap = len(claim_words.intersection(sentence_words))
            if overlap == 0:
                continue

            # Score the sentence quality
            quality_score = TextProcessor._score_sentence_quality(sentence)

            # Calculate relevance score (overlap + quality bonus)
            relevance_score = overlap + quality_score

            candidate_sentences.append({
                'sentence': sentence,
                'overlap': overlap,
                'quality_score': quality_score,
                'relevance_score': relevance_score
            })

        if not candidate_sentences:
            return ""

        # Sort by relevance score (highest first)
        candidate_sentences.sort(key=lambda x: x['relevance_score'], reverse=True)

        # Return the best sentence if it's not too long
        best_candidate = candidate_sentences[0]
        if len(best_candidate['sentence']) <= 500:
            return best_candidate['sentence']

        return ""

    @staticmethod
    def _score_sentence_quality(sentence: str) -> float:
        """
        Score sentence quality to prefer declarative statements over questions/titles.

        Args:
            sentence: The sentence to score

        Returns:
            Quality score (higher is better)
        """
        score = 0.0
        sentence_lower = sentence.lower()

        # Prefer sentences that end with periods (declarative)
        if sentence.endswith('.'):
            score += 2.0

        # Penalize questions (sentences ending with ?)
        if sentence.endswith('?'):
            score -= 3.0

        # Penalize potential titles (short, all caps, or starting with numbers)
        if len(sentence.split()) <= 8:
            score -= 1.0

        # Penalize all caps sentences (likely titles)
        if sentence.isupper() and len(sentence) > 10:
            score -= 2.0

        # Penalize sentences starting with numbers (likely lists)
        if sentence.strip() and sentence.strip()[0].isdigit():
            score -= 1.0

        # Penalize very short sentences
        if len(sentence.split()) < 5:
            score -= 1.0

        # Bonus for sentences with common declarative words
        declarative_indicators = ['is', 'are', 'was', 'were', 'has', 'have', 'had', 'does', 'do', 'did']
        for word in declarative_indicators:
            if f' {word} ' in sentence_lower:
                score += 0.5
                break

        return score


class ClaimParser:
    """Utility class for parsing claims from LLM responses."""

    CLAIM_PATTERN = r'- CLAIM_(\d+):\s*(.+?)(?=\n- CLAIM_|\n*$)'

    @staticmethod
    def parse_claims_from_response(response_text: str, source_text: str) -> List[Dict[str, Any]]:
        """
        Parse atomic claims from LLM response text.

        Args:
            response_text: LLM response containing claims
            source_text: Original text for finding source sentences

        Returns:
            List of claim dictionaries
        """
        claims = []

        for match in re.finditer(ClaimParser.CLAIM_PATTERN, response_text, re.DOTALL):
            claim_id = match.group(1)
            claim_text = match.group(2).strip()

            # Find the sentence containing this claim
            original_sentence = TextProcessor.find_sentence_containing_claim(source_text, claim_text)
            if not original_sentence:
                # Fallback: truncate the text if it's too long
                original_sentence = source_text[:200] + "..." if len(source_text) > 200 else source_text

            claims.append({
                "id": claim_id,
                "text": claim_text,
                "original_sentence": original_sentence,
                "confidence": 0.8  # Default confidence
            })

        return claims


class ClaimFormatter:
    """Utility class for formatting claims into structured objects."""

    @staticmethod
    def format_claim_data(
        claim_data: Dict[str, Any],
        paragraph_index: Optional[int],
        paragraph_text: Optional[str],
        source_text: str
    ) -> Dict[str, Any]:
        """
        Format claim data with paragraph information and source sentence.

        Args:
            claim_data: Raw claim data dictionary
            paragraph_index: Index of the paragraph (if split)
            paragraph_text: Text of the paragraph (if split)
            source_text: Full source text for fallback

        Returns:
            Formatted claim data dictionary
        """
        claim_text = claim_data.get("text", "")
        original_sentence = claim_data.get("original_sentence")

        # Determine context text
        context_text = paragraph_text if paragraph_text else source_text

        # If no original_sentence provided, try to find it
        if not original_sentence:
            original_sentence = TextProcessor.find_sentence_containing_claim(context_text, claim_text)
            if not original_sentence:
                # Fallback: use the context text (truncated if too long)
                original_sentence = context_text[:300] + "..." if len(context_text) > 300 else context_text

        return {
            "id": claim_data.get("id", ""),
            "text": claim_text,
            "original_sentence": original_sentence,
            "confidence": claim_data.get("confidence", 0.8),
            "paragraph_index": paragraph_index,
            "paragraph_text": paragraph_text
        }

