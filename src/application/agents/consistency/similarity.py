"""Similarity calculation module for consistency checking."""

from typing import List, Any

import numpy as np
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

from src.shared.config.settings import settings
from src.shared.llm.llm_manager import llm_manager
from src.shared.utils import get_logger

logger = get_logger(__name__)


class SimilarityCalculator:
    """Utility class for calculating similarity between claims."""

    @staticmethod
    async def calculate_similarity_matrix(
        summary_claims: List[Any],
        url_claims: List[Any]
    ) -> np.ndarray:
        """
        Calculate similarity matrix between summary and URL claims using embeddings.

        Args:
            summary_claims: List of summary claim objects with .text attribute
            url_claims: List of URL claim objects with .text attribute

        Returns:
            Similarity matrix as numpy array
        """
        summary_texts = [c.text for c in summary_claims]
        url_texts = [c.text for c in url_claims]

        if not summary_texts or not url_texts:
            logger.warning("No claims found in summary or URL content.")
            return np.zeros((len(summary_texts), len(url_texts)))

        try:
            http_client = llm_manager.get_http_client()
            embeddings = OpenAIEmbeddings(
                api_key=settings.openai_api_key,
                base_url=settings.openai_api_base,
                model=settings.embedding_model,
                http_client=http_client
            )

            all_texts = summary_texts + url_texts

            # Get embeddings (async)
            if hasattr(embeddings, 'aembed_documents'):
                all_embeddings = await embeddings.aembed_documents(all_texts)
            else:
                # Fallback: use sync version if async not available
                all_embeddings = embeddings.embed_documents(all_texts)

            # Verify embeddings were retrieved correctly
            if len(all_embeddings) != len(all_texts):
                raise ValueError(
                    f"Embedding count mismatch: expected {len(all_texts)} embeddings, "
                    f"got {len(all_embeddings)}"
                )

            # Split embeddings: first len(summary_texts) are summary, rest are URL
            summary_embeddings = np.array(all_embeddings[:len(summary_texts)])
            url_embeddings = np.array(all_embeddings[len(summary_texts):])

            # Verify matrix dimensions
            if summary_embeddings.shape[0] != len(summary_texts):
                raise ValueError(
                    f"Summary embeddings shape mismatch: expected {len(summary_texts)} rows, "
                    f"got {summary_embeddings.shape[0]}"
                )
            if url_embeddings.shape[0] != len(url_texts):
                raise ValueError(
                    f"URL embeddings shape mismatch: expected {len(url_texts)} rows, "
                    f"got {url_embeddings.shape[0]}"
                )


            similarity_matrix = cosine_similarity(summary_embeddings, url_embeddings)
            
            # Verify output matrix shape
            expected_shape = (len(summary_texts), len(url_texts))
            if similarity_matrix.shape != expected_shape:
                raise ValueError(
                    f"Similarity matrix shape mismatch: expected {expected_shape}, "
                    f"got {similarity_matrix.shape}"
                )

            logger.info(
                f"Calculated similarity matrix: shape={similarity_matrix.shape}, "
                f"min={similarity_matrix.min():.3f}, max={similarity_matrix.max():.3f}, "
                f"mean={similarity_matrix.mean():.3f}"
            )
            return similarity_matrix

        except Exception as e:
            logger.warning(f"Failed to calculate embeddings, using fallback similarity: {e}")
            return SimilarityCalculator._fallback_similarity(summary_claims, url_claims)

    @staticmethod
    def _fallback_similarity(
        summary_claims: List[Any],
        url_claims: List[Any]
    ) -> np.ndarray:
        """
        Fallback similarity calculation using text overlap (Jaccard similarity).

        Args:
            summary_claims: List of summary claim objects
            url_claims: List of URL claim objects

        Returns:
            Similarity matrix as numpy array
        """
        similarity_matrix = np.zeros((len(summary_claims), len(url_claims)))
        for i, summary_claim in enumerate(summary_claims):
            summary_words = set(summary_claim.text.lower().split())
            for j, url_claim in enumerate(url_claims):
                url_words = set(url_claim.text.lower().split())
                overlap = len(summary_words.intersection(url_words))
                total_words = len(summary_words.union(url_words))
                similarity_matrix[i, j] = overlap / total_words if total_words > 0 else 0.0
        return similarity_matrix

