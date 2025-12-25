"""Compare Claims Agent for managing few-shot examples."""

from typing import Optional
from ..few_shots.compare_claims.few_shots import COMPARE_CLAIMS_FEW_SHOTS
from ..utils.logger import get_logger
from ..config.settings import settings
from ..storage import get_database

logger = get_logger(__name__)


class CompareClaimsAgent:
    """
    Agent class for managing compare_claims few-shot examples.

    This is a utility agent that doesn't perform actual comparison operations,
    but manages the few-shot examples used by the compare_claims function.
    """

    # Class variable for caching custom few shots (optional performance optimization)
    _custom_few_shots_cache: Optional[str] = None

    @staticmethod
    def get_default_few_shots() -> str:
        """
        Get the default few-shot examples for claim comparison.

        Returns:
            Default few-shot examples string
        """
        return COMPARE_CLAIMS_FEW_SHOTS

    @classmethod
    def set_custom_few_shots(cls, custom_few_shots: Optional[str] = None) -> None:
        """
        Set custom few-shot examples for claim comparison.

        Args:
            custom_few_shots: Custom few-shot examples string. If None, clears custom few shots.
        """
        try:
            if settings.enable_persistence:
                database = get_database()
                if database:
                    success = database.save_custom_few_shots("compare_claims", custom_few_shots)
                    if success:
                        cls._custom_few_shots_cache = custom_few_shots  # Update cache
                        logger.info(f"Custom few shots saved for CompareClaimsAgent: {custom_few_shots is not None}")
                    else:
                        logger.warning("Failed to save custom few shots to database, using cache only")
                        cls._custom_few_shots_cache = custom_few_shots
                else:
                    # No database available, just use cache
                    cls._custom_few_shots_cache = custom_few_shots
                    logger.info(f"Custom few shots set for CompareClaimsAgent (no persistence): {custom_few_shots is not None}")
            else:
                cls._custom_few_shots_cache = custom_few_shots
                logger.info(f"Custom few shots set for CompareClaimsAgent (no persistence): {custom_few_shots is not None}")
        except Exception as e:
            logger.error(f"Error setting custom few shots: {e}")
            # Still update cache even if database fails
            cls._custom_few_shots_cache = custom_few_shots

    @classmethod
    def get_custom_few_shots(cls) -> Optional[str]:
        """
        Get currently set custom few-shot examples.

        Returns:
            Custom few-shot examples string or None if not set
        """
        try:
            database = get_database()
            if database:
                few_shots = database.load_custom_few_shots("compare_claims")
                if isinstance(few_shots, str) or few_shots is None:
                    cls._custom_few_shots_cache = few_shots
                    return few_shots
            return cls._custom_few_shots_cache
        except Exception as e:
            logger.error(f"Error loading custom few shots: {e}")
            return cls._custom_few_shots_cache

    @classmethod
    def get_effective_few_shots(cls) -> str:
        """
        Get the effective few-shot examples (custom if set, otherwise default).

        Returns:
            Effective few-shot examples string
        """
        custom_few_shots = cls.get_custom_few_shots()
        return custom_few_shots if custom_few_shots is not None else cls.get_default_few_shots()
