"""Model preloader utility for pre-downloading models before service starts.

This module helps pre-download GLiNER and other models to avoid downloading
during runtime, improving service startup time and first-request latency.
"""
from typing import Optional
from transformers import AutoTokenizer

from src.shared.config.settings import settings
from src.shared.utils import get_logger

_original_auto_tokenizer_from_pretrained = AutoTokenizer.from_pretrained

def _patched_auto_tokenizer_from_pretrained(*args, **kwargs):
    """Patch to force use_fast=False and prevent sentencepiece byte fallback warnings."""
    kwargs['use_fast'] = False
    if 'trust_remote_code' not in kwargs:
        kwargs['trust_remote_code'] = False
    return _original_auto_tokenizer_from_pretrained(*args, **kwargs)

AutoTokenizer.from_pretrained = _patched_auto_tokenizer_from_pretrained


logger = get_logger(__name__)


def preload_gliner_model(model_name: Optional[str] = None) -> bool:
    """
    Pre-download GLiNER model to HuggingFace cache.
    
    This function downloads the model files to HuggingFace cache directory,
    so subsequent uses can load from cache immediately without downloading.
    
    Args:
        model_name: Optional model name (defaults to settings.gliner_model_name)
        
    Returns:
        True if preload successful, False otherwise
    """
    try:
        from gliner import GLiNER
    except ImportError:
        logger.warning("GLiNER not available, skipping model preload")
        return False
    
    try:
        model_name = model_name or settings.gliner_model_name
        logger.info(f"Pre-loading GLiNER model: {model_name}")
        
        # Download model by initializing it (will cache automatically)
        # This downloads all model files to HuggingFace cache
        model = GLiNER.from_pretrained(model_name)
        
        logger.info(f"✓ GLiNER model pre-loaded successfully: {model_name}")
        logger.info("  Model files cached in HuggingFace cache directory")
        logger.info("  Subsequent loads will use cached files")
        
        # Clean up reference to free memory (model is cached)
        del model
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to pre-load GLiNER model: {e}", exc_info=True)
        return False


def preload_all_models() -> None:
    """
    Pre-load all models that should be downloaded before service starts.
    
    This includes:
    - GLiNER model (for privacy detection)
    - Future: Other models as needed
    """
    logger.info("=" * 60)
    logger.info("Pre-loading models...")
    logger.info("=" * 60)
    
    # Pre-load GLiNER model
    preload_gliner_model()
    
    logger.info("=" * 60)
    logger.info("Model pre-loading complete")
    logger.info("=" * 60)

