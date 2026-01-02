"""Text normalization utility for mixed Chinese-English input.

Handles tokenization issues by adding spaces between CJK and Latin characters,
and normalizing full-width characters to half-width.
"""

import re
import unicodedata


class TextNormalizer:
    """
    Text normalizer for mixed CJK (Chinese/Japanese/Korean) and Latin text.
    
    Features:
    1. Adds spaces between CJK characters and Latin/Numbers for better tokenization
    2. Normalizes full-width characters to half-width
    """
    
    # Unicode ranges for CJK characters
    # Chinese: CJK Unified Ideographs (main block)
    CJK_UNIFIED_IDEOGRAPHS = r'\u4e00-\u9fff'
    # Japanese: Hiragana, Katakana, plus Kanji from CJK blocks
    JAPANESE_HIRAGANA = r'\u3040-\u309f'
    JAPANESE_KATAKANA = r'\u30a0-\u30ff'
    # Korean: Hangul Syllables
    KOREAN_HANGUL = r'\uac00-\ud7a3'
    # Combine all CJK ranges
    CJK_PATTERN = rf'[{CJK_UNIFIED_IDEOGRAPHS}{JAPANESE_HIRAGANA}{JAPANESE_KATAKANA}{KOREAN_HANGUL}]'
    
    # Pattern for Latin characters and numbers
    LATIN_NUMBERS = r'[a-zA-Z0-9]'
    
    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize text by adding spaces between CJK and Latin/Numbers,
        and converting full-width to half-width characters.
        
        Examples:
            "我的ID是123" -> "我的 ID 是 123"
            "１２３" -> "123"
            "我的Email是test@example.com" -> "我的 Email 是 test@example.com"
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return text
        
        # Step 1: Normalize full-width to half-width
        normalized = TextNormalizer._normalize_fullwidth_to_halfwidth(text)
        
        # Step 2: Add spaces between CJK and Latin/Numbers
        normalized = TextNormalizer._add_cjk_latin_spacing(normalized)
        
        return normalized
    
    @staticmethod
    def _normalize_fullwidth_to_halfwidth(text: str) -> str:
        """
        Convert full-width characters to half-width.
        
        Handles:
        - Full-width digits (０-９) -> (0-9)
        - Full-width letters (Ａ-Ｚ, ａ-ｚ) -> (A-Z, a-z)
        - Full-width punctuation
        """
        # Use unicodedata to normalize full-width characters
        # NFKC normalization converts full-width to half-width
        normalized = unicodedata.normalize('NFKC', text)
        return normalized
    
    @staticmethod
    def _add_cjk_latin_spacing(text: str) -> str:
        """
        Add spaces between CJK characters and Latin/Numbers.
        
        Rules:
        1. CJK followed by Latin/Number: Add space after CJK
        2. Latin/Number followed by CJK: Add space after Latin/Number
        
        Examples:
            "我的ID是123" -> "我的 ID 是 123"
            "test中文" -> "test 中文"
            "中文test" -> "中文 test"
        """
        # Pattern to match boundary between CJK and Latin/Numbers
        # Case 1: CJK followed by Latin/Number -> add space after CJK
        pattern1 = rf'({TextNormalizer.CJK_PATTERN})({TextNormalizer.LATIN_NUMBERS})'
        text = re.sub(pattern1, r'\1 \2', text)
        
        # Case 2: Latin/Number followed by CJK -> add space after Latin/Number
        pattern2 = rf'({TextNormalizer.LATIN_NUMBERS})({TextNormalizer.CJK_PATTERN})'
        text = re.sub(pattern2, r'\1 \2', text)
        
        # Clean up multiple spaces (shouldn't happen, but just in case)
        text = re.sub(r' +', ' ', text)
        
        return text.strip()

