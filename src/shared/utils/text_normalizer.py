"""Text normalization utility for mixed Chinese-English input.

Handles tokenization issues by adding spaces between CJK and Latin characters,
and normalizing full-width characters to half-width.
"""

import re
import unicodedata


class TextNormalizer:
    """
    Enhanced text normalizer for mixed CJK (Chinese/Japanese/Korean) and Latin text.
    
    Features:
    1. Unicode normalization (NFC)
    2. Removes zero-width characters
    3. Converts full-width characters to half-width
    4. Adds spaces between CJK characters and Latin/Numbers for better tokenization
    5. Merges multiple spaces
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
    
    # Zero-width characters pattern (common ones used to bypass detection)
    ZERO_WIDTH_CHARS = re.compile(
        r'[\u200b\u200c\u200d\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2060\u2061\u2062\u2063\u2064\ufeff]'
    )
    
    @staticmethod
    def normalize(text: str) -> str:
        """
        Complete text normalization pipeline.
        
        Processing order:
        1. Unicode normalization (NFC)
        2. Remove zero-width characters
        3. Convert full-width to half-width
        4. Add spaces between CJK and Latin/Numbers
        5. Merge multiple spaces
        
        Examples:
            "我的WeChat是test123" -> "我的 WeChat 是 test123"
            "１２３ａｂｃ" -> "123abc"
            "测试\u200btest" -> "测试 test"  # zero-width character removed
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return text
        
        # Step 1: Unicode normalization (NFC)
        normalized = unicodedata.normalize('NFC', text)
        
        # Step 2: Remove zero-width characters
        normalized = TextNormalizer.remove_zero_width_chars(normalized)
        
        # Step 3: Convert full-width to half-width
        normalized = TextNormalizer.fullwidth_to_halfwidth(normalized)
        
        # Step 4: Add spaces between CJK and Latin/Numbers
        normalized = TextNormalizer.inject_cjk_spaces(normalized)
        
        # Step 5: Merge multiple spaces
        normalized = re.sub(r' +', ' ', normalized)
        
        return normalized.strip()
    
    @staticmethod
    def inject_cjk_spaces(text: str) -> str:
        """
        Inject spaces between CJK characters and Latin/Numbers.
        
        Rules:
        1. CJK followed by Latin/Number: Add space after CJK
        2. Latin/Number followed by CJK: Add space after Latin/Number
        
        Examples:
            "我的WeChat" -> "我的 WeChat"
            "Phone是123" -> "Phone 是 123"
            "测试test测试" -> "测试 test 测试"
            "我的ID是123" -> "我的 ID 是 123"
        
        Args:
            text: Input text
            
        Returns:
            Text with spaces injected
        """
        # Case 1: CJK followed by Latin/Number -> add space after CJK
        pattern1 = rf'({TextNormalizer.CJK_PATTERN})({TextNormalizer.LATIN_NUMBERS})'
        text = re.sub(pattern1, r'\1 \2', text)
        
        # Case 2: Latin/Number followed by CJK -> add space after Latin/Number
        pattern2 = rf'({TextNormalizer.LATIN_NUMBERS})({TextNormalizer.CJK_PATTERN})'
        text = re.sub(pattern2, r'\1 \2', text)
        
        return text
    
    @staticmethod
    def fullwidth_to_halfwidth(text: str) -> str:
        """
        Convert full-width characters to half-width.
        
        Handles:
        - Full-width digits (０-９) -> (0-9)
        - Full-width letters (Ａ-Ｚ, ａ-ｚ) -> (A-Z, a-z)
        - Full-width punctuation (：；，) -> (: ;,)
        
        Examples:
            "１２３" -> "123"
            "ａｂｃ" -> "abc"
            "：；，" -> ": ;,"
        
        Args:
            text: Input text
            
        Returns:
            Text with full-width characters converted to half-width
        """
        # NFKC normalization converts full-width to half-width
        normalized = unicodedata.normalize('NFKC', text)
        return normalized
    
    @staticmethod
    def remove_zero_width_chars(text: str) -> str:
        """
        Remove zero-width characters (may be used to bypass detection).
        
        Removes common zero-width characters:
        - Zero-width space (\u200b)
        - Zero-width non-joiner (\u200c)
        - Zero-width joiner (\u200d)
        - Left-to-right mark (\u200e)
        - Right-to-left mark (\u200f)
        - And other directional marks
        
        Examples:
            "测试\u200btest" -> "测试test"
            "text\u200c\u200dtext" -> "texttext"
        
        Args:
            text: Input text
            
        Returns:
            Text with zero-width characters removed
        """
        return TextNormalizer.ZERO_WIDTH_CHARS.sub('', text)
    
    @staticmethod
    def _normalize_fullwidth_to_halfwidth(text: str) -> str:
        """
        Legacy method name for backward compatibility.
        Use fullwidth_to_halfwidth() instead.
        """
        return TextNormalizer.fullwidth_to_halfwidth(text)
    
    @staticmethod
    def _add_cjk_latin_spacing(text: str) -> str:
        """
        Legacy method name for backward compatibility.
        Use inject_cjk_spaces() instead.
        """
        return TextNormalizer.inject_cjk_spaces(text)

