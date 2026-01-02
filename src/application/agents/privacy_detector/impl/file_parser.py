"""Smart file parser with funnel strategy for P-Guard 3.0.

Performance Funnel:
1. Cache Check: Automatic via @cached decorator (uses file content hash)
2. Processing: PDF/Doc parsing or OCR
3. Heuristic Gate: OCR quality check before Vision LLM
4. Cache Write: Automatic via @cached decorator (7-day TTL)
"""

import base64
import io
import re
from pathlib import Path
from typing import Optional, Callable, Dict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from rapidocr_onnxruntime import RapidOCR

from src.shared.cache import cached
from src.shared.constant.enums import FileType
from src.shared.utils import get_logger

logger = get_logger(__name__)

# Lazy-loaded OCR engine
_ocr_engine = None


def _lazy_load_ocr_engine():
    """Lazy load OCR engine (only on first use)."""
    global _ocr_engine
    if _ocr_engine is None:
        try:
            _ocr_engine = RapidOCR()
            logger.info("OCR engine initialized (rapidocr-onnxruntime)")
        except ImportError:
            logger.warning("rapidocr-onnxruntime not available, OCR will be disabled")
            _ocr_engine = False  # Mark as unavailable
    return _ocr_engine


class SmartFileParser:
    """
    Smart file parser with funnel strategy.
    
    Implements performance-optimized file analysis:
    - File fingerprinting (SHA256)
    - AOP caching (7-day TTL via @cached decorator)
    - OCR with lazy loading
    - Heuristic gates to avoid expensive Vision LLM calls
    
    Supported file formats:
    - PDF: .pdf
    - Word: .docx, .doc
    - Images: .png, .jpg, .jpeg, .webp, .heic, .bmp, .tiff
    - Text: .txt
    """
    
    # Heuristic constants for garbled text detection
    MIN_TEXT_LENGTH = 10
    MAX_SPECIAL_CHAR_RATIO = 0.3
    MIN_AVG_WORD_LENGTH = 2.0
    REPEATED_CHAR_THRESHOLD = 6

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        Initialize smart file parser.
        
        Args:
            llm: Optional LLM client for Vision LLM fallback
        """
        self.llm = llm
        self._handlers: Dict[FileType, Callable] = {
            FileType.PDF: self._parse_pdf,
            FileType.WORD_DOCX: self._parse_docx,
            FileType.IMAGE: self._process_image_pipeline,
            FileType.TEXT_TXT: self._parse_text_file,
        }

    def _get_file_type(self, filename: str) -> FileType:
        """Determine file type from filename extension."""
        ext = Path(filename).suffix.lower()
        if ext == '.pdf':
            return FileType.PDF
        elif ext in ['.docx']:
            return FileType.WORD_DOCX
        elif ext in ['.doc']:
            return FileType.WORD_DOC
        elif ext in ['.png', '.jpg', '.jpeg', '.webp', '.heic', '.bmp', '.tiff']:
            return FileType.IMAGE
        elif ext in ['.txt']:
            return FileType.TEXT_TXT
        return FileType.UNKNOWN

    def _is_garbled_text(self, text: str) -> bool:
        """
        Heuristic check for garbled OCR text.
        
        Returns True if text appears garbled/suspicious.
        """
        if not text or len(text.strip()) < self.MIN_TEXT_LENGTH:
            return True
        
        # Check for excessive special characters (not normal punctuation)
        special_chars = len(re.findall(r'[^\w\s\.\,\!\?\;\:\-\(\)]', text))
        if (special_chars / max(len(text), 1)) > self.MAX_SPECIAL_CHAR_RATIO:
            return True
        
        # Check for repeated characters (common OCR error)
        if re.search(rf'(.)\1{{{self.REPEATED_CHAR_THRESHOLD - 1},}}', text):
            return True
        
        # Check for very short words (OCR fragmentation)
        words = text.split()
        if words:
            avg_word_len = sum(len(w) for w in words) / len(words)
            if avg_word_len < self.MIN_AVG_WORD_LENGTH:
                return True
        
        return False
    
    async def _parse_pdf(self, file_bytes: bytes, **kwargs) -> str:
        """Parse PDF file using PyPDF2 with pdfplumber fallback."""
        try:
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text_parts = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
            return "\n".join(text_parts)
        except Exception as e:
            logger.warning(f"PyPDF2 parsing failed: {e}")
            # Try pdfplumber as fallback
            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    text_parts = [page.extract_text() for page in pdf.pages if page.extract_text()]
                    return "\n".join(text_parts)
            except Exception as e2:
                logger.error(f"PDF parsing (pdfplumber fallback) failed: {e2}")
                raise
    
    async def _parse_docx(self, file_bytes: bytes, **kwargs) -> str:
        """Parse DOCX file."""
        try:
            from docx import Document  # python-docx package
            doc = Document(io.BytesIO(file_bytes))
            return "\n".join(p.text for p in doc.paragraphs if p.text)
        except ImportError:
            logger.error("python-docx not available for DOCX parsing")
            raise RuntimeError("DOCX parsing requires python-docx package")
        except Exception as e:
            logger.error(f"DOCX parsing failed: {e}")
            raise

    async def _parse_text_file(self, file_bytes: bytes, **kwargs) -> str:
        """Parse simple text file."""
        return file_bytes.decode('utf-8', errors='ignore')

    async def _process_image_pipeline(self, file_bytes: bytes, filename: str = "") -> str:
        """
        Process image with OCR -> Quality Check -> Vision LLM fallback.
        """
        try:
            # 1. Try OCR
            text = await self._parse_image_ocr(file_bytes, filename)

            # 2. Heuristic Gate: Only call Vision LLM if OCR fails or text is garbled
            if not text or self._is_garbled_text(text):
                logger.info(f"OCR quality check failed for {filename}, using Vision LLM fallback")
                return await self._parse_image_vision_llm(file_bytes)

            logger.debug(f"OCR succeeded for {filename}")
            return text
        except Exception as e:
            logger.warning(f"OCR pipeline failed for {filename}: {e}, trying Vision LLM")
            return await self._parse_image_vision_llm(file_bytes)

    async def _parse_image_ocr(self, file_bytes: bytes, filename: str = "") -> str:
        """
        Parse image using OCR (rapidocr-onnxruntime).
        Supports formats: PNG, JPG, JPEG, WEBP, HEIC, BMP, TIFF
        """
        ocr_engine = _lazy_load_ocr_engine()
        if ocr_engine is False:
            raise RuntimeError("OCR engine not available")
        
        image_bytes = self._convert_heic_if_needed(file_bytes, filename)

        try:
            result, _ = ocr_engine(image_bytes)
            if not result:
                return ""

            # Extract text from OCR result: [[bbox, text, confidence], ...]
            return "\n".join(item[1] for item in result if len(item) >= 2 and item[1])
        except Exception as e:
            logger.error(f"OCR parsing failed: {e}")
            raise

    def _convert_heic_if_needed(self, file_bytes: bytes, filename: str) -> bytes:
        """Convert HEIC to PNG if needed."""
        if not filename or not filename.lower().endswith('.heic'):
            return file_bytes

        try:
            from PIL import Image
            img = Image.open(io.BytesIO(file_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')

            out = io.BytesIO()
            img.save(out, format='PNG')
            logger.debug("HEIC image converted to PNG for OCR")
            return out.getvalue()
        except ImportError:
            logger.warning("PIL/Pillow not available, HEIC conversion skipped")
            return file_bytes
        except Exception as e:
            logger.warning(f"HEIC conversion failed: {e}, using original bytes")
            return file_bytes

    async def _parse_image_vision_llm(self, file_bytes: bytes) -> str:
        """Parse image using Vision LLM (expensive fallback)."""
        if not self.llm:
            raise RuntimeError("LLM client not available for Vision LLM")
        
        prompt = "Extract all text from this image. Return only the text content, no explanations."
        try:
            image_base64 = base64.b64encode(file_bytes).decode('utf-8')
            message = HumanMessage(
                content=[
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": prompt}
                ]
            )
            
            response = await self.llm.ainvoke([message])
            return response.content
        except Exception as e:
            logger.error(f"Vision LLM parsing failed: {e}")
            raise
    
    @cached(ttl=604800)  # Cache for 7 days
    async def analyze_file(
        self,
        file_bytes: bytes,
        filename: str
    ) -> str:
        """
        Analyze file using funnel strategy.
        
        Funnel:
        1. Cache Check: Automatic via @cached decorator
        2. Processing: PDF/Doc parsing or OCR
        3. Heuristic Gate: OCR quality check before Vision LLM
        4. Cache Write: Automatic via @cached decorator

        Args:
            file_bytes: File content as bytes
            filename: Original filename

        Returns:
            Extracted text content
        """
        try:
            file_type = self._get_file_type(filename)
            handler = self._handlers.get(file_type)

            if not handler:
                raise ValueError(f"Unsupported file type: {file_type} ({filename})")

            # Cache is handled automatically by @cached decorator
            return await handler(file_bytes, filename=filename)

        except Exception as e:
            logger.error(f"File analysis failed for {filename}: {e}", exc_info=True)
            raise
