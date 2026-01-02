"""PDF file parser implementation optimized for privacy detection."""

from pathlib import Path
from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
from PyPDF2 import PdfReader
import logging
import pdfplumber

from .file_parser import (
    FileParser, ParseResult, PrivacyTextSegment, FileParserError
)
from .parser_config import ParserConfig
from ..constant.enums import FileType, ParseMode


class PDFParser(FileParser):
    """Parser for PDF files optimized for privacy detection."""

    def __init__(self, config: ParserConfig):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def supported_file_types(self) -> List[FileType]:
        return [FileType.PDF]

    @property
    def supported_modes(self) -> List[ParseMode]:
        return [ParseMode.LOCAL]  # Only local parsing supported for now

    async def parse_local(self, file_path: Path, file_type: FileType) -> ParseResult:
        """Parse PDF file using local libraries."""
        try:
            # Run PDF parsing in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                segments = await loop.run_in_executor(
                    executor, self._extract_privacy_segments, file_path
                )

            # Filter for privacy-relevant segments
            privacy_segments = self.filter_privacy_segments(segments)

            metadata = {
                "total_segments": len(segments),
                "privacy_segments": len(privacy_segments),
                "parser_used": "pdfplumber/PyPDF2",
                "file_info": self.get_file_info(file_path)
            }

            return ParseResult(
                segments=privacy_segments,
                metadata=metadata,
                success=True
            )

        except Exception as e:
            self.logger.error(f"Failed to parse PDF {file_path}: {e}")
            raise FileParserError(f"PDF parsing failed: {str(e)}")

    async def parse_cloud(self, file_path: Path, file_type: FileType) -> ParseResult:
        """Parse PDF file using cloud services (placeholder)."""
        # Placeholder for cloud PDF parsing (e.g., Google Document AI, AWS Textract)
        raise FileParserError("Cloud PDF parsing not implemented yet")

    def _extract_privacy_segments(self, file_path: Path) -> List[PrivacyTextSegment]:
        """Extract text segments that may contain privacy information."""
        segments = []

        try:
            pdf = pdfplumber.open(file_path)

            for page_num, page in enumerate(pdf.pages):
                try:
                    # Extract text by layout elements
                    text_elements = self._extract_text_elements(page)

                    for element in text_elements:
                        if element['text'].strip():
                            segment = PrivacyTextSegment(
                                text=element['text'].strip(),
                                confidence=element.get('confidence', 0.8),
                                location={
                                    "page": page_num + 1,
                                    "bbox": element.get('bbox'),
                                    "source": "pdf_text"
                                },
                                source_type="pdf_text",
                                privacy_indicators=[]
                            )
                            segments.append(segment)

                except Exception as e:
                    self.logger.warning(f"Failed to extract from page {page_num + 1}: {e}")

            pdf.close()

        except ImportError:
            # Fallback to PyPDF2
            segments = self._extract_with_pypdf2(file_path)

        return segments

    def _extract_text_elements(self, page) -> List[Dict[str, Any]]:
        """Extract text elements from a pdfplumber page."""
        elements = []

        # Extract regular text
        page_text = page.extract_text()
        if page_text:
            # Split into lines/paragraphs
            lines = page_text.split('\n')
            for line_num, line in enumerate(lines):
                if line.strip():
                    elements.append({
                        'text': line,
                        'bbox': None,  # Could extract from page.chars if needed
                        'confidence': 0.9,
                        'type': 'text_line'
                    })

        # Extract text from tables (often contain structured data)
        tables = page.extract_tables()
        for table_idx, table in enumerate(tables):
            for row_idx, row in enumerate(table):
                for col_idx, cell in enumerate(row):
                    if cell and str(cell).strip():
                        elements.append({
                            'text': str(cell).strip(),
                            'bbox': None,
                            'confidence': 0.95,  # Tables often have structured data
                            'type': 'table_cell',
                            'table_info': {
                                'table_index': table_idx,
                                'row': row_idx,
                                'col': col_idx
                            }
                        })

        return elements

    def _extract_with_pypdf2(self, file_path: Path) -> List[PrivacyTextSegment]:
        """Fallback extraction using PyPDF2."""
        segments = []

        try:
            reader = PdfReader(file_path)

            for page_num in range(len(reader.pages)):
                try:
                    page = reader.pages[page_num]
                    page_text = page.extract_text()

                    if page_text:
                        # Split into rough paragraphs
                        paragraphs = page_text.split('\n\n')
                        for para in paragraphs:
                            if para.strip():
                                segment = PrivacyTextSegment(
                                    text=para.strip(),
                                    confidence=0.7,  # Lower confidence for PyPDF2
                                    location={
                                        "page": page_num + 1,
                                        "source": "pdf_text_pypdf2"
                                    },
                                    source_type="pdf_text",
                                    privacy_indicators=[]
                                )
                                segments.append(segment)

                except Exception as e:
                    self.logger.warning(f"Failed to extract from page {page_num + 1} with PyPDF2: {e}")

        except ImportError:
            self.logger.error("Neither pdfplumber nor PyPDF2 available for PDF parsing")

        return segments
