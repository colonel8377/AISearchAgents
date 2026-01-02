"""Text file parser implementation."""

import asyncio
import logging
import chardet

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List
from .parser_config import ParserConfig
from ..constant.enums import FileType, ParseMode


from .file_parser import (
    FileParser, ParseResult, PrivacyTextSegment, FileParserError
)

logger = logging.getLogger(__name__)


class TextParser(FileParser):
    """Parser for plain text files (.txt, .csv) optimized for privacy detection."""

    SUPPORTED_EXTENSIONS = ['txt', 'csv']

    # Maximum lines to read for CSV files to prevent memory issues
    MAX_CSV_LINES = 10000

    def __init__(self, config: ParserConfig):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def supported_file_types(self) -> List[FileType]:
        return [FileType.TEXT_TXT, FileType.TEXT_CSV]

    @property
    def supported_modes(self) -> List[ParseMode]:
        return [ParseMode.LOCAL]  # Only local parsing supported

    async def parse_local(self, file_path: Path, file_type: FileType) -> ParseResult:
        """Parse text file using local processing."""
        try:
            # Run parsing in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                segments = await loop.run_in_executor(
                    executor, self._extract_privacy_segments, file_path, file_type
                )

            # Filter for privacy-relevant segments
            privacy_segments = self.filter_privacy_segments(segments)

            metadata = {
                "total_segments": len(segments),
                "privacy_segments": len(privacy_segments),
                "parser_used": "built-in",
                "file_info": self.get_file_info(file_path)
            }

            return ParseResult(
                segments=privacy_segments,
                metadata=metadata,
                success=True
            )

        except Exception as e:
            self.logger.error(f"Failed to parse text file {file_path}: {e}")
            raise FileParserError(f"Text file parsing failed: {str(e)}")

    async def parse_cloud(self, file_path: Path, file_type: FileType) -> ParseResult:
        """Parse text file using cloud services (placeholder)."""
        # Placeholder for cloud text parsing (e.g., Google Document AI)
        raise FileParserError("Cloud text parsing not implemented yet")

    def _extract_privacy_segments(self, file_path: Path, file_type: FileType) -> List[PrivacyTextSegment]:
        """Extract privacy-relevant text segments from text file."""
        segments = []

        # Detect encoding
        encoding = self._detect_encoding(file_path)

        try:
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                if file_type == FileType.TEXT_CSV:
                    content = self._parse_csv_content(f)
                else:  # TEXT_TXT
                    content = f.read()

            # Clean content
            content = self._clean_text_content(content)

            # Split content into segments (lines for text, rows for CSV)
            if file_type == FileType.TEXT_CSV:
                # For CSV, treat each row as a segment
                lines = content.split('\n')
                for line_num, line in enumerate(lines):
                    if line.strip():
                        segment = PrivacyTextSegment(
                            text=line.strip(),
                            confidence=0.9,  # CSV data often structured
                            location={
                                "line_number": line_num,
                                "source": "csv_row"
                            },
                            source_type="text",
                            privacy_indicators=[]
                        )
                        segments.append(segment)
            else:
                # For plain text, split by paragraphs or lines
                paragraphs = content.split('\n\n')
                line_num = 0
                for para in paragraphs:
                    if para.strip():
                        # Further split by lines if paragraph is long
                        lines = para.split('\n')
                        for line in lines:
                            if line.strip():
                                segment = PrivacyTextSegment(
                                    text=line.strip(),
                                    confidence=0.8,
                                    location={
                                        "line_number": line_num,
                                        "source": "text_line"
                                    },
                                    source_type="text",
                                    privacy_indicators=[]
                                )
                                segments.append(segment)
                                line_num += 1

        except Exception as e:
            self.logger.error(f"Text file reading failed: {str(e)}")

        return segments

    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding using chardet."""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB for detection
                result = chardet.detect(raw_data)

                # Use detected encoding, fallback to utf-8
                encoding = result.get('encoding', 'utf-8')
                confidence = result.get('confidence', 0)

                if confidence < 0.7:
                    logger.warning(f"Low confidence encoding detection ({confidence:.2f}) for {file_path}, using utf-8")
                    encoding = 'utf-8'

                return encoding

        except Exception as e:
            logger.warning(f"Encoding detection failed for {file_path}: {e}, using utf-8")
            return 'utf-8'

    def _parse_csv_content(self, file_handle) -> str:
        """Parse CSV content with line limit protection."""
        lines = []
        line_count = 0

        for line in file_handle:
            if line_count >= self.MAX_CSV_LINES:
                logger.warning(f"CSV file exceeds maximum line limit ({self.MAX_CSV_LINES}), truncating")
                break

            lines.append(line.rstrip('\n\r'))
            line_count += 1

        return '\n'.join(lines)

    def _clean_text_content(self, content: str) -> str:
        """Clean and normalize text content."""
        if not content:
            return ""

        # Remove excessive whitespace
        import re

        # Replace multiple spaces with single space
        content = re.sub(r' +', ' ', content)

        # Remove excessive newlines (more than 2 consecutive)
        content = re.sub(r'\n{3,}', '\n\n', content)

        # Strip leading/trailing whitespace
        content = content.strip()

        return content
