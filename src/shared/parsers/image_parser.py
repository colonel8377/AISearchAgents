"""Image file parser implementation with OCR support for privacy detection."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Any

from .file_parser import (
    FileParser, ParseResult, PrivacyTextSegment, FileParserError
)
from .parser_config import ParserConfig
from ..constant.enums import FileType, ParseMode

logger = logging.getLogger(__name__)


class ImageParser(FileParser):
    """Parser for image files using OCR, optimized for privacy detection."""

    # OCR Configuration
    OCR_CONFIDENCE_THRESHOLD = 30
    OCR_CONFIG = '--oem 3 --psm 6'
    MIN_IMAGE_DIMENSION = 300

    def __init__(self, config: ParserConfig):
        super().__init__(config)

    @property
    def supported_file_types(self) -> List[FileType]:
        return [
            FileType.IMAGE_JPG,
            FileType.IMAGE_JPEG,
            FileType.IMAGE_PNG,
            FileType.IMAGE_BMP,
            FileType.IMAGE_TIFF
        ]

    @property
    def supported_modes(self) -> List[ParseMode]:
        return [ParseMode.LOCAL]  # Only local OCR supported for now

    async def parse_local(self, file_path: Path, file_type: FileType) -> ParseResult:
        """Parse image file using local OCR."""
        try:
            # Run OCR in thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as executor:
                segments = await loop.run_in_executor(
                    executor, self._extract_ocr_segments, file_path
                )

            # Filter for privacy-relevant segments
            privacy_segments = self.filter_privacy_segments(segments)

            metadata = {
                "total_segments": len(segments),
                "privacy_segments": len(privacy_segments),
                "parser_used": "pytesseract",
                "ocr_engine": "tesseract",
                "file_info": self.get_file_info(file_path)
            }

            return ParseResult(
                segments=privacy_segments,
                metadata=metadata,
                success=True
            )

        except Exception as e:
            logger.error(f"Failed to parse image {file_path}: {e}", exc_info=True)
            raise FileParserError(f"Image OCR parsing failed: {str(e)}")

    async def parse_cloud(self, file_path: Path, file_type: FileType) -> ParseResult:
        """Parse image file using cloud OCR services (placeholder)."""
        # Placeholder for cloud OCR services (e.g., Google Vision API, AWS Rekognition)
        raise FileParserError("Cloud OCR parsing not implemented yet")

    def _extract_ocr_segments(self, file_path: Path) -> List[PrivacyTextSegment]:
        """Extract text segments from image using OCR."""
        segments = []

        try:
            import pytesseract
            from PIL import Image
        except ImportError as e:
            logger.error(f"OCR libraries not available. Install with: pip install pytesseract pillow: {e}")
            return segments

        try:
            # Open and preprocess image
            image = Image.open(file_path)
            processed_image = self._preprocess_image(image)

            # Get detailed OCR data
            ocr_data = pytesseract.image_to_data(
                processed_image,
                output_type=pytesseract.Output.DICT,
                config=self.OCR_CONFIG
            )

            # Extract text blocks with position information
            n_boxes = len(ocr_data['text'])
            for i in range(n_boxes):
                text = ocr_data['text'][i].strip()

                # Convert confidence to int safely
                try:
                    conf_val = ocr_data['conf'][i]
                    confidence = int(float(conf_val)) if conf_val != '-1' else 0
                except (ValueError, TypeError):
                    confidence = 0

                if text and confidence > self.OCR_CONFIDENCE_THRESHOLD:
                    # Get bounding box
                    x, y, w, h = (ocr_data['left'][i], ocr_data['top'][i],
                                 ocr_data['width'][i], ocr_data['height'][i])

                    segment = PrivacyTextSegment(
                        text=text,
                        confidence=confidence / 100.0,  # Convert to 0-1 scale
                        location={
                            "bbox": [x, y, x + w, y + h],
                            "source": "ocr",
                            "page": 1  # Images are single page
                        },
                        source_type="ocr",
                        privacy_indicators=[]
                    )
                    segments.append(segment)

            # Also try to extract larger text blocks
            full_text = pytesseract.image_to_string(processed_image, config=self.OCR_CONFIG)
            if full_text.strip():
                # Split into lines and create segments
                lines = full_text.split('\n')
                for line_num, line in enumerate(lines):
                    if line.strip():
                        segment = PrivacyTextSegment(
                            text=line.strip(),
                            confidence=0.8,  # Default confidence for line-based extraction
                            location={
                                "line_number": line_num,
                                "source": "ocr_lines",
                                "page": 1
                            },
                            source_type="ocr",
                            privacy_indicators=[]
                        )
                        segments.append(segment)

        except Exception as e:
            logger.error(f"OCR processing failed for {file_path}: {e}", exc_info=True)

        return segments

    def _preprocess_image(self, image: Any) -> Any:
        """
        Preprocess image for better OCR results.

        Args:
            image: PIL Image object

        Returns:
            Processed PIL Image
        """
        try:
            from PIL import Image, ImageOps

            # Convert to RGB if necessary
            if image.mode not in ('L', 'RGB'):
                image = image.convert('RGB')

            # Resize if image is too small (minimum 300 DPI equivalent)
            if image.width < self.MIN_IMAGE_DIMENSION or image.height < self.MIN_IMAGE_DIMENSION:
                # Scale up small images
                scale_factor = max(
                    self.MIN_IMAGE_DIMENSION / image.width,
                    self.MIN_IMAGE_DIMENSION / image.height
                )
                new_width = int(image.width * scale_factor)
                new_height = int(image.height * scale_factor)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convert to grayscale for better OCR
            image = ImageOps.grayscale(image)

            return image

        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image
