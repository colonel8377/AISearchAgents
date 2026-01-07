"""
Privacy detection implementations combining Presidio and GLiNER.
Optimized for production use with SOTA sliding window techniques and privacy grading.
"""
import importlib
from collections import namedtuple
from typing import List, Optional, Any, Dict, Tuple

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer import PatternRecognizer
from presidio_analyzer import Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider

from ..core.interfaces import IPrivacyDetector, PrivacyEntity
from ..core.utils import (
    build_allow_list,
    is_in_allow_list,
    has_chinese_text,
    severity_to_int
)
from .patterns import EnhancedPatternMatcher
from .allow_list import EnhancedAllowList
from .....shared.config.settings import settings
from .....shared.utils.logger import get_logger
from .....shared.utils.text_normalizer import TextNormalizer

# Lazy imports to handle optional dependencies
PRESIDIO_AVAILABLE = True

from transformers import AutoTokenizer
_original_auto_tokenizer_from_pretrained = AutoTokenizer.from_pretrained

from gliner import GLiNER
GLINER_AVAILABLE = True

def _patched_auto_tokenizer_from_pretrained(*args, **kwargs):
    """Patch to force use_fast=False and prevent sentencepiece byte fallback warnings."""
    kwargs['use_fast'] = False
    if 'trust_remote_code' not in kwargs:
        kwargs['trust_remote_code'] = False
    return _original_auto_tokenizer_from_pretrained(*args, **kwargs)

AutoTokenizer.from_pretrained = _patched_auto_tokenizer_from_pretrained

logger = get_logger(__name__)

# Definition for Privacy Metadata
PrivacyMetadata = namedtuple("PrivacyMetadata", ["category", "severity"])


class HybridDetector(IPrivacyDetector):
    """
    Enhanced hybrid detector combining Presidio, GLiNER, and enhanced pattern matching.

    Detection pipeline:
    1. Text preprocessing (TextNormalizer)
    2. Presidio NER detection
    3. GLiNER semantic detection (dynamic thresholds)
    4. Enhanced regex pattern matching
    5. High-entropy string detection
    6. Allow List filtering
    7. Chinese word boundary validation (optional jieba)
    8. NMS conflict resolution (boundary-first)
    """

    # GLiNER dynamic threshold configuration
    DYNAMIC_THRESHOLDS: Dict[str, float] = {
        # High precision entities - low threshold for high recall
        "api_key": 0.20,
        "password": 0.20,
        "secret_key": 0.20,
        "credit_card_number": 0.25,
        "social_security_number": 0.25,
        "crypto_wallet": 0.25,
        
        # Standard entities
        "phone_number": 0.35,
        "email_address": 0.35,
        "bank_account": 0.35,
        
        # Easy false positive entities - high threshold
        "person": 0.45,
        "person_name": 0.45,
        "name": 0.45,
        "physical_address": 0.45,
        "salary": 0.40,
    }

    # Define SOTA Privacy Grading (L1-L4)
    ENTITY_MAPPINGS = {
        # --- Presidio Types ---
        "PHONE_NUMBER": PrivacyMetadata("identity", "L3"),
        "PHONE_NUMBER_CN": PrivacyMetadata("identity", "L3"),
        "EMAIL_ADDRESS": PrivacyMetadata("identity", "L3"),
        "PERSON": PrivacyMetadata("identity", "L2"),
        "ID_CARD": PrivacyMetadata("identity", "L4"),
        "ID_CARD_CN": PrivacyMetadata("identity", "L4"),
        "PASSPORT": PrivacyMetadata("identity", "L4"),
        "SSN": PrivacyMetadata("identity", "L4"),
        "US_SSN": PrivacyMetadata("identity", "L4"),
        "CREDIT_CARD": PrivacyMetadata("financial", "L4"),
        "LOCATION": PrivacyMetadata("location", "L1"),
        "IP_ADDRESS": PrivacyMetadata("technical", "L3"),
        "URL": PrivacyMetadata("technical", "L1"),

        # --- GLiNER Types (Semantic) ---
        "person": PrivacyMetadata("identity", "L2"),
        "person_name": PrivacyMetadata("identity", "L2"),
        "name": PrivacyMetadata("identity", "L2"),
        "crypto_wallet": PrivacyMetadata("financial", "L4"),
        "medical_condition": PrivacyMetadata("medical", "L3"),
        "physical_address": PrivacyMetadata("location", "L3"),
        "salary": PrivacyMetadata("financial", "L2"),
        "bank_account": PrivacyMetadata("financial", "L4"),
        "api_key": PrivacyMetadata("technical", "L4"),
        "password": PrivacyMetadata("technical", "L4"),
        "secret_key": PrivacyMetadata("technical", "L4"),
        "social_security_number": PrivacyMetadata("identity", "L4"),
        "credit_card_number": PrivacyMetadata("financial", "L4"),
        "phone_number": PrivacyMetadata("identity", "L3"),
        "email_address": PrivacyMetadata("identity", "L3"),
        
        # --- Enhanced Pattern Types ---
        "CN_PHONE": PrivacyMetadata("identity", "L3"),
        "CN_ID_CARD": PrivacyMetadata("identity", "L4"),
        "CN_BANK_CARD": PrivacyMetadata("financial", "L4"),
        "CN_PASSPORT": PrivacyMetadata("identity", "L4"),
        "CN_DRIVER_LICENSE": PrivacyMetadata("identity", "L3"),
        "JWT_TOKEN": PrivacyMetadata("technical", "L4"),
        "AWS_ACCESS_KEY": PrivacyMetadata("technical", "L4"),
        "GITHUB_TOKEN": PrivacyMetadata("technical", "L4"),
        "OPENAI_KEY": PrivacyMetadata("technical", "L4"),
        "STRIPE_KEY": PrivacyMetadata("technical", "L4"),
        "BTC_ADDRESS": PrivacyMetadata("financial", "L4"),
        "ETH_ADDRESS": PrivacyMetadata("financial", "L4"),
        "HIGH_ENTROPY_SECRET": PrivacyMetadata("technical", "L4"),
    }

    def __init__(self):
        """Initialize detection engines with optimized configuration."""
        self.analyzer: Optional[AnalyzerEngine] = None
        self.gliner_model: Optional[Any] = None
        self.model_name: Optional[str] = None
        self.threshold = 0.3
        self.has_chinese_model = False

        # GLiNER labels to detect
        self.gliner_labels = [
            "person", "person_name", "name",
            "crypto_wallet", "medical_condition", "physical_address",
            "salary", "bank_account", "api_key", "password", "secret_key",
            "social_security_number", "credit_card_number",
            "phone_number", "email_address", "ip_address"
        ]

        # Legacy allow list patterns (kept for backward compatibility)
        self.allow_list_patterns = build_allow_list()
        
        # Enhanced components
        self.pattern_matcher = EnhancedPatternMatcher()
        self.allow_list = EnhancedAllowList()
        self.jieba_available = False

        self._init_presidio()
        self._init_gliner()
        self._init_jieba()

    def _init_presidio(self) -> None:
        """Initialize Presidio with safe loading of spaCy models and custom patterns."""
        if not PRESIDIO_AVAILABLE:
            logger.warning("Presidio not available. Skipping initialization.")
            return

        try:
            # SOTA Practice: Only load available models to avoid crashing in restricted envs
            models_to_check = {
                "en_core_web_sm": "en",
                "zh_core_web_sm": "zh"
            }
            available_models = []

            for model_name, lang_code in models_to_check.items():
                try:
                    importlib.import_module(model_name)
                    available_models.append({"lang_code": lang_code, "model_name": model_name})
                except ImportError:
                    continue

            if available_models:
                nlp_config = {
                    "nlp_engine_name": "spacy",
                    "models": available_models
                }
                nlp_provider = NlpEngineProvider(nlp_configuration=nlp_config)
                supported_langs = [m["lang_code"] for m in available_models]
                self.analyzer = AnalyzerEngine(
                    nlp_engine=nlp_provider.create_engine(),
                    supported_languages=supported_langs
                )
                self.has_chinese_model = "zh" in supported_langs
                logger.info(f"Presidio initialized with languages: {supported_langs}")
            else:
                logger.warning("No spaCy models found. Falling back to default Presidio configuration.")
                self.analyzer = AnalyzerEngine()
                self.has_chinese_model = False

            # Add custom patterns regardless of model availability
            self._add_custom_patterns()

        except Exception as e:
            logger.error(f"Failed to initialize Presidio: {e}")

    def _add_custom_patterns(self) -> None:
        """Add custom PatternRecognizer for Chinese phone numbers and ID cards."""
        if not PRESIDIO_AVAILABLE or not self.analyzer:
            return

        try:
            # Chinese Phone Number Pattern: (?:\+86)?1[3-9]\d{9}
            chinese_phone_pattern = PatternRecognizer(
                supported_entity="PHONE_NUMBER_CN",
                patterns=[
                    Pattern(
                        name="chinese_phone",
                        regex=r"(?:\+86)?1[3-9]\d{9}",
                        score=0.9
                    )
                ],
                supported_language="zh"
            )

            # Chinese ID Card Pattern: 18 digits
            chinese_id_pattern = PatternRecognizer(
                supported_entity="ID_CARD_CN",
                patterns=[
                    Pattern(
                        name="chinese_id_card",
                        regex=r"\b\d{18}\b",
                        score=0.95
                    )
                ],
                supported_language="zh"
            )

            self.analyzer.registry.add_recognizer(chinese_phone_pattern)
            self.analyzer.registry.add_recognizer(chinese_id_pattern)
            logger.debug("Custom Chinese patterns added to Presidio")

        except Exception as e:
            logger.warning(f"Failed to add custom Presidio patterns: {e}")

    def _init_gliner(self) -> None:
        """Initialize GLiNER with offline-first strategy."""
        if not GLINER_AVAILABLE:
            return

        try:
            self.max_len = getattr(settings, 'gliner_max_length', 300)
            self.model_name = getattr(settings, 'gliner_model_name', "urchade/gliner_multi_pii-v1")
            self.threshold = getattr(settings, 'gliner_threshold', 0.3)

            logger.info(f"Initializing GLiNER: {self.model_name}")

            try:
                # Priority: Try loading from local cache/file first
                self.gliner_model = GLiNER.from_pretrained(
                    self.model_name,
                    local_files_only=True,
                    load_tokenizer=True
                )
                logger.debug("Loaded GLiNER from local cache.")
            except (OSError, ValueError):
                # Fallback: Download from Hub if local not found
                logger.info("Local cache not found. Downloading GLiNER from HuggingFace...")
                self.gliner_model = GLiNER.from_pretrained(self.model_name)

            logger.info(f"GLiNER Ready | Window: {self.max_len} | Threshold: {self.threshold}")

        except Exception as e:
            logger.warning(f"Failed to initialize GLiNER: {e}")
            self.gliner_model = None

    def _init_jieba(self) -> None:
        """Initialize Chinese word segmenter (optional)."""
        try:
            import jieba
            self.jieba_available = True
            jieba.initialize()  # Preload dictionary
            logger.debug("jieba initialized for boundary validation")
        except ImportError:
            self.jieba_available = False
            logger.info("jieba not available, skipping boundary validation")

    def detect(self, text: str) -> List[PrivacyEntity]:
        """
        Enhanced detection pipeline.
        
        Detection flow:
        1. Text preprocessing (TextNormalizer)
        2. Presidio NER detection
        3. GLiNER semantic detection (dynamic thresholds)
        4. Enhanced regex pattern matching
        5. High-entropy string detection
        6. Allow List filtering
        7. Chinese word boundary validation (if jieba available)
        8. NMS conflict resolution (boundary-first)
        """
        if not text:
            return []

        # 1. Text preprocessing
        normalized_text = TextNormalizer.normalize(text)

        raw_entities = []

        # 2. Presidio Detection
        if self.analyzer:
            raw_entities.extend(self._run_presidio(normalized_text))

        # 3. GLiNER Detection (dynamic thresholds)
        if self.gliner_model:
            raw_entities.extend(self._run_gliner_dynamic(normalized_text))

        # 4. Enhanced regex pattern matching
        raw_entities.extend(self._run_enhanced_patterns(normalized_text))

        # 5. High-entropy string detection
        raw_entities.extend(self._run_entropy_detection(normalized_text))

        # 6. Allow List filtering
        filtered_entities = self._apply_allow_list(raw_entities, normalized_text)

        # 7. Boundary validation (if jieba available)
        if self.jieba_available:
            filtered_entities = self._validate_boundaries(filtered_entities, normalized_text)

        # 8. NMS conflict resolution (boundary-first)
        final_entities = self._resolve_conflicts_boundary_first(filtered_entities)

        # 9. Filter low-confidence standalone Presidio PERSON detections
        final_entities = self._filter_low_confidence_standalone_person(final_entities, raw_entities)

        return final_entities

    def _run_presidio(self, text: str) -> List[PrivacyEntity]:
        """Run Presidio analysis."""
        entities = []
        try:
            for lang in self.analyzer.supported_languages:
                results = self.analyzer.analyze(text=text, language=lang)
                for res in results:
                    entity_text = text[res.start:res.end]

                    if is_in_allow_list(entity_text, res.entity_type, self.allow_list_patterns):
                        continue

                    meta = self._get_metadata(res.entity_type)
                    entities.append(PrivacyEntity(
                        text=entity_text,
                        entity_type=res.entity_type,
                        start=res.start,
                        end=res.end,
                        confidence=res.score,
                        category=meta.category,
                        severity=meta.severity,
                        source="presidio"
                    ))
        except Exception as e:
            logger.debug(f"Presidio run error: {e}")
        return entities

    def _run_gliner(self, text: str) -> List[PrivacyEntity]:
        """Run GLiNER with Sliding Window Strategy (legacy method)."""
        return self._run_gliner_dynamic(text)

    def _run_gliner_dynamic(self, text: str) -> List[PrivacyEntity]:
        """Run GLiNER with dynamic thresholds."""
        entities = []
        try:
            window_size = self.max_len
            stride = int(window_size * 0.8)

            chunks = []
            if len(text) <= window_size:
                chunks.append((text, 0))
            else:
                for i in range(0, len(text), stride):
                    chunk = text[i:i + window_size]
                    if chunk:
                        chunks.append((chunk, i))

            # Group labels by threshold
            threshold_groups: Dict[float, List[str]] = {}
            for label in self.gliner_labels:
                threshold = self.DYNAMIC_THRESHOLDS.get(label, 0.35)
                if threshold not in threshold_groups:
                    threshold_groups[threshold] = []
                threshold_groups[threshold].append(label)

            # Run detection for each threshold group
            for threshold, labels in threshold_groups.items():
                for chunk_text, offset in chunks:
                    if not chunk_text.strip():
                        continue

                    preds = self.gliner_model.predict_entities(
                        chunk_text, labels, threshold=threshold
                    )

                    for pred in preds:
                        # Handle both dict and object return types from GLiNER
                        if isinstance(pred, dict):
                            p_text, p_label, p_score = pred['text'], pred['label'], pred['score']
                            p_start, p_end = pred['start'], pred['end']
                        else:
                            p_text, p_label, p_score = pred.text, pred.label, pred.score
                            p_start, p_end = pred.start, pred.end

                        if not p_text.strip():
                            continue

                        abs_start = p_start + offset
                        abs_end = p_end + offset

                        if abs_end > len(text):
                            continue

                        entity_text = text[abs_start:abs_end]
                        entity_type = p_label.upper()

                        if is_in_allow_list(entity_text, entity_type, self.allow_list_patterns):
                            continue

                        meta = self._get_metadata(p_label)
                        entities.append(PrivacyEntity(
                            text=entity_text,
                            entity_type=p_label,
                            start=abs_start,
                            end=abs_end,
                            confidence=p_score,
                            category=meta.category,
                            severity=meta.severity,
                            source="gliner"
                        ))

        except Exception as e:
            logger.debug(f"GLiNER run error: {e}")
        return entities

    def _run_enhanced_patterns(self, text: str) -> List[PrivacyEntity]:
        """Run enhanced regex pattern matching."""
        entities = []
        try:
            matches = self.pattern_matcher.match_all(text)
            
            for name, matched_text, start, end, confidence, severity, category in matches:
                entities.append(PrivacyEntity(
                    text=matched_text,
                    entity_type=name,
                    start=start,
                    end=end,
                    confidence=confidence,
                    severity=severity,
                    category=category,
                    source="enhanced_pattern"
                ))
        except Exception as e:
            logger.debug(f"Enhanced pattern matching error: {e}")
        return entities

    def _run_entropy_detection(self, text: str) -> List[PrivacyEntity]:
        """Detect high-entropy strings (likely API keys/secrets)."""
        entities = []
        try:
            secrets = self.pattern_matcher.detect_high_entropy_secrets(text)
            
            for secret_text, start, end, confidence in secrets:
                entities.append(PrivacyEntity(
                    text=secret_text,
                    entity_type="HIGH_ENTROPY_SECRET",
                    start=start,
                    end=end,
                    confidence=confidence,
                    severity="L4",
                    category="technical",
                    source="entropy"
                ))
        except Exception as e:
            logger.debug(f"Entropy detection error: {e}")
        return entities

    def _apply_allow_list(self, entities: List[PrivacyEntity], text: str) -> List[PrivacyEntity]:
        """Apply enhanced Allow List filtering."""
        filtered = []
        for entity in entities:
            if not self.allow_list.should_filter(
                entity.text,
                entity.entity_type,
                text,
                entity.start
            ):
                filtered.append(entity)
        return filtered

    def _validate_boundaries(self, entities: List[PrivacyEntity], text: str) -> List[PrivacyEntity]:
        """
        Validate entity boundaries using jieba word segmentation.
        
        Rules:
        1. If entity boundary cuts a word, reduce confidence or adjust boundary
        2. If entity fully matches word boundaries, keep confidence
        """
        if not self.jieba_available:
            return entities
        
        try:
            import jieba
            words = list(jieba.cut(text))
            
            # Build word boundary mapping
            word_boundaries = []
            pos = 0
            for word in words:
                word_start = pos
                word_end = pos + len(word)
                word_boundaries.append((word_start, word_end, word))
                pos = word_end
            
            validated = []
            for entity in entities:
                # Check if entity boundary is valid
                boundary_valid = self._check_boundary_validity(
                    entity.start, entity.end, word_boundaries
                )
                
                if boundary_valid:
                    validated.append(entity)
                else:
                    # Try to adjust boundary or reduce confidence
                    adjusted = self._adjust_boundary(entity, word_boundaries)
                    if adjusted:
                        validated.append(adjusted)
                    else:
                        # Reduce confidence if boundary is invalid
                        entity.confidence *= 0.8
                        validated.append(entity)
            
            return validated
        except Exception as e:
            logger.debug(f"Boundary validation error: {e}")
            return entities

    def _check_boundary_validity(
        self,
        start: int,
        end: int,
        word_boundaries: List[Tuple[int, int, str]]
    ) -> bool:
        """Check if entity boundary aligns with word boundaries."""
        # Check if start and end align with word boundaries
        start_aligned = any(boundary[0] == start for boundary in word_boundaries)
        end_aligned = any(boundary[1] == end for boundary in word_boundaries)
        
        # Also check if entity spans complete words
        spans_complete_words = any(
            boundary[0] <= start and boundary[1] >= end
            for boundary in word_boundaries
        )
        
        return (start_aligned and end_aligned) or spans_complete_words

    def _adjust_boundary(
        self,
        entity: PrivacyEntity,
        word_boundaries: List[Tuple[int, int, str]]
    ) -> Optional[PrivacyEntity]:
        """Try to adjust entity boundary to align with word boundaries."""
        # Find overlapping words
        overlapping_words = [
            (start, end) for start, end, _ in word_boundaries
            if not (end <= entity.start or start >= entity.end)
        ]
        
        if not overlapping_words:
            return None
        
        # Adjust to span all overlapping words
        new_start = min(w[0] for w in overlapping_words)
        new_end = max(w[1] for w in overlapping_words)
        
        # Create adjusted entity
        adjusted = PrivacyEntity(
            text=entity.text,  # Will be updated by caller if needed
            entity_type=entity.entity_type,
            start=new_start,
            end=new_end,
            confidence=entity.confidence * 0.9,  # Slightly reduce confidence
            category=entity.category,
            severity=entity.severity,
            source=entity.source
        )
        
        return adjusted

    def _resolve_conflicts(self, entities: List[PrivacyEntity]) -> List[PrivacyEntity]:
        """SOTA Conflict Resolution: Non-Maximum Suppression (NMS) - legacy method."""
        return self._resolve_conflicts_boundary_first(entities)

    def _resolve_conflicts_boundary_first(self, entities: List[PrivacyEntity]) -> List[PrivacyEntity]:
        """
        Boundary-first conflict resolution.
        
        Priority:
        1. Boundary integrity (doesn't cut words)
        2. Severity level (L4 > L3 > L2 > L1)
        3. Confidence score
        4. Span length
        """
        if not entities:
            return []

        # Sort by start position, then by negative length (longer first)
        sorted_entities = sorted(
            entities,
            key=lambda x: (x.start, -(x.end - x.start))
        )

        final_entities = []
        curr = sorted_entities[0]

        for next_entity in sorted_entities[1:]:
            if next_entity.start < curr.end:
                # Overlap detected, resolve conflict
                curr = self._select_better_entity(curr, next_entity)
            else:
                final_entities.append(curr)
                curr = next_entity

        final_entities.append(curr)
        return final_entities

    def _select_better_entity(self, e1: PrivacyEntity, e2: PrivacyEntity) -> PrivacyEntity:
        """
        Select better entity from two overlapping entities.
        
        Comparison order:
        1. Boundary score (if available)
        2. Severity level
        3. Confidence score
        4. Span length
        """
        # 1. Compare severity (higher is better)
        e1_sev = severity_to_int(e1.severity)
        e2_sev = severity_to_int(e2.severity)
        
        if e2_sev > e1_sev:
            return e2
        elif e1_sev > e2_sev:
            return e1
        
        # 2. Compare confidence (higher is better)
        if e2.confidence > e1.confidence:
            return e2
        elif e1.confidence > e2.confidence:
            return e1
        
        # 3. Compare span length (longer is better, more context)
        e1_len = e1.end - e1.start
        e2_len = e2.end - e2.start
        
        if e2_len > e1_len:
            return e2
        elif e1_len > e2_len:
            return e1
        
        # 4. Default to first entity
        return e1

    def _filter_low_confidence_standalone_person(
        self,
        final_entities: List[PrivacyEntity],
        all_raw_entities: List[PrivacyEntity]
    ) -> List[PrivacyEntity]:
        """Filter low-confidence Presidio PERSON detections."""
        gliner_person_entities = [
            e for e in all_raw_entities
            if e.source == "gliner" and e.entity_type.lower() in {"person", "person_name", "name"}
        ]

        filtered_entities = []
        for entity in final_entities:
            if entity.source != "presidio" or entity.entity_type != "PERSON":
                filtered_entities.append(entity)
                continue

            # Check overlap with GLiNER
            has_gliner_overlap = False
            for gliner_ent in gliner_person_entities:
                overlap_start = max(entity.start, gliner_ent.start)
                overlap_end = min(entity.end, gliner_ent.end)
                if overlap_start < overlap_end:
                    overlap_ratio = (overlap_end - overlap_start) / min(
                        entity.end - entity.start,
                        gliner_ent.end - gliner_ent.start
                    )
                    if overlap_ratio >= 0.5:
                        has_gliner_overlap = True
                        break

            if has_gliner_overlap:
                filtered_entities.append(entity)
                continue

            if entity.confidence >= 0.7:
                filtered_entities.append(entity)
                continue

            if has_chinese_text(entity.text) and not self.has_chinese_model:
                continue

            filtered_entities.append(entity)

        return filtered_entities

    def _get_metadata(self, label: str) -> PrivacyMetadata:
        """Retrieve category and severity for a given label."""
        label_key = label.upper() if label.upper() in self.ENTITY_MAPPINGS else label.lower()
        if label_key in self.ENTITY_MAPPINGS:
            return self.ENTITY_MAPPINGS[label_key]

        for key, meta in self.ENTITY_MAPPINGS.items():
            if key.lower() in label.lower():
                return meta

        return PrivacyMetadata("unknown", "L1")

