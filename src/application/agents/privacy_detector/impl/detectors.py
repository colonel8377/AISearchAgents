"""
Privacy detection implementations combining Presidio and GLiNER.
Optimized for production use with SOTA sliding window techniques and privacy grading.
"""
import importlib
from collections import namedtuple
from typing import List, Optional, Any

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
from .....shared.config.settings import settings
from .....shared.utils.logger import get_logger

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
    Hybrid detector combining Presidio (Rule/Model-based) and GLiNER (Zero-shot Semantic).

    Architecture:
    - Layer 1: Presidio (Fast, Regex/NER based) - High precision for standard formats (Email, Phone).
    - Layer 2: GLiNER (Deep Learning based) - High recall for semantic context (API Keys, Salaries).
    - Fusion: Non-Maximum Suppression (NMS) based conflict resolution.
    """

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

        # Allow list patterns for false positive filtering
        self.allow_list_patterns = build_allow_list()

        self._init_presidio()
        self._init_gliner()

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

    def detect(self, text: str) -> List[PrivacyEntity]:
        """Detect privacy entities using a fused approach."""
        if not text:
            return []

        raw_entities = []

        # 1. Presidio Detection
        if self.analyzer:
            raw_entities.extend(self._run_presidio(text))

        # 2. GLiNER Detection
        if self.gliner_model:
            raw_entities.extend(self._run_gliner(text))

        # 3. Resolve Conflicts & Finalize
        final_entities = self._resolve_conflicts(raw_entities)

        # 4. Filter low-confidence standalone Presidio PERSON detections
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
        """Run GLiNER with Sliding Window Strategy."""
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

            for chunk_text, offset in chunks:
                if not chunk_text.strip():
                    continue

                preds = self.gliner_model.predict_entities(
                    chunk_text, self.gliner_labels, threshold=self.threshold
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

    def _resolve_conflicts(self, entities: List[PrivacyEntity]) -> List[PrivacyEntity]:
        """SOTA Conflict Resolution: Non-Maximum Suppression (NMS)."""
        if not entities:
            return []

        sorted_entities = sorted(
            entities,
            key=lambda x: (x.start, -(x.end - x.start))
        )

        final_entities = []
        curr = sorted_entities[0]

        for next_ent in sorted_entities[1:]:
            if next_ent.start < curr.end:
                # Collision logic
                curr_sev_val = severity_to_int(curr.severity)
                next_sev_val = severity_to_int(next_ent.severity)

                if next_sev_val > curr_sev_val:
                    curr = next_ent
                elif next_sev_val == curr_sev_val:
                    if next_ent.confidence > curr.confidence:
                        curr = next_ent
                    elif (next_ent.end - next_ent.start) > (curr.end - curr.start):
                        curr = next_ent
            else:
                final_entities.append(curr)
                curr = next_ent

        final_entities.append(curr)
        return final_entities

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

