"""
Privacy detection and Synthetic Anonymization pipeline.
Optimized for Mixed Chinese/English environments with Hard Regex Fallbacks.
"""
import random
import re
import string

from collections import namedtuple
from typing import List, Optional, Any, Tuple

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

from ..core import PrivacyEntity
from ..core.utils import severity_to_int
from .patterns import EnhancedPatternMatcher
from .allow_list import EnhancedAllowList
from .....shared.utils.logger import get_logger
from .....shared.utils.text_normalizer import TextNormalizer


class IPrivacyDetector:
    def detect(self, text: str) -> List[PrivacyEntity]: raise NotImplementedError
    def anonymize(self, text: str) -> Tuple[str, List[PrivacyEntity]]: raise NotImplementedError


PRESIDIO_AVAILABLE = True
GLINER_AVAILABLE = True
logger = get_logger(__name__)

from transformers import AutoTokenizer
_original_tokenizer = AutoTokenizer.from_pretrained
def _patched_tokenizer(*args, **kwargs):
    kwargs['use_fast'] = False
    kwargs['trust_remote_code'] = False
    return _original_tokenizer(*args, **kwargs)
AutoTokenizer.from_pretrained = _patched_tokenizer
from gliner import GLiNER

PrivacyMetadata = namedtuple("PrivacyMetadata", ["category", "severity"])


class SyntheticGenerator:
    """
    Generates realistic 'Imagined' entities to replace sensitive data.
    Maintains consistency within a session (same input -> same output).
    """
    def __init__(self):
        self._cache = {}
        # Data Pools
        self.f_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "David", "Elizabeth"]
        self.l_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Chen", "Wang"]
        self.domains = ["example.com", "test.org", "privacy.net", "demo.io"]
        self.cities = ["New York", "London", "Tokyo", "Shanghai", "Paris", "Berlin"]

    def get_replacement(self, original_text: str, entity_type: str) -> str:
        # 1. Check consistency cache
        if original_text in self._cache:
            return self._cache[original_text]

        # 2. Generate new
        val = self._generate(entity_type)
        self._cache[original_text] = val
        return val

    def _generate(self, et: str) -> str:
        et = et.upper()

        # EMAIL
        if "EMAIL" in et:
            fn = random.choice(self.f_names).lower()
            ln = random.choice(self.l_names).lower()
            return f"{fn}.{ln}@{random.choice(self.domains)}"

        # PHONE
        if "PHONE" in et:
            # Generate generic international format US style
            return f"555.{random.randint(100,999)}.{random.randint(1000,9999)}"

        # PERSON
        if any(x in et for x in ["PERSON", "NAME"]):
            return f"{random.choice(self.f_names)} {random.choice(self.l_names)}"

        # ADDRESS
        if "ADDRESS" in et or "LOCATION" in et:
            return f"{random.randint(1,999)} Park Avenue, {random.choice(self.cities)}"

        # FINANCIAL
        if "CARD" in et or "BANK" in et:
            return f"4{random.randint(100,999)}-****-****-{random.randint(1000,9999)}"
        if "CRYPTO" in et or "WALLET" in et:
            return "0x" + "".join(random.choices(string.hexdigits, k=24))

        # TECHNICAL / API KEYS
        if any(x in et for x in ["KEY", "SECRET", "TOKEN", "PASSWORD"]):
            return "sk_live_" + "".join(random.choices(string.ascii_letters + string.digits, k=20))

        if "IP_ADDRESS" in et:
            return f"192.168.{random.randint(0,255)}.{random.randint(1,255)}"

        # Default fallback
        return f"<{et}_REDACTED>"


class HybridDetector(IPrivacyDetector):
    """
    SOTA Detector combining Regex (Priority), GLiNER (Context), and Presidio (NLP).
    """

    # 1. Metadata Configuration
    ENTITY_MAPPINGS = {
        "EMAIL_ADDRESS": PrivacyMetadata("identity", "L3"),
        "PHONE_NUMBER": PrivacyMetadata("identity", "L3"),
        "PERSON": PrivacyMetadata("identity", "L2"),
        "CREDIT_CARD": PrivacyMetadata("financial", "L4"),
        "API_KEY": PrivacyMetadata("technical", "L4"),
        "IP_ADDRESS": PrivacyMetadata("technical", "L3"),
        "CRYPTO_WALLET": PrivacyMetadata("financial", "L4"),
        "email_address": PrivacyMetadata("identity", "L3"),
        "phone_number": PrivacyMetadata("identity", "L3"),
        "person": PrivacyMetadata("identity", "L2"),
    }

    # 2. Hard Regex Patterns (High Priority / No Filtering)
    # These override NLP models to catch obvious things like emails/phones
    HARD_REGEX = [
        # Email: Standard lenient regex
        ("EMAIL_ADDRESS", r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
        # Phone: Catch 11-digit CN mobile (isolated from digits, but allowing text neighbors)
        ("PHONE_NUMBER", r"(?<!\d)(?:\+86)?1[3-9]\d{9}(?!\d)"),
        # API Keys: Common prefixes
        ("API_KEY", r"(?:sk|pk)_(?:live|test)_[a-zA-Z0-9]{20,}")
    ]

    def __init__(self):
        self.analyzer: Optional[AnalyzerEngine] = None
        self.gliner_model: Optional[Any] = None

        self._init_presidio()
        self._init_gliner()
        self._init_jieba()

        self.allow_list = EnhancedAllowList()
        self.pattern_matcher = EnhancedPatternMatcher()

    def _init_presidio(self):
        if not PRESIDIO_AVAILABLE: return
        try:
            # Attempt to load Spacy NLP engine
            nlp_config = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
            }
            provider = NlpEngineProvider(nlp_configuration=nlp_config)
            self.analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
        except Exception:
            # Fallback (Using default English model if available or empty)
            self.analyzer = AnalyzerEngine()

    def _init_gliner(self):
        if not GLINER_AVAILABLE: return
        try:
            self.gliner_model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
        except Exception:
            self.gliner_model = None

    def _init_jieba(self):
        try:
            import jieba
            jieba.initialize()
            self.jieba_available = True
        except ImportError:
            self.jieba_available = False

    # -------------------------------------------------------------------------
    # PUBLIC API: ANONYMIZE
    # -------------------------------------------------------------------------
    def anonymize(self, text: str) -> Tuple[str, List[PrivacyEntity]]:
        """
        Main function to detect and replace sensitive info with imagined entities.
        """
        if not text: return text, []

        # 1. Run Detection
        entities = self.detect(text)

        # 2. Sort Descending (CRITICAL: End -> Start) to avoid index shifting
        sorted_entities = sorted(entities, key=lambda x: x.start, reverse=True)

        # 3. Setup Generator
        generator = SyntheticGenerator()
        text_chars = list(text)

        # 4. Replacement Loop
        for entity in sorted_entities:
            # Extract original text to serve as seed for consistency
            original_snippet = text[entity.start : entity.end]

            # Get Imagined Entity
            replacement = generator.get_replacement(original_snippet, entity.entity_type)

            # Perform replacement in the character list
            text_chars[entity.start : entity.end] = list(replacement)

        masked_text = "".join(text_chars)
        return masked_text, entities

    # -------------------------------------------------------------------------
    # PUBLIC API: DETECT
    # -------------------------------------------------------------------------
    def detect(self, text: str) -> List[PrivacyEntity]:
        # Normalize (e.g., NFKC), but be careful not to strip characters needed for regex
        normalized_text = TextNormalizer.normalize(text)
        raw_entities = []

        # PHASE 1: Hard Regex (High Confidence, Bypass Filters)
        # This catches "test@hkust.com" instantly.
        for label, pattern in self.HARD_REGEX:
            for match in re.finditer(pattern, normalized_text):
                raw_entities.append(PrivacyEntity(
                    text=match.group(),
                    entity_type=label,
                    start=match.start(),
                    end=match.end(),
                    confidence=1.0,
                    category=self._get_meta(label).category,
                    severity=self._get_meta(label).severity,
                    source="hard_regex"
                ))

        # PHASE 2: NLP Models (Presidio + GLiNER)
        nlp_entities = []

        # Presidio
        if self.analyzer:
            try:
                results = self.analyzer.analyze(text=normalized_text, language="en")
                for r in results:
                    nlp_entities.append(self._create_entity(normalized_text, r.entity_type, r.start, r.end, r.score, "presidio"))
            except Exception: pass

        # GLiNER
        if self.gliner_model:
            labels = ["person", "name", "email_address", "phone_number", "address", "credit_card_number"]
            try:
                preds = self.gliner_model.predict_entities(normalized_text, labels, threshold=0.3)
                for p in preds:
                    if isinstance(p, dict):
                        nlp_entities.append(self._create_entity(normalized_text, p['label'], p['start'], p['end'], p['score'], "gliner"))
                    else:
                        nlp_entities.append(self._create_entity(normalized_text, p.label, p.start, p.end, p.score, "gliner"))
            except Exception: pass

        # PHASE 3: Filter & Merge NLP Results
        # Note: We do NOT filter Hard Regex results through Allow List or Jieba
        nlp_entities = self._apply_allow_list(nlp_entities, normalized_text)

        # Apply Boundary Check to NLP results only (Jieba)
        if self.jieba_available:
            nlp_entities = self._validate_boundaries(nlp_entities, normalized_text)

        # Combine All
        all_entities = raw_entities + nlp_entities

        # Resolve Conflicts (Overlaps)
        final_entities = self._resolve_conflicts(all_entities)

        return final_entities


    def _create_entity(self, text, label, start, end, score, source):
        meta = self._get_meta(label)
        return PrivacyEntity(text[start:end], label.upper(), start, end, score, meta.category, meta.severity, source)

    def _get_meta(self, label):
        l = label.upper()
        if l in self.ENTITY_MAPPINGS: return self.ENTITY_MAPPINGS[l]
        for k, v in self.ENTITY_MAPPINGS.items():
            if k in l: return v
        return PrivacyMetadata("unknown", "L1")

    def _apply_allow_list(self, entities, text):
        return [e for e in entities if not self.allow_list.should_filter(e.text, e.entity_type, text, e.start)]

    def _validate_boundaries(self, entities, text):
        """
        Use Jieba to fix boundary issues, e.g., Presidio detecting only half a Chinese name.
        Crucially, we SKIP this for emails/urls as tokenizers split them incorrectly.
        """
        if not self.jieba_available:
            return entities

        import jieba

        # Create a set of valid boundary indices
        boundaries = set()
        pos = 0
        for word in jieba.cut(text):
            boundaries.add(pos)
            pos += len(word)
        boundaries.add(pos)

        valid = []
        for e in entities:
            # Skip structured data
            if e.entity_type in ["EMAIL_ADDRESS", "URL", "IP_ADDRESS", "API_KEY"]:
                valid.append(e)
                continue

            # Check alignment
            if e.start in boundaries and e.end in boundaries:
                valid.append(e)
            else:
                # If high confidence, keep it anyway (NLP isn't perfect)
                if e.confidence > 0.75:
                    valid.append(e)
        return valid

    def _resolve_conflicts(self, entities):
        """
        Greedy Non-Maximum Suppression (NMS).
        Priority: Hard Regex > Severity > Length > Confidence.
        """
        if not entities: return []

        # Sort by Start Position
        sorted_ents = sorted(entities, key=lambda x: x.start)

        final = [sorted_ents[0]]
        for curr in sorted_ents[1:]:
            prev = final[-1]

            # Check Overlap
            if curr.start < prev.end:
                # 1. Hard Regex Wins
                if prev.source == "hard_regex": continue
                if curr.source == "hard_regex":
                    final[-1] = curr
                    continue

                # 2. Severity Wins
                p_sev = severity_to_int(prev.severity)
                c_sev = severity_to_int(curr.severity)
                if c_sev > p_sev:
                    final[-1] = curr
                    continue
                elif p_sev > c_sev:
                    continue

                # 3. Length Wins
                if (curr.end - curr.start) > (prev.end - prev.start):
                    final[-1] = curr

            else:
                final.append(curr)

        return final