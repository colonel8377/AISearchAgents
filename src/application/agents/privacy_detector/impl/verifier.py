"""
LLM-based privacy entity verification with Mask-Then-Ask strategy.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from src.application.agents.privacy_detector.core import PrivacyEntity
from src.shared.utils.logger import get_logger
from src.shared.config.settings import settings
from src.shared.llm import llm_retry


logger = get_logger(__name__)


class PrivacyVerifier:
    """
    Mask-Then-Ask privacy verifier.
    
    Uses LLM semantic understanding to verify detection results and filter false positives.
    """

    VERIFICATION_PROMPT_TEMPLATE = """You are a privacy security expert. I need you to verify if a detected item is actually sensitive personal information.

Context: "{context}"

Detection: The text "{placeholder}" was flagged as {entity_type}. 

Question: Based on the surrounding context, is this actually a {entity_type}?  
Consider: 
1. Does the context suggest this is real personal data or just an example/documentation?
2. Is the format consistent with a real {entity_type}?
3. Are there any indicators that this is a placeholder, test data, or version number?

Answer with: 
- "YES" if this is likely real sensitive data
- "NO" if this is likely a false positive (example, test data, version number, etc.)
- Brief reasoning (1 sentence)

Format: YES/NO - [reasoning]"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self._cache: Dict[str, Tuple[bool, float, str]] = {}

    async def verify_entities(
        self,
        entities: List[PrivacyEntity],
        text: str,
        sanitized_text: Optional[str] = None,
        metadata_registry: Optional[Dict[str, Any]] = None
    ) -> List[PrivacyEntity]:
        """
        Verify entities using Mask-Then-Ask strategy.

        Only verifies entities with confidence between MIN and MAX thresholds.
        High confidence (>MAX) entities are accepted directly.
        Low confidence (<MIN) entities are discarded.

        Privacy Safety: Never sends actual entity values to LLM.
        Uses placeholders from sanitized text.

        Args:
            entities: List of detected privacy entities
            text: Original text containing entities
            sanitized_text: Sanitized text with placeholders (optional)
            metadata_registry: Metadata registry mapping placeholders to types (optional)

        Returns:
            Filtered list of entities (with non-sensitive entities removed)
        """
        if not entities:
            return entities

        # Get thresholds from settings
        min_threshold = getattr(settings, 'llm_verification_threshold_min', 0.4)
        max_threshold = getattr(settings, 'llm_verification_threshold_max', 0.8)

        # Categorize entities by confidence
        high_confidence_entities = []  # > max_threshold: accept directly
        verify_entities = []  # min_threshold <= confidence <= max_threshold: verify
        low_confidence_entities = []  # < min_threshold: discard

        for entity in entities:
            if entity.confidence > max_threshold:
                high_confidence_entities.append(entity)
            elif entity.confidence >= min_threshold:
                verify_entities.append(entity)
            else:
                # Low confidence: discard
                logger.debug(f"Discarding low confidence entity: {entity.entity_type} (confidence={entity.confidence:.3f})")
                low_confidence_entities.append(entity)

        if not verify_entities:
            # No entities to verify, return high confidence entities only
            logger.debug(f"LLM verification: {len(high_confidence_entities)} high confidence, {len(low_confidence_entities)} low confidence discarded")
            return high_confidence_entities

        # Verify entities using Mask-Then-Ask
        verified_entities = await self._verify_entities_mask_then_ask(
            verify_entities, text, sanitized_text, metadata_registry
        )

        # Combine high confidence and verified entities
        final_entities = high_confidence_entities + verified_entities
        logger.debug(
            f"LLM verification complete: {len(high_confidence_entities)} high confidence, "
            f"{len(verified_entities)} verified, {len(verify_entities) - len(verified_entities)} filtered out, "
            f"{len(low_confidence_entities)} low confidence discarded"
        )
        return final_entities

    async def verify_single(
        self,
        entity: PrivacyEntity,
        context: str,
        placeholder: str
    ) -> Tuple[bool, float, str]:
        """
        Verify a single entity.

        Args:
            entity: Entity to verify
            context: Context text (sanitized)
            placeholder: Placeholder text (e.g., "<EMAIL_1>")

        Returns:
            (is_valid, confidence_adjustment, reasoning)
        """
        # Check cache
        cache_key = f"{entity.entity_type}:{context[:100]}:{placeholder}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Build verification prompt
        prompt = self.VERIFICATION_PROMPT_TEMPLATE.format(
            context=context,
            placeholder=placeholder,
            entity_type=entity.entity_type
        )

        # Call LLM
        try:
            response = await self._call_llm(prompt)
            is_valid, confidence_adj, reasoning = self._parse_response(response)
        except Exception as e:
            logger.warning(f"Verification failed for {entity.entity_type}: {e}")
            # Fail-safe: default to valid
            is_valid, confidence_adj, reasoning = True, 0.0, "Verification failed, defaulting to valid"

        # Cache result
        self._cache[cache_key] = (is_valid, confidence_adj, reasoning)

        return is_valid, confidence_adj, reasoning

    async def verify_batch(
        self,
        entities: List[PrivacyEntity],
        contexts: List[str],
        placeholders: List[str]
    ) -> List[Tuple[bool, float, str]]:
        """
        Batch verify multiple entities.

        Combines multiple verification requests into a single LLM call for efficiency.
        """
        if len(entities) != len(contexts) or len(entities) != len(placeholders):
            raise ValueError("entities, contexts, and placeholders must have same length")

        # Build batch prompt
        batch_prompt_parts = [
            "You are a privacy security expert. Verify multiple detected items.",
            "",
            "Contexts and detections:"
        ]

        for idx, (entity, context, placeholder) in enumerate(zip(entities, contexts, placeholders)):
            batch_prompt_parts.append(
                f"\n[{idx}] Entity Type: {entity.entity_type}\n"
                f"Context: \"{context}\"\n"
                f"Detection: \"{placeholder}\" was flagged as {entity.entity_type}.\n"
                f"Is this actually a {entity.entity_type}? (YES/NO - reasoning)"
            )

        batch_prompt_parts.append(
            "\nReturn JSON array with one result per detection:\n"
            '[{"index": 0, "is_valid": true/false, "reasoning": "brief explanation"}, ...]'
        )

        batch_prompt = "\n".join(batch_prompt_parts)

        # Call LLM
        try:
            response = await self._call_llm(batch_prompt)
            results = self._parse_batch_response(response, len(entities))
        except Exception as e:
            logger.warning(f"Batch verification failed: {e}")
            # Fail-safe: default all to valid
            results = [(True, 0.0, "Verification failed, defaulting to valid")] * len(entities)

        return results

    async def _verify_entities_mask_then_ask(
        self,
        entities: List[PrivacyEntity],
        original_text: str,
        sanitized_text: Optional[str],
        metadata_registry: Optional[Dict[str, Any]]
    ) -> List[PrivacyEntity]:
        """
        Verify entities using Mask-Then-Ask strategy.
        """
        verified_entities = []

        # Use sanitized text if available, otherwise use original with masking
        if sanitized_text and metadata_registry:
            # Find placeholders for each entity
            entity_placeholders = {}
            for entity in entities:
                # Try to find placeholder in metadata registry
                found = False
                for registry_key, meta in metadata_registry.items():
                    if not isinstance(meta, dict):
                        continue
                    original_value = meta.get("original_value", "")
                    if original_value == entity.text:
                        # Use synthetic_value (from sanitizer) or sanitized_value (for backward compatibility)
                        placeholder_val = meta.get("synthetic_value") or meta.get("sanitized_value")
                        if placeholder_val:
                            entity_placeholders[entity] = placeholder_val
                            found = True
                            break
                if not found:
                    entity_placeholders[entity] = "<CANDIDATE>"
            
            # Extract contexts for each entity
            contexts = []
            placeholders = []
            for entity in entities:
                placeholder = entity_placeholders.get(entity, "<CANDIDATE>")
                # Extract context window around placeholder
                placeholder_pos = sanitized_text.find(placeholder)
                if placeholder_pos != -1:
                    context_start = max(0, placeholder_pos - 50)
                    context_end = min(len(sanitized_text), placeholder_pos + len(placeholder) + 50)
                    context = sanitized_text[context_start:context_end]
                else:
                    # Fallback: use entity position in original text
                    context_start = max(0, entity.start - 50)
                    context_end = min(len(original_text), entity.end + 50)
                    context = original_text[context_start:context_end]
                
                contexts.append(context)
                placeholders.append(placeholder)
            
            # Batch verify
            verification_results = await self.verify_batch(entities, contexts, placeholders)
            
            # Filter entities based on verification results
            for entity, (is_valid, conf_adj, reasoning) in zip(entities, verification_results):
                if is_valid:
                    # Adjust confidence
                    entity.confidence = min(1.0, entity.confidence + conf_adj)
                    verified_entities.append(entity)
                else:
                    logger.debug(
                        f"LLM verified entity as false positive: {entity.entity_type} "
                        f"at {entity.start}-{entity.end} (reasoning: {reasoning})"
                    )
        else:
            # Fallback: use original masking approach
            verified_entities = await self._verify_entities_batch(entities, original_text)

        return verified_entities

    async def _verify_entities_batch(
        self,
        entities: List[PrivacyEntity],
        text: str
    ) -> List[PrivacyEntity]:
        """
        Verify a batch of entities using LLM with privacy-safe masking.
        """
        verified_entities = []

        # Process entities in batches to avoid token limits
        batch_size = 5  # Process 5 entities at a time
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            try:
                # Create masked context for batch
                masked_contexts = []
                for entity in batch:
                    # Create masked text by replacing entity value with <CANDIDATE>
                    # CRITICAL: Never send actual entity value to LLM
                    masked_text = (
                        text[:entity.start] +
                        "<CANDIDATE>" +
                        text[entity.end:]
                    )
                    masked_contexts.append({
                        "entity": entity,
                        "masked_context": masked_text,
                        "position": f"position {entity.start}-{entity.end}"
                    })

                # Build verification prompt
                verification_prompt = self._build_verification_prompt(masked_contexts)
                system_prompt = """You are a privacy verification assistant. Analyze the given contexts and determine if the <CANDIDATE> placeholders represent sensitive information.

Return JSON format with an array of results, one for each candidate:
[
  {"index": 0, "is_sensitive": true, "reasoning": "brief explanation"},
  {"index": 1, "is_sensitive": false, "reasoning": "brief explanation"},
  ...
]

Only mark as sensitive if the candidate represents actual privacy information (passwords, API keys, personal identifiers, etc.)."""

                # Call LLM for verification
                response_text = await self._call_llm(verification_prompt, system_prompt)

                # Parse verification results
                verification_results = self._parse_verification_response(response_text, len(batch))

                # Filter entities based on verification results
                for idx, entity in enumerate(batch):
                    if idx < len(verification_results):
                        result = verification_results[idx]
                        if result.get("is_sensitive", True):  # Default to sensitive (fail-safe)
                            verified_entities.append(entity)
                        else:
                            logger.debug(
                                f"LLM verified entity as non-sensitive: {entity.entity_type} "
                                f"at {entity.start}-{entity.end} (confidence={entity.confidence:.3f})"
                            )
                    else:
                        # If parsing failed for this entity, keep it (fail-safe)
                        logger.warning(f"Verification result missing for entity {idx}, keeping entity (fail-safe)")
                        verified_entities.append(entity)

            except Exception as e:
                # Fail-safe: if verification fails, keep all entities in batch
                logger.warning(f"LLM verification failed for batch: {e}, keeping all entities (fail-safe)", exc_info=True)
                verified_entities.extend(batch)

        return verified_entities

    def _build_verification_prompt(self, masked_contexts: List[Dict[str, Any]]) -> str:
        """
        Build verification prompt with masked contexts.
        """
        prompt_parts = [
            "Analyze the following contexts and determine if each <CANDIDATE> placeholder represents sensitive information.",
            "",
            "Contexts:"
        ]

        for idx, context_info in enumerate(masked_contexts):
            entity = context_info["entity"]
            masked_context = context_info["masked_context"]
            position = context_info["position"]

            # Extract context window around the candidate (50 chars before/after)
            candidate_pos = masked_context.find("<CANDIDATE>")
            if candidate_pos != -1:
                start = max(0, candidate_pos - 50)
                end = min(len(masked_context), candidate_pos + len("<CANDIDATE>") + 50)
                context_window = masked_context[start:end]
            else:
                context_window = masked_context[:100]  # Fallback

            prompt_parts.append(
                f"\n[{idx}] Entity Type: {entity.entity_type}\n"
                f"Position: {position}\n"
                f"Context: \"{context_window}\""
            )

        prompt_parts.append(
            "\nReturn JSON array with one result per candidate:\n"
            '[{"index": 0, "is_sensitive": true/false, "reasoning": "brief explanation"}, ...]'
        )

        return "\n".join(prompt_parts)

    def _parse_verification_response(self, response_text: str, expected_count: int) -> List[Dict[str, Any]]:
        """
        Parse LLM verification response JSON.
        """
        try:
            response_text = response_text.strip()

            # Handle markdown code blocks
            if "```" in response_text:
                import re
                json_pattern = r'```\w*\s*(\[.*?\])'
                match = re.search(json_pattern, response_text, re.DOTALL)
                if match:
                    response_text = match.group(1).strip()
                else:
                    json_start = response_text.find('[')
                    json_end = response_text.rfind(']') + 1
                    if json_start != -1 and json_end > json_start:
                        response_text = response_text[json_start:json_end]

            parsed = json.loads(response_text)

            if not isinstance(parsed, list):
                # Try to extract array from dict
                if isinstance(parsed, dict) and "results" in parsed:
                    parsed = parsed["results"]
                else:
                    raise json.JSONDecodeError("Response is not an array", response_text, 0)

            # Validate and normalize results
            results = []
            for item in parsed:
                if isinstance(item, dict):
                    # Extract index (default to position in array)
                    idx = item.get("index", len(results))
                    if not isinstance(idx, int):
                        try:
                            idx = int(idx)
                        except (ValueError, TypeError):
                            idx = len(results)

                    # Extract is_sensitive (default to True for fail-safe)
                    is_sensitive = item.get("is_sensitive", True)
                    if not isinstance(is_sensitive, bool):
                        is_sensitive = bool(is_sensitive)

                    results.append({
                        "index": idx,
                        "is_sensitive": is_sensitive,
                        "reasoning": item.get("reasoning", "")
                    })

            # Sort by index and ensure we have expected_count results
            results.sort(key=lambda x: x["index"])
            while len(results) < expected_count:
                # Fill missing results with fail-safe default (sensitive)
                results.append({
                    "index": len(results),
                    "is_sensitive": True,
                    "reasoning": "Missing result, defaulting to sensitive (fail-safe)"
                })

            return results[:expected_count]  # Return only expected count

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse verification response: {e}, using fail-safe defaults")
            # Fail-safe: return all sensitive (keep entities)
            return [
                {"index": i, "is_sensitive": True, "reasoning": "Parse error, defaulting to sensitive (fail-safe)"}
                for i in range(expected_count)
            ]

    def _parse_response(self, response: str) -> Tuple[bool, float, str]:
        """
        Parse LLM verification response.

        Expected format: "YES/NO - [reasoning]"
        """
        response = response.strip().upper()

        if response.startswith("YES"):
            reasoning = response[3:].strip(" -")
            return True, 0.1, reasoning  # Valid, boost confidence by 0.1
        elif response.startswith("NO"):
            reasoning = response[2:].strip(" -")
            return False, -0.3, reasoning  # Invalid, reduce confidence by 0.3
        else:
            # Unable to parse, default to valid (fail-safe)
            return True, 0.0, "Unable to parse response, defaulting to valid"

    def _parse_batch_response(self, response_text: str, expected_count: int) -> List[Tuple[bool, float, str]]:
        """
        Parse batch verification response JSON.
        """
        try:
            response_text = response_text.strip()

            # Handle markdown code blocks
            if "```" in response_text:
                json_pattern = r'```\w*\s*(\[.*?\])'
                match = re.search(json_pattern, response_text, re.DOTALL)
                if match:
                    response_text = match.group(1).strip()
                else:
                    json_start = response_text.find('[')
                    json_end = response_text.rfind(']') + 1
                    if json_start != -1 and json_end > json_start:
                        response_text = response_text[json_start:json_end]

            parsed = json.loads(response_text)

            if not isinstance(parsed, list):
                if isinstance(parsed, dict) and "results" in parsed:
                    parsed = parsed["results"]
                else:
                    raise json.JSONDecodeError("Response is not an array", response_text, 0)

            # Parse results
            results = []
            for item in parsed:
                if isinstance(item, dict):
                    idx = item.get("index", len(results))
                    is_valid = item.get("is_valid", True)  # Default to valid (fail-safe)
                    reasoning = item.get("reasoning", "")
                    
                    confidence_adj = 0.1 if is_valid else -0.3
                    results.append((is_valid, confidence_adj, reasoning))

            # Ensure we have expected_count results
            while len(results) < expected_count:
                results.append((True, 0.0, "Missing result, defaulting to valid"))

            return results[:expected_count]

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse batch verification response: {e}")
            # Fail-safe: return all valid
            return [(True, 0.0, "Parse error, defaulting to valid")] * expected_count

    def get_verification_priority(self, entity: PrivacyEntity) -> int:
        """
        Get verification priority.

        Higher return value means more verification needed:
        - PERSON/NAME: 3 (highest priority, most false positives)
        - PHONE (non-standard format): 3
        - ADDRESS: 2
        - EMAIL/CREDIT_CARD: 1 (clear format, less verification needed)
        - API_KEY/PASSWORD: 1 (depends on context)
        """
        entity_type_lower = entity.entity_type.lower()
        
        if entity_type_lower in ["person", "person_name", "name"]:
            return 3
        elif entity_type_lower in ["phone_number", "phone"]:
            # Check if standard format
            if re.match(r'^\+?[0-9]{10,15}$', entity.text):
                return 1
            return 3
        elif entity_type_lower in ["physical_address", "address"]:
            return 2
        elif entity_type_lower in ["email_address", "email", "credit_card_number", "credit_card"]:
            return 1
        elif entity_type_lower in ["api_key", "password", "secret_key"]:
            return 1
        
        return 2  # Default priority

    @llm_retry
    async def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Call LLM with retry logic.
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        response = await self.llm.ainvoke(messages)
        return response.content

