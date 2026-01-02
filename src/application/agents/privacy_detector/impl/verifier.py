"""
LLM-based privacy entity verification.
"""

import json
from typing import List, Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from src.application.agents.privacy_detector.core import PrivacyEntity
from src.shared.utils.logger import get_logger
from src.shared.config.settings import settings
from src.shared.llm import llm_retry


logger = get_logger(__name__)


class PrivacyVerifier:
    """
    Verifies detected privacy entities using LLM with privacy-safe masking.
    """

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    async def verify_entities(
        self,
        entities: List[PrivacyEntity],
        text: str
    ) -> List[PrivacyEntity]:
        """
        Verify entities using LLM with privacy-safe masking.

        Only verifies entities with confidence between MIN and MAX thresholds.
        High confidence (>MAX) entities are accepted directly.
        Low confidence (<MIN) entities are discarded.

        Privacy Safety: Never sends actual entity values to LLM.
        Uses <CANDIDATE> placeholder in masked context.

        Args:
            entities: List of detected privacy entities
            text: Original text containing entities

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

        # Verify entities using LLM (with privacy-safe masking)
        verified_entities = await self._verify_entities_batch(verify_entities, text)

        # Combine high confidence and verified entities
        final_entities = high_confidence_entities + verified_entities
        logger.debug(
            f"LLM verification complete: {len(high_confidence_entities)} high confidence, "
            f"{len(verified_entities)} verified, {len(verify_entities) - len(verified_entities)} filtered out, "
            f"{len(low_confidence_entities)} low confidence discarded"
        )
        return final_entities

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

