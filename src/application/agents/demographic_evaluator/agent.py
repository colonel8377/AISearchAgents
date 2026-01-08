"""Demographic Evaluator Agent implementation for evaluating sentences from demographic perspectives."""

import asyncio
import json
import re
from typing import Dict, List, Optional, Any, Union

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI

from src.application.few_shots.demographic_evaluator.few_shots import DEMOGRAPHIC_EVALUATOR_FEW_SHOTS
from src.infrastructure.repositories import AgentProtocol
from src.infrastructure.storage.persistence import get_database
from src.shared.cache import cached
from src.shared.config.settings import settings, ExecutionMode
from src.shared.constant.enums import CoTMode
from src.shared.llm import llm_retry
from src.shared.llm.llm_manager import llm_manager
from src.shared.utils import get_logger
from src.shared.utils.sentence_splitter import split_sentences, clean_text

logger = get_logger(__name__)


class DemographicEvaluatorAgent(AgentProtocol):
    """
    Implements a Demographic Evaluator Agent that evaluates sentences from
    a specific demographic perspective.

    The agent takes a demographic profile (JSON) and a list of sentences,
    then evaluates each sentence from that demographic's perspective, providing
    agree/disagree judgments with reasoning.

    Supports optional Chain of Thought (CoT) reasoning via execution_mode parameter.
    Supports optional few-shot examples for better evaluation quality.
    Supports per-message mode for individual sentence evaluation.
    """

    # Class variable for caching custom few shots (optional performance optimization)
    _custom_few_shots_cache: Optional[str] = None
    
    # Agreement scale descriptions
    AGREEMENT_SCALE_CONTINUOUS = "on a scale from 0.0 (fully disagree) to 1.0 (fully agree)"
    AGREEMENT_SCALE_BINARY = "as 0 (disagree) or 1 (agree)"
    AGREEMENT_OUTPUT_CONTINUOUS = "0.0 to 1.0, // 1 if you fully agree, 0 if you fully disagree"
    AGREEMENT_OUTPUT_BINARY = "0 or 1, // 0 if you disagree, 1 if you agree"

    # Base prompt templates (will be formatted with agreement scale)
    SYSTEM_PROMPT_TEMPLATE_COT = """You are a person with the following demographic profile:
{demography_json}

Your task is to evaluate a list of sentences provided by the user.

RULES:
1. For each sentence, decide your level of agreement {agreement_scale}.
2. Provide detailed step-by-step reasoning from your personal perspective (Chain of Thought).
   - Explain your thought process step by step
   - Show the reasoning chain that leads to your agree/disagree decision
   - Be thorough in explaining why you hold this position
3. STRICT CONSTRAINT: Do not explicitly mention your demographic traits in the reasoning. Speak naturally as that person.
4. Output must be a valid JSON object with a "judgments" key.

OUTPUT SCHEMA:
{{
  "judgments": [
    {{
      "index": Integer,
      "sentence": "Original sentence",
      "agree": {agreement_output},
      "reason": "Your detailed step-by-step reasoning showing your thought process"
    }}
  ]
}}"""
    
    SYSTEM_PROMPT_TEMPLATE_NO_COT = """You are a person with the following demographic profile:
{demography_json}

Your task is to evaluate a list of sentences provided by the user.

RULES:
1. For each sentence, decide your level of agreement {agreement_scale}.
2. Provide a brief reasoning from your personal perspective.
3. STRICT CONSTRAINT: Do not explicitly mention your demographic traits in the reasoning. Speak naturally as that person.
4. Output must be a valid JSON object with a "judgments" key.

OUTPUT SCHEMA:
{{
  "judgments": [
    {{
      "index": Integer, // This index corresponds to the index of the sentence in the input list.
      "sentence": "Original sentence", // the original sentence being evaluated, identified by its index
      "agree": {agreement_output},
      "reason": "Your subjective reasoning"
    }}
  ]
}}"""

    SYSTEM_PROMPT_TEMPLATE_PER_MESSAGE_COT = """You are a person with the following demographic profile:
{demography_json}

Your task is to evaluate the LATEST message sent by the user.

RULES:
1. Decide your level of agreement with the sentence {agreement_scale}.
2. Provide detailed step-by-step reasoning from your personal perspective (Chain of Thought).
   - Explain your thought process step by step
   - Show the reasoning chain that leads to your agree/disagree decision
   - Be thorough in explaining why you hold this position
3. STRICT CONSTRAINT: Do not explicitly mention your demographic traits in the reasoning. Speak naturally as that person.
4. Output must be a valid JSON object.

OUTPUT SCHEMA:
{{
  "agree": {agreement_output},
  "reason": "Your detailed step-by-step reasoning showing your thought process"
}}"""

    SYSTEM_PROMPT_TEMPLATE_PER_MESSAGE_NO_COT = """You are a person with the following demographic profile:
{demography_json}

Your task is to evaluate the LATEST message sent by the user.

RULES:
1. Decide your level of agreement with that message {agreement_scale}.
2. Provide a brief reasoning from your personal perspective.
3. STRICT CONSTRAINT: Do not explicitly mention your demographic traits in the reasoning. Speak naturally as that person.
4. Output must be a valid JSON object.

OUTPUT SCHEMA:
{{
  "agree": {agreement_output},
  "reason": "Your subjective reasoning"
}}"""

    @staticmethod
    def get_default_few_shots() -> str:
        """Get the default few-shot examples for demographic evaluation."""
        return DEMOGRAPHIC_EVALUATOR_FEW_SHOTS

    @classmethod
    def set_custom_few_shots(cls, custom_few_shots: Optional[str] = None) -> None:
        """
        Set custom few-shot examples for demographic evaluation.

        Args:
            custom_few_shots: Custom few-shot examples string. If None, clears custom few shots.
        """
        if settings.enable_persistence:
            database = get_database()
            success = database.save_custom_few_shots("demographic_evaluator", custom_few_shots)
            if success:
                cls._custom_few_shots_cache = custom_few_shots
                logger.info(f"Custom few shots saved for DemographicEvaluatorAgent: {custom_few_shots is not None}")
            else:
                logger.warning("Failed to save custom few shots to persistence")
        else:
            cls._custom_few_shots_cache = custom_few_shots
            logger.info(f"Custom few shots set for DemographicEvaluatorAgent (no persistence): {custom_few_shots is not None}")

    @classmethod
    def get_custom_few_shots(cls) -> Optional[str]:
        """Get currently set custom few-shot examples."""
        if settings.enable_persistence:
            database = get_database()
            few_shots = database.load_custom_few_shots("demographic_evaluator")
            if isinstance(few_shots, str) or few_shots is None:
                cls._custom_few_shots_cache = few_shots
            return few_shots
        else:
            return cls._custom_few_shots_cache

    @classmethod
    def get_effective_few_shots(cls) -> str:
        """Get effective few-shot examples (custom if set, otherwise default)."""
        custom_few_shots = cls.get_custom_few_shots()
        return custom_few_shots if custom_few_shots is not None else cls.get_default_few_shots()
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.7
    ):
        """
        Initialize the DemographicEvaluatorAgent.
        
        Args:
            model_name: Name of the LLM model to use (defaults to settings.openai_model)
            api_key: OpenAI API key or compatible API key
            api_base: Base URL for the API
            temperature: Temperature for LLM responses
        """
        model_name = model_name or settings.openai_model
        api_key = api_key or settings.openai_api_key
        api_base = api_base or settings.openai_api_base

        logger.info(f"Initializing DemographicEvaluatorAgent: model={model_name}, temperature={temperature}")
        
        http_client = llm_manager.get_http_client()
        
        self.llm = ChatOpenAI(
            model_name=model_name,
            api_key=api_key,
            base_url=api_base,
            temperature=temperature,
            max_retries=settings.openai_max_retries,
            timeout=settings.openai_timeout,
            http_client=http_client
        )
        
        # Initialize custom few shots
        self._custom_few_shots = None
        if settings.enable_persistence:
            try:
                database = get_database()
                self._custom_few_shots = database.load_custom_few_shots("demographic_evaluator")
            except Exception as e:
                logger.warning(f"Failed to load custom few shots: {e}")

        logger.debug("DemographicEvaluatorAgent initialized")


    def _normalize_sentences(self, sentences: Union[List[str], str]) -> List[str]:
        """
        Normalize sentences input to a list of strings.

        Args:
            sentences: List of sentences or single string

        Returns:
            List of sentence strings
        """
        if sentences is None:
            logger.warning("Received None for sentences")
            return []

        if isinstance(sentences, str):
            logger.info("Received string input for sentences, splitting into individual sentences")
            sentences = clean_text(sentences)
            return split_sentences(sentences)

        if isinstance(sentences, list):
            if len(sentences) == 0:
                logger.warning("Empty sentences list provided")
                return []
            # Handle edge case where list contains a single string element
            if len(sentences) == 1 and isinstance(sentences[0], str):
                logger.info("Received list with single string, splitting into individual sentences")
                text = clean_text(sentences[0])
                return split_sentences(text)
            return sentences

        logger.warning(f"Unexpected sentences type: {type(sentences)}")
        return []

    @staticmethod
    def _determine_cot_mode(
        use_cot: Union[CoTMode, bool],
        execution_mode: Optional[ExecutionMode]
    ) -> bool:
        """
        Determine if Chain of Thought should be enabled.

        Args:
            use_cot: Chain of Thought mode or boolean
            execution_mode: Execution mode (overrides use_cot if provided)

        Returns:
            True if CoT should be enabled, False otherwise
        """
        if execution_mode is not None:
            effective_cot = execution_mode in ("chain_local", "chain_online")
            logger.debug(f"Using execution_mode={execution_mode}, CoT enabled={effective_cot}")
            return effective_cot

        if isinstance(use_cot, CoTMode):
            effective_cot = use_cot.value in ("chain_local", "chain_online")
            logger.debug(f"Using CoTMode={use_cot.value}, CoT enabled={effective_cot}")
            return effective_cot

        effective_cot = bool(use_cot)
        logger.debug(f"Using use_cot={use_cot}, CoT enabled={effective_cot}")
        return effective_cot

    def _get_few_shots(
        self,
        use_few_shots: bool,
        custom_few_shots: Optional[str]
    ) -> str:
        """
        Get few-shot examples to use.

        Args:
            use_few_shots: Whether to use few-shot examples
            custom_few_shots: Optional custom few-shot examples

        Returns:
            Few-shot examples string (empty if disabled)
        """
        if custom_few_shots is not None and not use_few_shots:
            raise ValueError("If custom_few_shots is provided, use_few_shots must be True")

        if not use_few_shots:
            logger.info("Few-shot examples disabled")
            return ""

        if custom_few_shots is not None:
            logger.info("Using explicitly provided custom few-shot examples")
            return custom_few_shots

        if self._custom_few_shots is not None:
            logger.info("Using stored custom few-shot examples")
            return self._custom_few_shots

        logger.info("Using default few-shot examples")
        return DEMOGRAPHIC_EVALUATOR_FEW_SHOTS

    def _build_system_prompt(
        self,
        demography_json: Dict[str, Any],
        use_cot: bool,
        per_message: bool,
        few_shots: str,
        is_binary_agreement: bool,
    ) -> str:
        """
        Build system prompt for LLM.

        Args:
            demography_json: Demographic profile dictionary
            use_cot: Whether to use Chain of Thought
            per_message: Whether in per-message mode
            few_shots: Few-shot examples string
            is_binary_agreement: If True, only return 0 or 1 (binary), otherwise return 0.0 to 1.0 (continuous)

        Returns:
            Complete system prompt string
        """
        demography_str = json.dumps(demography_json, indent=2)
        
        # Select agreement scale and output format
        agreement_scale = self.AGREEMENT_SCALE_BINARY if is_binary_agreement else self.AGREEMENT_SCALE_CONTINUOUS
        agreement_output = self.AGREEMENT_OUTPUT_BINARY if is_binary_agreement else self.AGREEMENT_OUTPUT_CONTINUOUS

        # Select template based on mode
        if per_message:
            if use_cot:
                template = self.SYSTEM_PROMPT_TEMPLATE_PER_MESSAGE_COT
            else:
                template = self.SYSTEM_PROMPT_TEMPLATE_PER_MESSAGE_NO_COT
        else:
            if use_cot:
                template = self.SYSTEM_PROMPT_TEMPLATE_COT
            else:
                template = self.SYSTEM_PROMPT_TEMPLATE_NO_COT

        # Format template with all variables
        base_prompt = template.format(
            demography_json=demography_str,
            agreement_scale=agreement_scale,
            agreement_output=agreement_output
        )

        if few_shots:
            return f"{base_prompt}\n\n{few_shots}"
        return base_prompt

    @staticmethod
    def _clean_json_response(response_text: str) -> str:
        """
        Clean JSON response text by removing markdown code blocks.

        Args:
            response_text: Raw response text from LLM

        Returns:
            Cleaned JSON string
        """
        response_text = response_text.strip()

        if "```" in response_text:
            code_block_match = re.search(r"```(?:json)?\s*(.*?)\s*```", response_text, re.DOTALL)
            if code_block_match:
                response_text = code_block_match.group(1)
            else:
                response_text = response_text.replace("```json", "").replace("```", "")

        return response_text

    @staticmethod
    def _parse_json_with_fallback(response_text: str, context: str = "") -> Dict[str, Any]:
        """
        Parse JSON response with fallback regex extraction.

        Args:
            response_text: Response text from LLM
            context: Context string for error messages

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If JSON cannot be parsed
        """
        try:
            cleaned_text = DemographicEvaluatorAgent._clean_json_response(response_text)
            return json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed{context}: {e}. Attempting regex extraction.")
            json_match = re.search(r'\{.*?}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except Exception as inner_e:
                    logger.error(f"Failed to parse extracted JSON{context}: {inner_e}")
                    raise ValueError(f"Invalid JSON response from LLM{context}: {e}")
            else:
                logger.error(f"No JSON object found in response{context}: {response_text}")
                raise ValueError(f"Invalid JSON response from LLM{context}: {e}")
    
    @llm_retry
    async def _evaluate_single_sentence(
        self,
        sentence: str,
        index: int,
        system_prompt: str,
        conversation_history: Optional[List[BaseMessage]] = None,
        is_binary_agreement: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate a single sentence in per-message mode.

        Args:
            sentence: Sentence to evaluate
            index: Index of the sentence
            system_prompt: System prompt for LLM
            conversation_history: Optional list of previous messages (SystemMessage and HumanMessage)

        Returns:
            Judgment dictionary with index, sentence, agree, and reason

        Raises:
            ValueError: If LLM response is invalid
        """
        # Build messages list with conversation history
        messages = []
        
        # Always add system prompt first
        messages.append(SystemMessage(content=system_prompt))
        
        # Add conversation history (all previous messages)
        # Filter out any SystemMessage from history to avoid duplicates (we use our own system_prompt)
        if conversation_history:
            for msg in conversation_history:
                # Skip SystemMessage in history since we already have our system_prompt
                if not isinstance(msg, SystemMessage):
                    messages.append(msg)
        
        # Add current sentence as user message
        messages.append(HumanMessage(content=sentence))

        logger.info(f"Calling LLM for sentence {index}: {sentence[:50]}...")

        response = await self.llm.ainvoke(messages)
        response_text = response.content

        logger.debug(f"LLM response for sentence {index}: {len(response_text)} characters")

        result = self._parse_json_with_fallback(response_text, f" for sentence {index}")

        # Build judgment dictionary for normalization
        # Note: sentence is provided separately, not from LLM response
        judgment_dict = {
            "sentence": sentence,  # Use the provided sentence, not from LLM
            "agree": result.get("agree"),
            "reason": result.get("reason")
        }
        
        # Normalize using the same helper method for consistency
        normalized_judgment = self._normalize_judgment(judgment_dict, index, is_binary_agreement)
        
        return normalized_judgment

    async def _evaluate_per_message(
        self,
        sentences: List[str],
        demography_json: Dict[str, Any],
        use_cot: bool,
        few_shots: str,
        is_binary_agreement: bool,
    ) -> Dict[str, Any]:
        """
        Evaluate sentences in per-message mode (one LLM call per sentence).
        
        Optimized to run all evaluations in parallel by building each task with
        progressively longer conversation history. Each task includes all previous
        sentences up to its index, allowing parallel execution while maintaining context.

        Args:
            sentences: List of sentences to evaluate
            demography_json: Demographic profile dictionary
            use_cot: Whether to use Chain of Thought
            few_shots: Few-shot examples string

        Returns:
            Dictionary with judgments list
        """
        system_prompt = self._build_system_prompt(demography_json, use_cot, per_message=True, few_shots=few_shots, is_binary_agreement=is_binary_agreement)
        
        # Create tasks with progressively longer conversation history
        # Each task includes all previous sentences before its index
        tasks = []
        for index, sentence in enumerate(sentences):
            # Build conversation history for this specific task
            # Include all sentences before current index
            task_history = []
            for prev_index in range(index):
                task_history.append(HumanMessage(content=sentences[prev_index]))
            
            # Create task with its own conversation history
            task = self._evaluate_single_sentence(sentence, index, system_prompt, task_history, is_binary_agreement)
            tasks.append(task)
        
        logger.info(f"Parallelizing {len(sentences)} sentence evaluations with conversation context")
        
        # Execute all tasks in parallel
        try:
            judgments = await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Failed during parallel evaluation: {e}", exc_info=True)
            raise
        
        # Sort by index to ensure order (in case tasks complete out of order)
        judgments = sorted(judgments, key=lambda x: x["index"])
        
        logger.info(f"Successfully evaluated {len(judgments)} sentences in per-message mode (parallel)")
        return {"judgments": judgments}

    def _build_batch_user_message(self, sentences: List[str], use_cot: bool) -> str:
        """
        Build user message for batch evaluation mode.

        Args:
            sentences: List of sentences to evaluate
            use_cot: Whether to use Chain of Thought

        Returns:
            User message string
        """
        sentences_text = "\n".join(f"Index {i}. {sentence}" for i, sentence in enumerate(sentences))

        if use_cot:
            return f"""Please evaluate the following sentences.
For each sentence, provide detailed step-by-step reasoning showing your thought process:
1. Consider the sentence in context of your values and experiences
2. Think through how your background influences your perspective
3. Explain the reasoning chain that leads to your agree/disagree decision

SENTENCES TO EVALUATE:
{sentences_text}

Return your evaluation as a JSON object."""

        return f"Please evaluate the following sentences:\n\n{sentences_text}"

    def _normalize_judgment(self, judgment: Dict[str, Any], index: int, is_binary_agreement: bool = False) -> Dict[str, Any]:
        """
        Normalize a judgment dictionary to ensure consistent format.

        Args:
            judgment: Judgment dictionary from LLM response
            index: Index of the judgment (used as fallback if not in judgment)

        Returns:
            Normalized judgment dictionary with consistent types

        Raises:
            ValueError: If judgment structure is invalid
        """
        normalized = {}
        
        # Ensure index is present and is an integer
        index_value = judgment.get("index", index)
        try:
            # Explicitly convert to int, handling edge cases
            if index_value is None:
                normalized["index"] = int(index)
            elif isinstance(index_value, (int, float)):
                normalized["index"] = int(index_value)
            elif isinstance(index_value, str):
                # Try to parse as int, but log warning if it looks like a datetime
                if any(char in index_value for char in ['-', ':', ' ']) and len(index_value) > 10:
                    logger.warning(f"Unexpected datetime string '{index_value}' found for index, using fallback index {index}")
                    normalized["index"] = int(index)
                else:
                    normalized["index"] = int(float(index_value))  # Handle "1.0" -> 1
            else:
                logger.warning(f"Unexpected index type {type(index_value)}, using fallback index {index}")
                normalized["index"] = int(index)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to convert index '{index_value}' to int: {e}, using fallback index {index}")
            normalized["index"] = int(index)
        
        # Ensure sentence is present
        if "sentence" not in judgment:
            raise ValueError(f"Judgment {index} missing 'sentence' key")
        normalized["sentence"] = str(judgment["sentence"])
        
        # Validate and normalize agree value
        if "agree" not in judgment:
            raise ValueError(f"Judgment {index} missing 'agree' key")
        
        agree_value = judgment["agree"]
        if not isinstance(agree_value, (int, float)):
            raise ValueError(f"Judgment {index} 'agree' must be a number, got {type(agree_value)}")
        
        agree_float = float(agree_value)
        
        if is_binary_agreement:
            # Binary mode: only accept 0 or 1
            if agree_float not in [0.0, 1.0]:
                raise ValueError(f"Judgment {index} 'agree' must be 0 or 1 (binary mode), got {agree_float}")
            normalized["agree"] = int(agree_float)  # Return as int (0 or 1)
        else:
            # Continuous mode: accept 0.0 to 1.0
            if not (0.0 <= agree_float <= 1.0):
                raise ValueError(f"Judgment {index} 'agree' must be between 0.0 and 1.0, got {agree_float}")
            normalized["agree"] = agree_float  # Return as float
        
        # Ensure reason is present
        if "reason" not in judgment:
            raise ValueError(f"Judgment {index} missing 'reason' key")
        normalized["reason"] = str(judgment["reason"])
        
        return normalized

    def _validate_batch_response(self, result: Dict[str, Any], is_binary_agreement: bool = False) -> None:
        """
        Validate batch mode response structure.

        Args:
            result: Parsed JSON response
            is_binary_agreement: If True, validates binary agreement (0 or 1). If False, validates continuous (0.0 to 1.0).

        Raises:
            ValueError: If response structure is invalid
        """
        if "judgments" not in result:
            raise ValueError("Response missing 'judgments' key")

        if not isinstance(result["judgments"], list):
            raise ValueError("'judgments' must be a list")
        
        # Note: Actual agree value validation is done in _normalize_judgment

    @llm_retry
    async def _evaluate_batch(
        self,
        sentences: List[str],
        demography_json: Dict[str, Any],
        use_cot: bool,
        few_shots: str,
        is_binary_agreement: bool,
    ) -> Dict[str, Any]:
        """
        Evaluate sentences in batch mode (single LLM call with all sentences).

        Args:
            sentences: List of sentences to evaluate
            demography_json: Demographic profile dictionary
            use_cot: Whether to use Chain of Thought
            few_shots: Few-shot examples string

        Returns:
            Dictionary with judgments list
        """
        system_prompt = self._build_system_prompt(demography_json, use_cot, per_message=False, few_shots=few_shots, is_binary_agreement=is_binary_agreement)
        user_message = self._build_batch_user_message(sentences, use_cot)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]

        logger.debug(f"Calling LLM with {len(sentences)} sentences to evaluate")

        response = await self.llm.ainvoke(messages)
        response_text = response.content

        logger.info(f"LLM response generated: {len(response_text)} characters")

        result = self._parse_json_with_fallback(response_text)
        self._validate_batch_response(result, is_binary_agreement)

        # Normalize all judgments to ensure consistent format
        normalized_judgments = []
        for i, judgment in enumerate(result["judgments"]):
            normalized_judgment = self._normalize_judgment(judgment, i, is_binary_agreement)
            normalized_judgments.append(normalized_judgment)

        logger.info(f"Successfully parsed {len(normalized_judgments)} judgments")
        return {"judgments": normalized_judgments}

    async def evaluate_sentences(
        self,
        demography_json: Dict[str, Any],
        sentences: Union[List[str], str],
        use_cot: Union[CoTMode, bool] = CoTMode.NO_CHAIN,
        execution_mode: Optional[ExecutionMode] = None,
        use_few_shots: bool = True,
        custom_few_shots: Optional[str] = None,
        per_message: bool = False,
        is_binary_agreement: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate a list of sentences from a demographic perspective.

        Args:
            demography_json: Dictionary containing demographic profile information
            sentences: List of sentences to evaluate (or single string)
            use_cot: Chain of Thought mode ('chain_online', 'chain_local', 'no_chain') or boolean
            execution_mode: Execution mode for CoT ('chain_online', 'chain_local', 'no_chain').
                          If provided, overrides use_cot parameter.
                          - 'chain_online' or 'chain_local': Enable CoT reasoning
                          - 'no_chain': Disable CoT reasoning
            use_few_shots: Whether to include few-shot examples in the prompt (default: True)
            custom_few_shots: Optional custom few-shot examples to use instead of defaults.
                             If provided, use_few_shots must be True
            per_message: If True, evaluate each sentence as a separate user message (one LLM call per sentence).
                        When True, index and sentence are calculated in Python, LLM only provides agree/disagree and reasoning.
            is_binary_agreement: If True, only return 0 or 1 (binary agreement). If False (default), return 0.0 to 1.0 (continuous scale).

        Returns:
            Dictionary containing judgments with the following structure:
            {
                "judgments": [
                    {
                        "index": int,
                        "sentence": str,
                        "agree": float (0.0 to 1.0),
                        "reason": str
                    }
                ]
            }
        """
        # Normalize input
        sentences_list = self._normalize_sentences(sentences)
        if not sentences_list:
            return {"judgments": []}

        # Determine evaluation parameters
        effective_cot = self._determine_cot_mode(use_cot, execution_mode)
        few_shots = self._get_few_shots(use_few_shots, custom_few_shots)

        logger.info(
            f"Evaluating {len(sentences_list)} sentences for demographic profile "
            f"(CoT={effective_cot}, use_few_shots={use_few_shots}, per_message={per_message})"
        )

        # Route to appropriate evaluation method
        try:
            if per_message:
                return await self._evaluate_per_message(
                    sentences_list, demography_json, effective_cot, few_shots, is_binary_agreement
                )
            else:
                return await self._evaluate_batch(sentences_list, demography_json, effective_cot, few_shots, is_binary_agreement)
        except Exception as e:
            logger.error(f"Failed to evaluate sentences: {e}", exc_info=True)
            raise
    
    def reset(self) -> None:
        """Reset the agent to initial state."""
        logger.info("Resetting DemographicEvaluatorAgent state")
        # No state to reset for this agent (stateless agent)
        logger.debug("Agent reset complete")