"""Demographic Evaluator Agent implementation for evaluating sentences from demographic perspectives."""

import json
from typing import Dict, List, Optional, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ...utils.logger import get_logger
from ...config.settings import settings, ExecutionMode
from ...utils.llm_client import llm_manager
from ...utils.agent_cache import cached
from ...prompts.demographic_evaluator.few_shots import DEMOGRAPHIC_EVALUATOR_FEW_SHOTS

logger = get_logger(__name__)


class DemographicEvaluatorAgent:
    """
    Implements a Demographic Evaluator Agent that evaluates sentences from
    a specific demographic perspective.

    The agent takes a demographic profile (JSON) and a list of sentences,
    then evaluates each sentence from that demographic's perspective, providing
    agree/disagree judgments with reasoning.

    Supports optional Chain of Thought (CoT) reasoning via execution_mode parameter.
    Supports optional few-shot examples for better evaluation quality.
    """

    # Class variable to store custom few shots (persistent across instances)
    _custom_few_shots: Optional[str] = None
    
    @staticmethod
    def get_default_few_shots() -> str:
        """
        Get the default few-shot examples for demographic evaluation.

        Returns:
            str: Default few-shot examples
        """
        return DEMOGRAPHIC_EVALUATOR_FEW_SHOTS

    @classmethod
    def set_custom_few_shots(cls, custom_few_shots: Optional[str] = None) -> None:
        """
        Set custom few-shot examples for demographic evaluation.

        Args:
            custom_few_shots: Custom few-shot examples string. If None, clears custom few shots.
        """
        cls._custom_few_shots = custom_few_shots
        logger.info(f"Custom few shots set for DemographicEvaluatorAgent: {custom_few_shots is not None}")

    @classmethod
    def get_custom_few_shots(cls) -> Optional[str]:
        """
        Get currently set custom few-shot examples.

        Returns:
            Custom few-shot examples string or None if not set
        """
        return cls._custom_few_shots

    @classmethod
    def get_effective_few_shots(cls) -> str:
        """
        Get effective few-shot examples (custom if set, otherwise default).

        Returns:
            Effective few-shot examples string
        """
        return cls._custom_few_shots if cls._custom_few_shots is not None else cls.get_default_few_shots()
    
    SYSTEM_PROMPT_TEMPLATE_COT = """You are a person with the following demographic profile:
{demography_json}

Your task is to evaluate a list of sentences provided by the user.

RULES:
1. For each sentence, decide if you agree (1) or disagree (0).
2. Provide detailed step-by-step reasoning from your personal perspective (Chain of Thought).
   - Explain your thought process step by step
   - Consider how your background, experiences, and values influence your judgment
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
      "agree": 1 or 0,
      "reason": "Your detailed step-by-step reasoning showing your thought process"
    }}
  ]
}}"""
    
    SYSTEM_PROMPT_TEMPLATE_NO_COT = """You are a person with the following demographic profile:
{demography_json}

Your task is to evaluate a list of sentences provided by the user.

RULES:
1. For each sentence, decide if you agree (1) or disagree (0).
2. Provide a brief reasoning from your personal perspective.
3. STRICT CONSTRAINT: Do not explicitly mention your demographic traits in the reasoning. Speak naturally as that person.
4. Output must be a valid JSON object with a "judgments" key.

OUTPUT SCHEMA:
{{
  "judgments": [
    {{
      "index": Integer,
      "sentence": "Original sentence",
      "agree": 1 or 0,
      "reason": "Your subjective reasoning"
    }}
  ]
}}"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        proxy: Optional[str] = None
    ):
        """
        Initialize the DemographicEvaluatorAgent.
        
        Args:
            model_name: Name of the LLM model to use (defaults to settings.openai_model)
            api_key: OpenAI API key or compatible API key
            api_base: Base URL for the API
            temperature: Temperature for LLM responses
            proxy: Optional HTTP proxy for API requests
        """
        model_name = model_name or settings.openai_model
        logger.info(f"Initializing DemographicEvaluatorAgent: model={model_name}, temperature={temperature}")
        
        # Use shared HTTP client for better connection pooling and performance
        http_client = llm_manager.get_http_client(proxy=proxy)
        
        self.llm = ChatOpenAI(
            model_name=model_name,
            api_key=api_key,
            base_url=api_base,
            temperature=temperature,
            max_retries=settings.openai_max_retries,
            timeout=settings.openai_timeout,
            http_client=http_client
        )
        
        logger.debug("DemographicEvaluatorAgent initialized")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception)
    )
    @cached(exclude_class_name=True)
    def evaluate_sentences(
        self,
        demography_json: Dict[str, Any],
        sentences: List[str],
        use_cot: bool = False,
        execution_mode: Optional[ExecutionMode] = None,
        use_few_shots: bool = True,
        custom_few_shots: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a list of sentences from a demographic perspective.
        
        Args:
            demography_json: Dictionary containing demographic profile information
            sentences: List of sentences to evaluate
            use_cot: Whether to use Chain of Thought reasoning (default: False)
            execution_mode: Execution mode for CoT ('chain_online', 'chain_local', 'no_chain').
                          If provided, overrides use_cot parameter.
                          - 'chain_online' or 'chain_local': Enable CoT reasoning
                          - 'no_chain': Disable CoT reasoning
            use_few_shots: Whether to include few-shot examples in the prompt (default: True)
            custom_few_shots: Optional custom few-shot examples to use instead of defaults.
                             If provided, use_few_shots must be True
            
        Returns:
            Dictionary containing judgments with the following structure:
            {
                "judgments": [
                    {
                        "index": int,
                        "sentence": str,
                        "agree": int (1 or 0),
                        "reason": str
                    }
                ]
            }
        """
        # Determine if CoT should be used
        if execution_mode is not None:
            use_cot = execution_mode in ("chain_local", "chain_online")
            logger.debug(f"Using execution_mode={execution_mode}, CoT enabled={use_cot}")
        else:
            logger.debug(f"Using use_cot={use_cot}")
        
        # Validate few-shots parameters
        if custom_few_shots is not None and not use_few_shots:
            raise ValueError("If custom_few_shots is provided, use_few_shots must be True")
        
        # Determine which few-shots to use
        few_shots = ""
        if use_few_shots:
            if custom_few_shots is not None:
                # Use explicitly provided custom few shots
                few_shots = custom_few_shots
                logger.info("Using explicitly provided custom few-shot examples")
            elif self._custom_few_shots is not None:
                # Use stored custom few shots
                few_shots = self._custom_few_shots
                logger.info("Using stored custom few-shot examples")
            else:
                # Use default few shots
                few_shots = DEMOGRAPHIC_EVALUATOR_FEW_SHOTS
                logger.info("Using default few-shot examples")
        else:
            logger.info("Few-shot examples disabled")
        
        logger.info(f"Evaluating {len(sentences)} sentences for demographic profile (CoT={use_cot}, use_few_shots={use_few_shots})")
        
        # Convert demography_json to formatted string
        demography_str = json.dumps(demography_json, indent=2)
        
        # Select base system prompt based on CoT setting
        if use_cot:
            base_system_prompt = self.SYSTEM_PROMPT_TEMPLATE_COT.format(
                demography_json=demography_str
            )
        else:
            base_system_prompt = self.SYSTEM_PROMPT_TEMPLATE_NO_COT.format(
                demography_json=demography_str
            )
        
        # Append few-shots if enabled
        if few_shots:
            system_prompt = f"{base_system_prompt}\n\n{few_shots}"
        else:
            system_prompt = base_system_prompt
        
        # Build user message with sentences
        sentences_text = "\n".join(f"{i+1}. {sentence}" for i, sentence in enumerate(sentences))
        
        if use_cot:
            user_message = f"""Please evaluate the following sentences.
For each sentence, provide detailed step-by-step reasoning showing your thought process:
1. Consider the sentence in context of your values and experiences
2. Think through how your background influences your perspective
3. Explain the reasoning chain that leads to your agree/disagree decision

SENTENCES TO EVALUATE:
{sentences_text}

Return your evaluation as a JSON object."""
        else:
            user_message = f"Please evaluate the following sentences:\n\n{sentences_text}"
        
        try:
            # Build messages for the LLM
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]
            
            logger.debug(f"Calling LLM with {len(sentences)} sentences to evaluate")
            
            # Generate response
            response = self.llm.invoke(messages)
            response_text = response.content
            
            logger.info(f"LLM response generated: {len(response_text)} characters")
            
            # Parse JSON response
            try:
                # Try to extract JSON from the response (in case it's wrapped in markdown)
                response_text = response_text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                result = json.loads(response_text)
                
                # Validate structure
                if "judgments" not in result:
                    raise ValueError("Response missing 'judgments' key")
                
                if not isinstance(result["judgments"], list):
                    raise ValueError("'judgments' must be a list")
                
                # Validate each judgment
                for i, judgment in enumerate(result["judgments"]):
                    if "index" not in judgment:
                        judgment["index"] = i
                    if "sentence" not in judgment:
                        raise ValueError(f"Judgment {i} missing 'sentence' key")
                    if "agree" not in judgment:
                        raise ValueError(f"Judgment {i} missing 'agree' key")
                    if judgment["agree"] not in [0, 1]:
                        raise ValueError(f"Judgment {i} 'agree' must be 0 or 1, got {judgment['agree']}")
                    if "reason" not in judgment:
                        raise ValueError(f"Judgment {i} missing 'reason' key")
                
                logger.info(f"Successfully parsed {len(result['judgments'])} judgments")
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.debug(f"Response text: {response_text}")
                raise ValueError(f"Invalid JSON response from LLM: {e}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate sentences: {e}", exc_info=True)
            raise
    
    def evaluate_sentences_from_json_str(
        self,
        demography_json_str: str,
        sentences: List[str],
        use_cot: bool = False,
        execution_mode: Optional[ExecutionMode] = None,
        use_few_shots: bool = True,
        custom_few_shots: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate sentences using a JSON string for demographic profile.
        
        Args:
            demography_json_str: JSON string containing demographic profile
            sentences: List of sentences to evaluate
            use_cot: Whether to use Chain of Thought reasoning (default: False)
            execution_mode: Execution mode for CoT ('chain_online', 'chain_local', 'no_chain').
                          If provided, overrides use_cot parameter.
            use_few_shots: Whether to include few-shot examples in the prompt (default: True)
            custom_few_shots: Optional custom few-shot examples to use instead of defaults.
            
        Returns:
            Dictionary containing judgments
        """
        demography_json = json.loads(demography_json_str)
        return self.evaluate_sentences(
            demography_json, 
            sentences, 
            use_cot=use_cot, 
            execution_mode=execution_mode,
            use_few_shots=use_few_shots,
            custom_few_shots=custom_few_shots
        )

