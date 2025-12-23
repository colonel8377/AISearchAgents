"""Example usage of the Demographic Evaluator Agent with optional Chain of Thought (CoT) support."""

import json
from src.agents.demographic_evaluator import DemographicEvaluatorAgent
from src.config.settings import settings


def example_basic():
    """Example 1: Basic evaluation without Chain of Thought."""
    print("="*80)
    print("EXAMPLE 1: Basic Evaluation (No CoT)")
    print("="*80)
    
    # Initialize the agent
    agent = DemographicEvaluatorAgent(
        model_name=settings.openai_model,
        api_key=settings.openai_api_key,
        api_base=settings.openai_api_base,
        temperature=0.7
    )
    
    # Define a demographic profile
    demography_json = {
        "age": 35,
        "gender": "female",
        "education": "college_graduate",
        "political_affiliation": "progressive",
        "location": "urban",
        "occupation": "teacher",
        "income_level": "middle_class",
        "cultural_background": "hispanic"
    }
    
    # Define sentences to evaluate
    sentences = [
        "Public schools should receive more funding for arts and music programs.",
        "Immigration policies should prioritize family reunification.",
        "Climate change requires immediate government action."
    ]
    
    print("\nDemographic Profile:")
    print(json.dumps(demography_json, indent=2))
    print("\nEvaluating sentences (without CoT)...\n")
    
    # Evaluate sentences without CoT (with few shots by default)
    try:
        result = agent.evaluate_sentences(
            demography_json=demography_json,
            sentences=sentences,
            use_cot=False,
            use_few_shots=True  # Use default few shots
        )
        
        # Display results
        print("Evaluation Results:")
        print("-"*80)
        for judgment in result["judgments"]:
            agree_status = "AGREE" if judgment["agree"] == 1 else "DISAGREE"
            print(f"\n[{judgment['index'] + 1}] {agree_status}")
            print(f"Sentence: {judgment['sentence']}")
            print(f"Reason: {judgment['reason']}")
            print("-" * 80)
        
        # Summary statistics
        agree_count = sum(1 for j in result["judgments"] if j["agree"] == 1)
        disagree_count = len(result["judgments"]) - agree_count
        print(f"\nSummary: {agree_count} agree, {disagree_count} disagree out of {len(result['judgments'])} sentences")
        
    except Exception as e:
        print(f"Error evaluating sentences: {e}")
        raise


def example_with_cot():
    """Example 2: Evaluation with Chain of Thought reasoning."""
    print("\n\n" + "="*80)
    print("EXAMPLE 2: Evaluation with Chain of Thought (CoT)")
    print("="*80)
    
    # Initialize the agent
    agent = DemographicEvaluatorAgent(
        model_name=settings.openai_model,
        api_key=settings.openai_api_key,
        api_base=settings.openai_api_base,
        temperature=0.7
    )
    
    # Define a demographic profile
    demography_json = {
        "age": 28,
        "gender": "male",
        "education": "high_school",
        "political_affiliation": "conservative",
        "location": "rural",
        "occupation": "farmer",
        "income_level": "working_class",
        "cultural_background": "white"
    }
    
    # Define sentences to evaluate
    sentences = [
        "Healthcare should be a universal right, not a privilege.",
        "Tax cuts for corporations benefit the economy.",
        "Gun ownership is a fundamental right protected by the Constitution."
    ]
    
    print("\nDemographic Profile:")
    print(json.dumps(demography_json, indent=2))
    print("\nEvaluating sentences (with CoT)...\n")
    
    # Evaluate sentences with CoT (with few shots by default)
    try:
        result = agent.evaluate_sentences(
            demography_json=demography_json,
            sentences=sentences,
            use_cot=True,
            use_few_shots=True  # Use default few shots
        )
        
        # Display results
        print("Evaluation Results (with detailed reasoning):")
        print("-"*80)
        for judgment in result["judgments"]:
            agree_status = "AGREE" if judgment["agree"] == 1 else "DISAGREE"
            print(f"\n[{judgment['index'] + 1}] {agree_status}")
            print(f"Sentence: {judgment['sentence']}")
            print(f"Detailed Reasoning: {judgment['reason']}")
            print("-" * 80)
        
        # Summary statistics
        agree_count = sum(1 for j in result["judgments"] if j["agree"] == 1)
        disagree_count = len(result["judgments"]) - agree_count
        print(f"\nSummary: {agree_count} agree, {disagree_count} disagree out of {len(result['judgments'])} sentences")
        
    except Exception as e:
        print(f"Error evaluating sentences: {e}")
        raise


def example_with_execution_mode():
    """Example 3: Evaluation using execution_mode parameter."""
    print("\n\n" + "="*80)
    print("EXAMPLE 3: Evaluation with execution_mode parameter")
    print("="*80)
    
    # Initialize the agent
    agent = DemographicEvaluatorAgent(
        model_name=settings.openai_model,
        api_key=settings.openai_api_key,
        api_base=settings.openai_api_base,
        temperature=0.7
    )
    
    # Define a demographic profile
    demography_json = {
        "age": 45,
        "gender": "non_binary",
        "education": "graduate_degree",
        "political_affiliation": "independent",
        "location": "suburban",
        "occupation": "engineer",
        "income_level": "upper_middle_class",
        "cultural_background": "asian_american"
    }
    
    # Define sentences to evaluate
    sentences = [
        "Renewable energy should replace fossil fuels within the next decade.",
        "Social media platforms should have stricter content moderation policies."
    ]
    
    print("\nDemographic Profile:")
    print(json.dumps(demography_json, indent=2))
    print("\nEvaluating sentences with execution_mode='chain_local'...\n")
    
    # Evaluate sentences using execution_mode
    try:
        result = agent.evaluate_sentences(
            demography_json=demography_json,
            sentences=sentences,
            execution_mode="chain_local"  # This enables CoT
        )
        
        # Display results
        print("Evaluation Results:")
        print("-"*80)
        for judgment in result["judgments"]:
            agree_status = "AGREE" if judgment["agree"] == 1 else "DISAGREE"
            print(f"\n[{judgment['index'] + 1}] {agree_status}")
            print(f"Sentence: {judgment['sentence']}")
            print(f"Reasoning: {judgment['reason']}")
            print("-" * 80)
        
    except Exception as e:
        print(f"Error evaluating sentences: {e}")
        raise


def example_with_custom_few_shots():
    """Example 4: Evaluation with custom few-shot examples."""
    print("\n\n" + "="*80)
    print("EXAMPLE 4: Evaluation with Custom Few-Shot Examples")
    print("="*80)
    
    # Initialize the agent
    agent = DemographicEvaluatorAgent(
        model_name=settings.openai_model,
        api_key=settings.openai_api_key,
        api_base=settings.openai_api_base,
        temperature=0.7
    )
    
    # Define a demographic profile
    demography_json = {
        "age": 30,
        "gender": "male",
        "education": "college_graduate",
        "political_affiliation": "moderate",
        "location": "suburban",
        "occupation": "marketing_manager",
        "income_level": "upper_middle_class",
        "cultural_background": "white"
    }
    
    # Define sentences to evaluate
    sentences = [
        "Remote work should become the standard for office jobs.",
        "Social media companies should be regulated more strictly."
    ]
    
    # Custom few-shot examples
    custom_few_shots = """EXAMPLE:
Demographic Profile:
{
  "age": 30,
  "occupation": "marketing_manager",
  "location": "suburban"
}

Sentence: "Work-life balance is important for employee well-being."
Evaluation: {
  "index": 0,
  "sentence": "Work-life balance is important for employee well-being.",
  "agree": 1,
  "reason": "Having worked in corporate environments, I've seen how burnout affects productivity and mental health. A healthy work-life balance isn't just nice to have - it's essential for long-term success."
}"""
    
    print("\nDemographic Profile:")
    print(json.dumps(demography_json, indent=2))
    print("\nEvaluating sentences with custom few-shot examples...\n")
    
    try:
        result = agent.evaluate_sentences(
            demography_json=demography_json,
            sentences=sentences,
            use_few_shots=True,
            custom_few_shots=custom_few_shots
        )
        
        # Display results
        print("Evaluation Results:")
        print("-"*80)
        for judgment in result["judgments"]:
            agree_status = "AGREE" if judgment["agree"] == 1 else "DISAGREE"
            print(f"\n[{judgment['index'] + 1}] {agree_status}")
            print(f"Sentence: {judgment['sentence']}")
            print(f"Reasoning: {judgment['reason']}")
            print("-" * 80)
        
    except Exception as e:
        print(f"Error evaluating sentences: {e}")
        raise


def example_without_few_shots():
    """Example 5: Evaluation without few-shot examples."""
    print("\n\n" + "="*80)
    print("EXAMPLE 5: Evaluation without Few-Shot Examples")
    print("="*80)
    
    # Initialize the agent
    agent = DemographicEvaluatorAgent(
        model_name=settings.openai_model,
        api_key=settings.openai_api_key,
        api_base=settings.openai_api_base,
        temperature=0.7
    )
    
    # Define a demographic profile
    demography_json = {
        "age": 25,
        "gender": "female",
        "education": "some_college",
        "political_affiliation": "progressive",
        "location": "urban",
        "occupation": "barista",
        "income_level": "low_income",
        "cultural_background": "mixed"
    }
    
    # Define sentences to evaluate
    sentences = [
        "Minimum wage should be increased to $20 per hour."
    ]
    
    print("\nDemographic Profile:")
    print(json.dumps(demography_json, indent=2))
    print("\nEvaluating sentences without few-shot examples...\n")
    
    try:
        result = agent.evaluate_sentences(
            demography_json=demography_json,
            sentences=sentences,
            use_few_shots=False  # Disable few shots
        )
        
        # Display results
        print("Evaluation Results:")
        print("-"*80)
        for judgment in result["judgments"]:
            agree_status = "AGREE" if judgment["agree"] == 1 else "DISAGREE"
            print(f"\n[{judgment['index'] + 1}] {agree_status}")
            print(f"Sentence: {judgment['sentence']}")
            print(f"Reasoning: {judgment['reason']}")
            print("-" * 80)
        
    except Exception as e:
        print(f"Error evaluating sentences: {e}")
        raise


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("Demographic Evaluator Agent Examples")
    print("="*80)
    print("\nThis example demonstrates:")
    print("1. Basic evaluation without Chain of Thought (with few shots)")
    print("2. Evaluation with Chain of Thought (detailed reasoning)")
    print("3. Evaluation using execution_mode parameter")
    print("4. Evaluation with custom few-shot examples")
    print("5. Evaluation without few-shot examples")
    print("\n" + "="*80)
    
    try:
        example_basic()
        example_with_cot()
        example_with_execution_mode()
        example_with_custom_few_shots()
        example_without_few_shots()
        
        print("\n\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        raise


if __name__ == "__main__":
    main()

