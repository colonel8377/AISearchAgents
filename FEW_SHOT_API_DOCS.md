# Few-Shot Control API Documentation

## Overview

All agents now support flexible few-shot prompt control, allowing users to:
1. **Get default few-shot examples** via static methods
2. **Enable/disable few-shots** via `use_few_shots` parameter
3. **Provide custom few-shot examples** via `custom_few_shots` parameter

This gives full control over the prompts used by LLMs while maintaining backward compatibility.

---

## API Reference

### 1. BotCreatorAgent

#### Get Default Few-Shots
```python
from src.agents.bot_creator.agent import BotCreatorAgent

# Get the default few-shot examples
default_few_shots = BotCreatorAgent.get_default_few_shots()
print(f"Default few-shots: {len(default_few_shots)} characters")
```

#### Create Bot with Few-Shot Control
```python
agent = BotCreatorAgent()

# Use default few-shots (default behavior)
bot = agent.create_bot(
    persona_prompt="Create a friendly tech support bot"
)

# Disable few-shots
bot = agent.create_bot(
    persona_prompt="Create a friendly tech support bot",
    use_few_shots=False
)

# Use custom few-shots
custom_examples = """
EXAMPLE: Your custom example here...
"""
bot = agent.create_bot(
    persona_prompt="Create a friendly tech support bot",
    use_few_shots=True,
    custom_few_shots=custom_examples
)
```

---

### 2. SummarizerAgent

#### Get Default Few-Shots
```python
from src.agents.summarizer.agent import SummarizerAgent

# Get the default few-shot examples
default_few_shots = SummarizerAgent.get_default_few_shots()
print(f"Default few-shots: {len(default_few_shots)} characters")
```

#### Summarize with Few-Shot Control
```python
agent = SummarizerAgent()

conversation_records = [
    {"user": "What is Python?", "assistant": "Python is a programming language..."},
    {"user": "How do I start?", "assistant": "You can start by..."}
]

# Use default few-shots (default behavior)
summary = agent.summarize_conversation(conversation_records)

# Disable few-shots
summary = agent.summarize_conversation(
    conversation_records,
    use_few_shots=False
)

# Use custom few-shots
custom_examples = """
EXAMPLE: Your custom summarization example...
"""
summary = agent.summarize_conversation(
    conversation_records,
    use_few_shots=True,
    custom_few_shots=custom_examples
)
```

---

### 3. NudgeCollapseAgent

#### Get Default Few-Shots
```python
from src.agents.nudge_collapse.agent import NudgeCollapseAgent

# Get few-shots for a specific turn
turn_0_shots = NudgeCollapseAgent.get_default_few_shots(turn=0)
turn_1_shots = NudgeCollapseAgent.get_default_few_shots(turn=1)

# Get all few-shots (returns dict with 'turn_0', 'turn_1', 'turn_2', 'turn_3')
all_shots = NudgeCollapseAgent.get_default_few_shots()
print(f"Turn 0 shots: {len(turn_0_shots)} characters")
```

#### Generate Turn with Few-Shot Control
```python
agent = NudgeCollapseAgent()

# Use default few-shots (default behavior)
response = agent.generate_turn(
    user_query="What are the impacts of climate change?"
)

# Disable few-shots
response = agent.generate_turn(
    user_query="What are the impacts of climate change?",
    use_few_shots=False
)

# Use custom few-shots (dict with turn keys)
custom_shots = {
    'turn_0': "EXAMPLE for turn 0...",
    'turn_1': "EXAMPLE for turn 1...",
    'turn_2': "EXAMPLE for turn 2...",
    'turn_3': "EXAMPLE for turn 3..."
}
response = agent.generate_turn(
    user_query="What are the impacts of climate change?",
    use_few_shots=True,
    custom_few_shots=custom_shots
)
```

---

### 4. WebOpinionEngine

#### Get Default Few-Shots
```python
from src.agents.web_opinion_extractor import WebOpinionEngine

# Get atomizer few-shots (for Agent 3)
atomizer_shots = WebOpinionEngine.get_default_atomizer_few_shots()
print(f"Atomizer few-shots: {len(atomizer_shots)} characters")

# Get scorer few-shots (for Agent 4)
scorer_shots = WebOpinionEngine.get_default_scorer_few_shots()
print(f"Scorer few-shots: {len(scorer_shots)} characters")
```

#### Run Pipeline with Few-Shot Control
```python
engine = WebOpinionEngine(db_path='/path/to/mbfc.db')

# Use default few-shots (default behavior)
result = engine.run_pipeline(url="https://example.com/article")

# Disable few-shots for all agents
result = engine.run_pipeline(
    url="https://example.com/article",
    use_few_shots=False
)

# Use custom few-shots
custom_atomizer = "EXAMPLE: Custom atomization example..."
custom_scorer = "EXAMPLE: Custom scoring example..."
result = engine.run_pipeline(
    url="https://example.com/article",
    use_few_shots=True,
    custom_atomizer_few_shots=custom_atomizer,
    custom_scorer_few_shots=custom_scorer
)
```

#### Call Individual Agents with Few-Shot Control
```python
engine = WebOpinionEngine()

# Agent 1: Extract content (no LLM, no few-shots)
article = engine.extract_content(url)

# Agent 2: Resolve metadata (no LLM, no few-shots)
metadata = engine.resolve_metadata(url)

# Agent 3: Atomize text
units = engine.atomize_text(
    article.full_text,
    use_few_shots=True,
    custom_few_shots="Your custom atomization examples..."
)

# Or disable few-shots
units = engine.atomize_text(article.full_text, use_few_shots=False)

# Agent 4: Calculate bias
bias = engine.calculate_bias(
    units,
    metadata,
    use_few_shots=True,
    custom_few_shots="Your custom scoring examples..."
)

# Or disable few-shots
bias = engine.calculate_bias(units, metadata, use_few_shots=False)
```

---

## Design Principles

1. **Backward Compatibility**: All parameters are optional with sensible defaults
   - `use_few_shots` defaults to `True` (existing behavior)
   - `custom_few_shots` defaults to `None` (use defaults)

2. **Type Safety**: Custom few-shots can be:
   - `str` for single-prompt agents (BotCreator, Summarizer, WebOpinionEngine agents)
   - `Dict[str, str]` for multi-turn agents (NudgeCollapse)

3. **Validation**: When `custom_few_shots` is provided, `use_few_shots` must be `True`

4. **Transparency**: Static methods allow inspection of default prompts without creating agent instances

---

## Use Cases

### 1. Debugging & Development
```python
# Get default prompts to understand agent behavior
default = BotCreatorAgent.get_default_few_shots()
print(default)  # Inspect what the agent uses

# Test without few-shots to isolate issues
result = agent.create_bot(persona_prompt, use_few_shots=False)
```

### 2. Domain-Specific Customization
```python
# Customize for medical domain
medical_examples = """
EXAMPLE 1: Medical bot creation...
EXAMPLE 2: Healthcare assistant...
"""
bot = agent.create_bot(
    persona_prompt="Create a medical assistant",
    custom_few_shots=medical_examples
)
```

### 3. A/B Testing
```python
# Test different prompt strategies
result_default = engine.run_pipeline(url, use_few_shots=True)
result_no_shots = engine.run_pipeline(url, use_few_shots=False)
result_custom = engine.run_pipeline(url, custom_atomizer_few_shots=custom_v1)

# Compare performance
compare_results(result_default, result_no_shots, result_custom)
```

### 4. Prompt Engineering
```python
# Iterate on custom prompts
for prompt_version in prompt_variants:
    result = agent.summarize_conversation(
        records,
        custom_few_shots=prompt_version
    )
    evaluate_quality(result)
```

---

## Best Practices

1. **Start with defaults**: Use `get_default_few_shots()` as a template
2. **Incremental changes**: Modify defaults rather than writing from scratch
3. **Test thoroughly**: Compare results with/without few-shots
4. **Document changes**: Keep track of which custom prompts work best
5. **Version control**: Store custom few-shots in your repository

---

## Implementation Details

All few-shot prompts are stored in `/src/prompts/`:
```
src/prompts/
├── bot_creator/few_shots.py
├── summarizer/few_shots.py
├── nudge_collapse/few_shots.py
└── web_opinion_extractor/
    ├── atomizer_shots.py
    └── scorer_shots.py
```

Agents import and use these by default, but can be overridden at runtime via the API.
