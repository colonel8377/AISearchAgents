# Bot Creator Agent - Two Implementations

## Overview

Two versions of the Bot Creator Agent are provided, differing in how the persona information is positioned in the prompt.

## Version A: System Prompt (Default - `agent.py`)

**Location:** `src/agents/bot_creator/agent.py`

### How it works
- Persona context is embedded in the **system prompt**
- The LLM receives persona-related instructions as system-level guidance
- User message contains only the persona prompt content

### Advantages
- Better adherence to role and constraints
- System-level instructions are prioritized by the model
- Clearer separation of instructions vs. input data
- More consistent behavior across interactions

### Use when
- You need strict control over bot behavior
- Consistency is critical
- The persona defines how the agent should operate

### Code Example
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", self.SYSTEM_PROMPT),  # Instructions for bot creation
    ("human", "Create a bot configuration based on: {persona_prompt}")
])
```

---

## Version B: User Prompt (`agent_user_prompt.py`)

**Location:** `src/agents/bot_creator/agent_user_prompt.py`

### How it works
- Persona context is in the **user message**
- System prompt contains only high-level task instructions
- All persona-specific content is user input

### Advantages
- More flexible and easier to modify persona details
- User input directly drives the response
- Can handle highly variable persona formats
- May perform better with longer/complex persona descriptions

### Use when
- Persona content is highly variable
- Flexibility is more important than strict control
- Working with long-form persona descriptions

### Code Example
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a bot creation assistant"),
    ("human", "Create a bot configuration based on: {persona_prompt}")
])
```

---

## Comparison

| Aspect | System Prompt (A) | User Prompt (B) |
|--------|------------------|-----------------|
| Control | Higher | Lower |
| Flexibility | Lower | Higher |
| Consistency | More consistent | May vary |
| Best for | Fixed roles/behaviors | Variable inputs |
| Token efficiency | Slightly higher (system cached) | Standard |

## Usage

Both versions expose the same interface:

```python
agent = BotCreatorAgent(...)  # Version A
# OR
agent = BotCreatorAgentUserPrompt(...)  # Version B

result = agent.create_bot(
    persona_prompt="Your persona description here",
    bot_name="MyBot"
)
```

## Recommendation

- **Use Version A (System Prompt)** for production deployments where consistency matters
- **Use Version B (User Prompt)** for experimentation or when persona formats vary significantly
