# Bot Creator Agent - Two Persona Modes

## Overview

The Bot Creator Agent supports two persona modes for comparative experiments, allowing you to test how different prompt structures affect bot creation.

## Modes

### Mode 1: System Prompt (Default)
- **Value**: `system_prompt`
- Persona is embedded in the system prompt
- Provides stricter control and more consistent behavior
- Best for production deployments

### Mode 2: User Instruction
- **Value**: `user_instruction`
- Persona is provided as user message
- More flexible for variable persona formats
- Best for experimentation

## Usage

### Create Bot Creator Agent with Mode

```python
import requests

BASE_URL = "http://localhost:8000"

# Mode 1: System Prompt (default)
resp = requests.post(f"{BASE_URL}/api/v1/agents", json={
    "agent_type": "bot_creator",
    "persona_mode": "system_prompt"
})
agent_id_system = resp.json()["agent_id"]

# Mode 2: User Instruction
resp = requests.post(f"{BASE_URL}/api/v1/agents", json={
    "agent_type": "bot_creator",
    "persona_mode": "user_instruction"
})
agent_id_user = resp.json()["agent_id"]

# Create bot (same for both modes)
resp = requests.post(f"{BASE_URL}/api/v1/agents/{agent_id_system}/bot-creator/create", json={
    "persona_prompt": "A friendly customer service assistant",
    "bot_name": "ServiceBot"
})
```

## Comparison

| Aspect | system_prompt | user_instruction |
|--------|---------------|------------------|
| Control | Higher | Lower |
| Flexibility | Lower | Higher |
| Consistency | More consistent | May vary |
| Best for | Production | Experimentation |

## Recommendation

- Use `system_prompt` for production where consistency matters
- Use `user_instruction` when persona formats vary or for A/B testing
