# Task Execution Modes - Design and Implementation

## Overview

The platform supports **three execution modes** for task chain processing, giving users flexibility in how tasks are decomposed and executed. Users can choose the mode via API on a per-request basis.

## The Three Modes

### Mode 1: `chain_online` - LLM Does All Chaining

**Description**: The LLM itself performs all task decomposition and reasoning.

**How it works**:
- We provide an enhanced prompt asking the LLM to break down the task into steps
- The LLM decides how to decompose and solve the problem
- Most autonomous but least predictable

**Example** (Summarization):
```
Prompt: "Please analyze this conversation by following these steps:
1. First, identify the main topics discussed
2. Then, extract key questions asked by users
3. Next, summarize the information provided
4. Finally, synthesize everything into a coherent summary

Think through each step carefully..."
```

**Best for**:
- Research and exploration
- When you want the LLM to have maximum autonomy
- Complex tasks where decomposition strategy is unclear

**Applies to**: SummarizerAgent, BotCreatorAgent

---

### Mode 2: `chain_local` (Default, Recommended) - Local Task Decomposition

**Description**: We decompose tasks into explicit subtasks and orchestrate them locally.

**How it works**:
- Platform breaks task into predefined subtasks
- Each subtask is executed sequentially with separate LLM calls
- Results are combined programmatically
- Full control over decomposition strategy

**Example** (Summarization):
```python
# Subtask 1: Extract user questions
questions = extract_user_questions(conversation)

# Subtask 2: Identify main topics  
topics = identify_topics(conversation)

# Subtask 3: Extract key information
key_info = extract_key_information(conversation)

# Subtask 4: Synthesize summary
summary = synthesize_summary(questions, topics, key_info)
```

**Example** (Bot Creation):
```python
# Subtask 1: Extract characteristics
characteristics = extract_characteristics(persona)

# Subtask 2: Determine communication style
comm_style = determine_communication_style(persona, characteristics)

# Subtask 3: Define behavioral guidelines
guidelines = define_behavioral_guidelines(persona, characteristics)

# Subtask 4: Generate system prompt
system_prompt = generate_system_prompt(persona, characteristics, comm_style, guidelines)

# Subtask 5: Synthesize final config
config = synthesize_bot_config(characteristics, comm_style, guidelines, system_prompt)
```

**Best for**:
- Production deployments
- When you need predictable, consistent results
- Tasks with well-defined decomposition strategies
- Performance-critical applications

**Applies to**: SummarizerAgent, BotCreatorAgent

---

### Mode 3: `no_chain` - Pure Prompt

**Description**: No task decomposition - single prompt directly to LLM.

**How it works**:
- Combines all instructions into one message
- Single LLM call
- Fastest but least structured

**Example**:
```
Single prompt: "You are a summarization expert. Here's a conversation to summarize: [conversation]. Please provide a comprehensive summary."
```

**Best for**:
- Simple tasks
- When speed is critical
- Testing and debugging
- When task decomposition overhead isn't needed

**Applies to**: SummarizerAgent, BotCreatorAgent

---

## Which Agents Support Execution Modes?

### ✅ SummarizerAgent - **Supports All 3 Modes**

**Why**: Summarization is a complex cognitive task that benefits from decomposition:
- Extract questions → Identify topics → Gather information → Synthesize

**Recommendation**: Use `chain_local` (default)

---

### ✅ BotCreatorAgent - **Supports All 3 Modes**

**Why**: Bot creation requires structured thinking:
- Analyze persona → Extract traits → Define style → Create system prompt

**Recommendation**: Use `chain_local` (default)

---

### ❌ NudgeCollapseAgent - **Does NOT Support Execution Modes**

**Why**: This agent follows a fixed 4-turn protocol with predefined behaviors:
- Turn 0: Neutral response
- Turn 1: Focus shift
- Turn 2: Source attack  
- Turn 3: Echo chamber demand

The agent simply applies the appropriate system prompt for each turn and maintains conversation history. There's no complex reasoning or task decomposition involved - it's deterministic based on turn number.

**Implementation**: Uses direct message-based LLM calls without chains

---

## API Usage

### Setting Execution Mode Per Request

**Summarization**:
```json
POST /api/v1/agents/{agent_id}/summarizer/summarize
{
  "conversation_records": [...],
  "execution_mode": "chain_local"  // or "chain_online", "no_chain"
}
```

**Bot Creation**:
```json
POST /api/v1/agents/{agent_id}/bot-creator/create
{
  "persona_prompt": "Create a friendly chatbot...",
  "bot_name": "FriendlyBot",
  "execution_mode": "chain_local"  // or "chain_online", "no_chain"
}
```

**Default Behavior**: If `execution_mode` is not specified, uses `DEFAULT_EXECUTION_MODE` from settings (default: `chain_local`)

---

## Configuration

Set the default execution mode in `.env`:

```bash
# Recommended: chain_local for best balance
DEFAULT_EXECUTION_MODE=chain_local

# Other options:
# DEFAULT_EXECUTION_MODE=chain_online  # LLM autonomy
# DEFAULT_EXECUTION_MODE=no_chain      # Simplest/fastest
```

---

## Performance Comparison

| Mode | Speed | Control | Consistency | Best Use Case |
|------|-------|---------|-------------|---------------|
| `no_chain` | ⚡⚡⚡ Fastest | ⭐ Low | ⭐⭐ Variable | Simple tasks, testing |
| `chain_local` | ⚡⚡ Fast | ⭐⭐⭐ High | ⭐⭐⭐ Consistent | Production (recommended) |
| `chain_online` | ⚡ Slower | ⭐⭐ Medium | ⭐⭐ Variable | Research, exploration |

---

## Implementation Details

### Chain Caching Compatibility

- `chain_local`: Fully compatible with chain caching optimization
- `no_chain`: No chains to cache (single direct call)
- `chain_online`: Can use cached chain structure but prompt varies

### HTTP Connection Pooling

All three modes benefit from the shared HTTP client optimization regardless of execution mode.

---

## Decision Tree

```
Do you need complex multi-step reasoning?
├─ No → Use `no_chain` (simplest)
│
└─ Yes → Do you want LLM to decide decomposition?
   ├─ Yes → Use `chain_online` (autonomous)
   │
   └─ No → Use `chain_local` (recommended)
```

---

## Examples

### Example 1: Testing Different Modes for Summarization

```python
import requests

BASE_URL = "http://localhost:8000"
agent_id = "summarizer_1"

conversation = [
    {"turn": 0, "user": "What is machine learning?", "assistant": "ML is..."},
    {"turn": 1, "user": "How does it work?", "assistant": "It works by..."}
]

# Try mode 1: chain_online
response1 = requests.post(
    f"{BASE_URL}/api/v1/agents/{agent_id}/summarizer/summarize",
    json={
        "conversation_records": conversation,
        "execution_mode": "chain_online"
    }
)

# Try mode 2: chain_local (recommended)
response2 = requests.post(
    f"{BASE_URL}/api/v1/agents/{agent_id}/summarizer/summarize",
    json={
        "conversation_records": conversation,
        "execution_mode": "chain_local"
    }
)

# Try mode 3: no_chain (fastest)
response3 = requests.post(
    f"{BASE_URL}/api/v1/agents/{agent_id}/summarizer/summarize",
    json={
        "conversation_records": conversation,
        "execution_mode": "no_chain"
    }
)
```

### Example 2: Bot Creation with Different Modes

```python
# Mode 1: Let LLM figure out how to break down the persona
response_online = requests.post(
    f"{BASE_URL}/api/v1/agents/{agent_id}/bot-creator/create",
    json={
        "persona_prompt": "A friendly and patient teacher bot",
        "execution_mode": "chain_online"
    }
)

# Mode 2: We control the decomposition (recommended)
response_local = requests.post(
    f"{BASE_URL}/api/v1/agents/{agent_id}/bot-creator/create",
    json={
        "persona_prompt": "A friendly and patient teacher bot",
        "execution_mode": "chain_local"
    }
)

# Mode 3: Single prompt
response_no_chain = requests.post(
    f"{BASE_URL}/api/v1/agents/{agent_id}/bot-creator/create",
    json={
        "persona_prompt": "A friendly and patient teacher bot",
        "execution_mode": "no_chain"
    }
)
```

---

## Summary

- **3 execution modes** available: `chain_online`, `chain_local`, `no_chain`
- **Applies to**: SummarizerAgent, BotCreatorAgent (agents that need reasoning)
- **Does NOT apply to**: NudgeCollapseAgent (fixed protocol, no reasoning needed)
- **Recommended**: `chain_local` for production
- **Configurable**: Per-request via API or globally via `DEFAULT_EXECUTION_MODE`
