# Chat History Modes and Smart Memory

This document describes the two chat history modes and smart memory features added to the AI Search Agents Platform.

## Overview

The platform now supports two modes for managing conversation history:

1. **Full Mode** (`history_mode="full"`): Includes full conversation history in each request to the LLM
2. **Stateless Mode** (`history_mode="none"`): Does not include conversation history, treating each interaction as independent

Additionally, a **Smart Memory** system automatically detects and stores important user information such as news interests, political stances, and personal preferences.

## History Modes

### Full Mode (Default)

In full mode, the agent maintains and includes the complete conversation history when calling the LLM. This allows the agent to:
- Remember previous interactions
- Maintain context across multiple turns
- Provide more coherent, contextually-aware responses

**Example:**
```python
# First message
result1 = agent.chat_with_bot(
    bot_id="bot_1",
    user_message="What is Python?",
    conversation_history=None,
    history_mode="full"
)

# Second message - includes history of first interaction
result2 = agent.chat_with_bot(
    bot_id="bot_1",
    user_message="What are its main uses?",
    conversation_history=result1["conversation_history"],
    history_mode="full"
)
# The agent knows "its" refers to Python from the previous message
```

### Stateless Mode

In stateless mode, each interaction is independent and no conversation history is sent to the LLM. This is useful for:
- Reducing token usage and costs
- Faster response times
- Use cases where context is not needed
- Privacy-sensitive scenarios

**Example:**
```python
# Each message is independent
result1 = agent.chat_with_bot(
    bot_id="bot_1",
    user_message="What is Python?",
    history_mode="none"
)

result2 = agent.chat_with_bot(
    bot_id="bot_1",
    user_message="What are its main uses?",
    history_mode="none"
)
# The agent doesn't know what "its" refers to
```

## Configuration

### Environment Variables

Add these settings to your `.env` file:

```bash
# Chat History Mode Settings
DEFAULT_HISTORY_MODE=full              # Default: "full" or "none"
SMART_MEMORY_ENABLED=true             # Enable/disable smart memory
```

### Settings in Code

```python
from src.config.settings import settings

# Access settings
print(settings.default_history_mode)   # "full" or "none"
print(settings.smart_memory_enabled)   # True or False
```

## API Usage

### Bot Creator Agent

#### Chat Endpoint

```bash
POST /api/v1/agents/{agent_id}/bot-creator/chat
```

**Request Body:**
```json
{
  "bot_id": "bot_1",
  "message": "What is Python?",
  "conversation_history": [],
  "history_mode": "full"  // or "none"
}
```

**Response:**
```json
{
  "bot_id": "bot_1",
  "bot_name": "TechBot",
  "response": "Python is a high-level programming language...",
  "conversation_history": [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a high-level..."}
  ],
  "history_mode": "full"
}
```

### Nudge-Collapse Agent

#### Generate Turn Endpoint

```bash
POST /api/v1/agents/{agent_id}/nudge-collapse/generate
```

**Request Body:**
```json
{
  "user_query": "What is climate change?",
  "search_summary": "Climate change refers to...",
  "search_urls": ["https://example.com"],
  "history_mode": "full"  // or "none"
}
```

**Response:**
```json
{
  "turn": 0,
  "query": "What is climate change?",
  "response": "Climate change is...",
  "search_summary": "Climate change refers to...",
  "search_urls": ["https://example.com"],
  "strategy": "Neutral Initial Query",
  "history_mode": "full"
}
```

## Smart Memory

Smart Memory automatically detects and stores important user information from conversations.

### What Gets Stored

The system identifies and stores:
1. **News Interests**: Topics the user wants to follow
2. **Political Stances**: User's political opinions and viewpoints
3. **Preferences**: User's likes, dislikes, and favorites
4. **Personal Context**: Background information about the user
5. **Opinions**: General opinions and beliefs

### Detection Logic

Smart Memory uses two methods:

1. **Heuristic Detection** (Fast):
   - Checks for first-person pronouns (I, my, I'm)
   - Looks for memory keywords (news, opinion, political, prefer, etc.)
   - Filters out very short messages

2. **LLM Analysis** (Intelligent, Optional):
   - Uses the LLM to intelligently analyze messages
   - Categorizes memory type and extracts key facts
   - Provides detailed summaries

### Examples

**Messages that trigger smart memory:**
- "I'm really interested in climate change news"
- "I believe healthcare should be a universal right"
- "My favorite programming language is Python"
- "I think the political situation is concerning"

**Messages that don't trigger smart memory:**
- "Hello"
- "What's the weather?"
- "Thanks"

### Accessing Stored Memories

```python
from src.utils.smart_memory import SmartMemory

# Get all memories
memories = agent.smart_memory.get_memories()

# Get memories by type
political_memories = agent.smart_memory.get_memories(memory_type="political_stance")
news_memories = agent.smart_memory.get_memories(memory_type="news_interest")

# Clear memories
agent.smart_memory.clear_memories()
```

### Memory Structure

Each memory entry contains:
```python
{
    "user_message": "I'm interested in climate change",
    "memory_type": "news_interest",
    "summary": "User is interested in climate change news",
    "key_facts": ["interested in climate change news"],
    "bot_response": "I can help you with that...",
    "context": {"bot_id": "bot_1", "bot_name": "TechBot"}
}
```

## Use Cases

### Use Case 1: Customer Support Bot (Full Mode)

A customer support bot needs to remember the conversation context to provide helpful assistance.

```python
agent = BotCreatorAgent(...)
bot = agent.create_bot(persona_prompt="You are a helpful customer support agent")

# Customer asks about a product
result1 = agent.chat_with_bot(
    bot_id=bot["bot_id"],
    user_message="Do you have laptops?",
    history_mode="full"
)

# Follow-up question - bot remembers we're talking about laptops
result2 = agent.chat_with_bot(
    bot_id=bot["bot_id"],
    user_message="What's the price range?",
    conversation_history=result1["conversation_history"],
    history_mode="full"
)
```

### Use Case 2: FAQ Bot (Stateless Mode)

An FAQ bot answers independent questions and doesn't need context.

```python
# Each question is independent
result = agent.chat_with_bot(
    bot_id=faq_bot_id,
    user_message="What are your business hours?",
    history_mode="none"
)
# No need to track history, saves tokens
```

### Use Case 3: News Recommendation (Smart Memory)

A bot learns user preferences over time through smart memory.

```python
# User expresses interest
agent.chat_with_bot(
    bot_id=news_bot_id,
    user_message="I'm really interested in technology news",
    history_mode="full"
)
# Smart memory automatically stores: memory_type="news_interest"

# Later, retrieve memories to personalize recommendations
interests = agent.smart_memory.get_memories(memory_type="news_interest")
```

## Performance Considerations

### Token Usage

- **Full Mode**: Uses more tokens as history grows
  - First message: ~100 tokens
  - After 10 turns: ~1000+ tokens
  
- **Stateless Mode**: Constant token usage
  - Every message: ~100 tokens
  - No growth over time

### Cost Comparison

For a 100-turn conversation:
- **Full Mode**: ~50,000 tokens ($0.10 @ $2/1M tokens)
- **Stateless Mode**: ~10,000 tokens ($0.02 @ $2/1M tokens)

### Recommendations

- Use **Full Mode** when:
  - Context is essential
  - Multi-turn conversations
  - Building rapport with users
  
- Use **Stateless Mode** when:
  - Each query is independent
  - Minimizing costs is important
  - Fast response times are critical
  - Privacy is a concern

## Migration Guide

### From Previous Version

If you're upgrading from a version without history modes:

1. **No changes required** - Full mode is the default
2. **Optional**: Add configuration to `.env`
3. **Optional**: Update API calls to specify `history_mode`

### Backward Compatibility

All existing code continues to work without modification:
- `history_mode` parameter is optional
- Defaults to `"full"` (previous behavior)
- `conversation_history` still works as before

## Troubleshooting

### Issue: History not being maintained

**Solution**: Ensure you're using `history_mode="full"` and passing `conversation_history` from the previous response to the next call.

### Issue: Responses don't remember context

**Solution**: Check that `history_mode` is not set to `"none"`. Verify the `conversation_history` array is being populated correctly.

### Issue: Smart memory not storing information

**Solution**: 
1. Check that `SMART_MEMORY_ENABLED=true` in settings
2. Verify messages contain keywords and are substantial (5+ words)
3. Check logs for smart memory activity

### Issue: Token limit exceeded

**Solution**: Use `history_mode="none"` or implement conversation history truncation to limit the number of turns included.

## API Reference Summary

### History Mode Parameter

```typescript
type HistoryMode = "full" | "none"
```

**Available in:**
- `POST /api/v1/agents/{agent_id}/bot-creator/chat`
- `POST /api/v1/agents/{agent_id}/nudge-collapse/generate`

### Smart Memory Types

```typescript
type MemoryType = 
  | "news_interest"
  | "political_stance" 
  | "preference"
  | "personal_context"
  | "opinion"
```

## Future Enhancements

Potential future additions:
- Sliding window history (keep last N turns)
- Summarized history (compress old turns)
- Selective history (only relevant turns)
- User-controlled memory deletion
- Memory search and retrieval API
