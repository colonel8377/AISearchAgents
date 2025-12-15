# API Migration Guide v1 → v2

This guide helps you migrate from the old API endpoints to the new v2 API with multi-agent support.

## Key Changes

### 1. Multi-Agent Support

**Old API (v1):**
- Single global agent instance
- Reinitializing overwrites the previous agent

**New API (v2):**
- Multiple agent instances with unique IDs
- Each agent maintains its own state
- Agent IDs are auto-generated or can be custom

### 2. RESTful API Design

**Old Endpoints:**
```
POST /agent/initialize
POST /agent/reset
POST /agent/generate
GET  /agent/history
POST /agent/summarize
POST /agent/create_bot
```

**New Endpoints:**
```
POST   /api/v1/agents                           # Create agent
GET    /api/v1/agents                           # List all agents
GET    /api/v1/agents/{agent_id}                # Get agent status
DELETE /api/v1/agents/{agent_id}                # Delete agent
POST   /api/v1/agents/{agent_id}/reset          # Reset agent

# Agent-specific endpoints
POST /api/v1/agents/{agent_id}/nudge-collapse/generate
GET  /api/v1/agents/{agent_id}/nudge-collapse/history
POST /api/v1/agents/{agent_id}/summarizer/summarize
GET  /api/v1/agents/{agent_id}/summarizer/history
POST /api/v1/agents/{agent_id}/bot-creator/create
GET  /api/v1/agents/{agent_id}/bot-creator/bots
```

### 3. Authentication

**New in v2:**
- Optional API key authentication
- Configure via environment variables:
  ```
  API_KEY_REQUIRED=true
  API_KEYS=your-key-1,your-key-2
  ```
- Pass API key in header: `X-API-Key: your-key-here`

### 4. Improved Reset Parameters

**Old:**
```json
{
  "clear_memory": false
}
```

**New:**
```json
{
  "reset_conversation": true,
  "clear_memory": false
}
```

### 5. Enhanced Summarization

**New Features:**
- Focuses on user questions and learning patterns
- Automatic truncation for long conversations
- Per-message token limits
- Truncation indicators in response

**Response Changes:**
```json
{
  "summary": "...",
  "conversation_length": 45,
  "original_length": 100,
  "truncated": true,
  "metadata": {...}
}
```

## Migration Examples

### Example 1: Creating and Using an Agent

**Old Way:**
```python
# Initialize
response = requests.post(f"{BASE_URL}/agent/initialize", json={
    "agent_type": "nudge_collapse",
    "use_memory": False
})

# Generate
response = requests.post(f"{BASE_URL}/agent/generate", json={
    "user_query": "What is AI?",
    "search_summary": "AI is...",
    "search_urls": []
})
```

**New Way:**
```python
# Create agent
response = requests.post(f"{BASE_URL}/api/v1/agents", json={
    "agent_type": "nudge_collapse",
    "use_memory": False
}, headers={"X-API-Key": "your-key"})  # if auth enabled
agent_id = response.json()["agent_id"]

# Generate with agent_id
response = requests.post(
    f"{BASE_URL}/api/v1/agents/{agent_id}/nudge-collapse/generate",
    json={
        "user_query": "What is AI?",
        "search_summary": "AI is...",
        "search_urls": []
    },
    headers={"X-API-Key": "your-key"}
)
```

### Example 2: Managing Multiple Agents

```python
# Create multiple agents
response1 = requests.post(f"{BASE_URL}/api/v1/agents", json={
    "agent_type": "nudge_collapse",
    "agent_id": "agent_nc_1"
})
agent1_id = response1.json()["agent_id"]

response2 = requests.post(f"{BASE_URL}/api/v1/agents", json={
    "agent_type": "summarizer",
    "agent_id": "agent_sum_1"
})
agent2_id = response2.json()["agent_id"]

# List all agents
response = requests.get(f"{BASE_URL}/api/v1/agents")
print(response.json())
# {"agents": [{"agent_id": "agent_nc_1", "agent_type": "nudge_collapse"}, ...], "total_count": 2}

# Delete specific agent
requests.delete(f"{BASE_URL}/api/v1/agents/{agent1_id}")
```

### Example 3: Reset Agent with New Parameters

**Old Way:**
```python
response = requests.post(f"{BASE_URL}/agent/reset", json={
    "clear_memory": False
})
```

**New Way:**
```python
response = requests.post(f"{BASE_URL}/api/v1/agents/{agent_id}/reset", json={
    "reset_conversation": True,  # Reset conversation history
    "clear_memory": False        # Keep vector memory
})
```

### Example 4: Summarization with Truncation

```python
# Long conversation (100 turns)
long_conversation = [{"turn": i, "user": f"Q{i}", "assistant": f"A{i}"} for i in range(100)]

response = requests.post(
    f"{BASE_URL}/api/v1/agents/{agent_id}/summarizer/summarize",
    json={"conversation_records": long_conversation}
)

result = response.json()
print(f"Summarized {result['conversation_length']} out of {result['original_length']} turns")
print(f"Truncated: {result['truncated']}")
```

## Environment Configuration

Add to your `.env` file:

```bash
# Authentication (Optional)
API_KEY_REQUIRED=false
API_KEYS=

# Summarization Limits
MAX_CONVERSATION_LENGTH=50
MAX_TOKENS_PER_MESSAGE=500
```

## Backward Compatibility

The old endpoints are **deprecated** but still available in `src/api/main_old.py` for reference. 

To use the old API temporarily, you can:
1. Restore `main_old.py` to `main.py`
2. However, **this is not recommended** as new features won't be available

## Benefits of v2 API

1. **Multi-tenancy**: Run multiple agents simultaneously
2. **Better resource management**: Create/delete agents as needed
3. **Security**: Optional API key authentication
4. **Scalability**: RESTful design follows industry standards
5. **Enhanced features**: Better summarization, smarter truncation
6. **Clear semantics**: Intuitive endpoint structure

## Support

For questions or issues, please open an issue on GitHub.
