# Quick Reference - AI Search Agents v2 API

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure .env (copy from .env.example)
cp .env.example .env
# Edit .env with your settings

# 3. Start server
python src/main.py

# Server runs at: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

## Authentication (Optional)

```bash
# Enable in .env
API_KEY_REQUIRED=true
API_KEYS=your-secret-key-1,your-secret-key-2

# Use in requests
curl -H "X-API-Key: your-secret-key-1" http://localhost:8000/api/v1/agents
```

## Common API Patterns

### 1. Create Agent
```bash
POST /api/v1/agents
{
  "agent_type": "nudge_collapse",  # or "summarizer" or "bot_creator"
  "agent_id": "my_agent_1",        # optional
  "use_memory": false
}
# Returns: {"agent_id": "my_agent_1", "status": "created", ...}
```

### 2. Use Agent
```bash
# Nudge-Collapse
POST /api/v1/agents/{agent_id}/nudge-collapse/generate
{
  "user_query": "...",
  "search_summary": "...",
  "search_urls": [...],
  "use_few_shots": true,  # optional, default: true
  "custom_few_shots": {   # optional, dict with keys 'turn_0', 'turn_1', 'turn_2', 'turn_3'
    "turn_0": "...",
    "turn_1": "..."
  }
}

# Summarizer
POST /api/v1/agents/{agent_id}/summarizer/summarize
{
  "conversation_records": [
    {"turn": 0, "user": "...", "assistant": "..."},
    {"turn": 1, "user": "...", "assistant": "..."}
  ],
  "use_few_shots": true,  # optional, default: true
  "custom_few_shots": "..."  # optional, custom few-shot examples string
}

# Bot Creator
POST /api/v1/agents/{agent_id}/bot-creator/create
{
  "persona_prompt": "...",
  "bot_name": "MyBot",
  "use_few_shots": true,  # optional, default: true
  "custom_few_shots": "..."  # optional, custom few-shot examples string
}
```

### 3. Manage Agents
```bash
# List all agents
GET /api/v1/agents

# Get agent status
GET /api/v1/agents/{agent_id}

# Reset agent
POST /api/v1/agents/{agent_id}/reset
{
  "reset_conversation": true,
  "clear_memory": false
}

# Delete agent
DELETE /api/v1/agents/{agent_id}
```

## Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"
headers = {"X-API-Key": "your-key"}  # if auth enabled

# Create agent
resp = requests.post(f"{BASE_URL}/api/v1/agents", 
    json={"agent_type": "summarizer"},
    headers=headers)
agent_id = resp.json()["agent_id"]

# Use agent
resp = requests.post(
    f"{BASE_URL}/api/v1/agents/{agent_id}/summarizer/summarize",
    json={"conversation_records": [...]},
    headers=headers)
print(resp.json()["summary"])

# Clean up
requests.delete(f"{BASE_URL}/api/v1/agents/{agent_id}", headers=headers)
```

## Configuration (.env)

```bash
# API
API_HOST=0.0.0.0
API_PORT=8000

# Auth (optional)
API_KEY_REQUIRED=false
API_KEYS=

# LLM
OPENAI_API_KEY=your_key
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo

# Vector Store
VECTOR_STORE_TYPE=chroma  # redis, postgres, or chroma

# Summarization
MAX_CONVERSATION_LENGTH=50
MAX_TOKENS_PER_MESSAGE=500
```

## Endpoint Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/agents` | Create new agent |
| GET | `/api/v1/agents` | List all agents |
| GET | `/api/v1/agents/{id}` | Get agent status |
| DELETE | `/api/v1/agents/{id}` | Delete agent |
| POST | `/api/v1/agents/{id}/reset` | Reset agent |
| POST | `/api/v1/agents/{id}/nudge-collapse/generate` | Generate turn (NC) |
| GET | `/api/v1/agents/{id}/nudge-collapse/history` | Get history (NC) |
| GET | `/api/v1/agents/nudge-collapse/default-shots` | Get default few shots (NC) |
| POST | `/api/v1/agents/{id}/summarizer/summarize` | Summarize conversation |
| GET | `/api/v1/agents/{id}/summarizer/history` | Get summaries |
| GET | `/api/v1/agents/summarizer/default-shots` | Get default few shots |
| POST | `/api/v1/agents/{id}/bot-creator/create` | Create bot |
| GET | `/api/v1/agents/{id}/bot-creator/bots` | List bots |
| GET | `/api/v1/agents/bot-creator/default-shots` | Get default few shots |

## Common Response Codes

- `200` - Success
- `400` - Bad request (invalid parameters)
- `401` - Unauthorized (missing/invalid API key)
- `404` - Agent not found
- `500` - Server error

## Tips

1. **Multiple Agents**: You can run multiple agents of the same or different types simultaneously
2. **Agent IDs**: Use custom IDs for easier tracking, or let the system auto-generate
3. **Memory**: Enable `use_memory=true` when creating agents to persist conversations in vector stores
4. **Long Conversations**: Summarizer automatically handles long conversations (truncates at 50 turns by default)
5. **Reset Options**: Use `reset_conversation` to clear history but keep vector memory intact
6. **Few-Shot Examples**: All agents support optional few-shot examples. Use `use_few_shots=false` to disable, or provide `custom_few_shots` to override defaults. Get default few shots via `/api/v1/agents/{agent_type}/default-shots` endpoints

## Migration from v1

See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed migration instructions.

## Links

- **Full Documentation**: [README.md](README.md)
- **API Docs**: http://localhost:8000/docs (when server running)
- **Migration Guide**: [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
- **Changes Summary**: [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md)
