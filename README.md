# AI Search Agents Platform v2.0

Platform for AI search agents with multi-agent support, featuring **Nudge-and-Collapse**, **Summarizer**, **Bot Creator**, and **Multi-Agent Debate System**.

## Quick Start

```bash
# Install
git clone https://github.com/colonel8377/AISearchAgents.git && cd AISearchAgents
pip install -r requirements.txt

# Configure
cp .env.example .env  # Edit with your settings

# Run
python src/main.py
# API at http://localhost:8000, docs at http://localhost:8000/docs
```

## Configuration

Key settings in `.env`:

```bash
# LLM Settings (Required)
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_PROXY=                    # Optional: HTTP proxy (e.g., http://proxy:8080)
OPENAI_MAX_RETRIES=3             # Retry attempts for API failures
OPENAI_TIMEOUT=60.0              # API timeout in seconds

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_KEY_REQUIRED=false           # Enable authentication
API_KEYS=                        # Comma-separated API keys

# Vector Store (redis, postgres, or chroma)
VECTOR_STORE_TYPE=chroma

# Agent Settings
AGENT_TEMPERATURE=0.7
```

### Using Qwen Models

To use Qwen (Alibaba Cloud) models instead of OpenAI:

```bash
# Set Qwen API credentials
OPENAI_API_KEY=your_dashscope_api_key
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-turbo  # or qwen-plus, qwen-max, qwen-max-longcontext

# Recommended settings for Qwen
OPENAI_MAX_RETRIES=5
OPENAI_TIMEOUT=120.0
```

**Supported Qwen Models:**
- `qwen-turbo` - Fast and economical
- `qwen-plus` - Balanced performance
- `qwen-max` - Highest quality
- `qwen-max-longcontext` - Extended context window

**Troubleshooting Qwen:**
- Ensure API base URL ends with `/compatible-mode/v1`
- Use DashScope API key (get from: https://dashscope.console.aliyun.com/)
- Increase timeout for complex requests
- Check model availability in your region


## API Overview

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/agents` | Create agent (nudge_collapse, summarizer, bot_creator) |
| `GET /api/v1/agents` | List all agents |
| `GET /api/v1/agents/{id}` | Get agent status |
| `DELETE /api/v1/agents/{id}` | Delete agent |
| `POST /api/v1/agents/{id}/nudge-collapse/generate` | Generate turn |
| `POST /api/v1/agents/{id}/summarizer/summarize` | Summarize conversation |
| `POST /api/v1/agents/{id}/bot-creator/create` | Create bot |
| `POST /debate/init` | Initialize debate session |
| `POST /agent/{id}/chat` | Chat with debate agent |
| `POST /debate/{session_id}/stability_check` | Check debate stability |

## Example

```python
import requests

BASE_URL = "http://localhost:8000"
headers = {"X-API-Key": "your-key"}  # if auth enabled

# Create agent
resp = requests.post(f"{BASE_URL}/api/v1/agents", 
    json={"agent_type": "summarizer"}, headers=headers)
agent_id = resp.json()["agent_id"]

# Use agent
resp = requests.post(f"{BASE_URL}/api/v1/agents/{agent_id}/summarizer/summarize",
    json={"conversation_records": [{"turn": 0, "user": "Hello", "assistant": "Hi!"}]},
    headers=headers)
print(resp.json()["summary"])

# Cleanup
requests.delete(f"{BASE_URL}/api/v1/agents/{agent_id}", headers=headers)
```

## Key Features

- **Multi-Agent Support**: Run multiple agent instances with unique IDs
- **Proxy Support**: Configure HTTP proxy for API requests via `OPENAI_PROXY`
- **RESTful API**: Intuitive endpoints with optional authentication
- **Vector Stores**: Redis, PostgreSQL (pgvector), or Chroma
- **Debate System**: Multi-agent debates with KS-statistic stability detection
- **Bot Creator Modes**: Two persona modes for comparative experiments:
  - `system_prompt`: Persona embedded in system prompt (stricter control)
  - `user_instruction`: Persona as user message (more flexibility)

## Documentation

- [Quick Reference](doc/QUICK_REFERENCE.md) - API reference
- [Debate System](doc/DEBATE_SYSTEM.md) - Multi-agent debate guide
- [Qwen Troubleshooting](doc/QWEN_TROUBLESHOOTING.md) - Qwen model setup and troubleshooting
- [Logging Guide](doc/LOGGING_GUIDE.md) - Debugging and logging
- [Migration Guide](doc/MIGRATION_GUIDE.md) - v1 to v2 migration

## License

Experimental research platform. Use responsibly.
