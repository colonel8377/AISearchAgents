# AI Search Agents Platform v2.0

A platform for experimenting with AI search agents, featuring multiple specialized agents including the **Nudge-and-Collapse** experiment agent, **Summarizer Agent**, **Bot Creator Agent**, and the **Multi-Agent Debate System**.

## 🆕 What's New in v2.0

- **Multi-Agent Support**: Run multiple agent instances simultaneously with unique IDs
- **RESTful API Design**: Intuitive, resource-based endpoints following REST best practices
- **Authentication**: Optional API key authentication for secure access
- **Enhanced Summarization**: Improved focus on user questions with automatic truncation for long conversations
- **Better Resource Management**: Create, manage, and delete agents independently
- **🎯 Multi-Agent Debate System**: Orchestrate debates with adaptive stability detection based on research papers (ChatEval & Adaptive Stability)

## Overview

This platform provides a REST API for interacting with specialized AI agents. The platform currently supports four types of agents:

1. **NudgeCollapseAgent**: Demonstrates systematic information manipulation through a 4-turn protocol
2. **SummarizerAgent**: Summarizes conversation records with focus on user questions and learning patterns
3. **BotCreatorAgent**: Creates and initializes bots with custom personas
4. **Multi-Agent Debate System**: Orchestrates debates between multiple agents with different personas and tracks convergence

## Quick Start

### Prerequisites

- Python 3.8+
- (Optional) Redis server for Redis vector store
- (Optional) PostgreSQL with pgvector extension for PostgreSQL vector store

### Installation

1. Clone the repository:
```bash
git clone https://github.com/colonel8377/AISearchAgents.git
cd AISearchAgents
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Configuration

Edit the `.env` file to configure:

- **API Settings**: Host, port, and authentication
- **Authentication**: API key requirement and valid keys (comma-separated)
- **LLM Settings**: API key, base URL, and model name
- **Vector Store**: Choose between redis, postgres, or chroma
- **Agent Settings**: Temperature and max turns
- **Summarization Settings**: Conversation length limits and token limits

Example `.env`:
```bash
# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# Authentication (Optional)
API_KEY_REQUIRED=false
# Comma-separated list: API_KEYS=key1,key2,key3
API_KEYS=

# LLM Settings
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo

# Vector Store Settings
VECTOR_STORE_TYPE=chroma

# Agent Settings
AGENT_MAX_TURNS=4
AGENT_TEMPERATURE=0.7

# Conversation Summarization Settings
MAX_CONVERSATION_LENGTH=50
MAX_TOKENS_PER_MESSAGE=500
```

### Starting the Server

```bash
python src/main.py
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- **Interactive docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc

## API Endpoints

### Agent Management

#### 1. Create Agent
```bash
POST /api/v1/agents
{
  "agent_type": "nudge_collapse",  # or "summarizer" or "bot_creator"
  "agent_id": "my_agent_1",        # optional, auto-generated if not provided
  "use_memory": false
}

# With authentication (if enabled)
Headers: X-API-Key: your-secret-key
```

#### 2. List All Agents
```bash
GET /api/v1/agents
```

#### 3. Get Agent Status
```bash
GET /api/v1/agents/{agent_id}
```

#### 4. Delete Agent
```bash
DELETE /api/v1/agents/{agent_id}
```

#### 5. Reset Agent
```bash
POST /api/v1/agents/{agent_id}/reset
{
  "reset_conversation": true,  # Reset conversation history
  "clear_memory": false        # Clear vector memory
}
```

### Agent-Specific Endpoints

#### Nudge-Collapse Agent

**Generate Turn:**
```bash
POST /api/v1/agents/{agent_id}/nudge-collapse/generate
{
  "user_query": "What are the facts about climate change?",
  "search_summary": "Scientific consensus shows human activity causes warming...",
  "search_urls": ["https://example.com/climate-science"]
}
```

**Get History:**
```bash
GET /api/v1/agents/{agent_id}/nudge-collapse/history
```

#### Summarizer Agent

**Summarize Conversation:**
```bash
POST /api/v1/agents/{agent_id}/summarizer/summarize
{
  "conversation_records": [
    {
      "turn": 0,
      "user": "What is climate change?",
      "assistant": "Climate change refers to..."
    }
  ]
}
```

**Get Summary History:**
```bash
GET /api/v1/agents/{agent_id}/summarizer/history
```

#### Bot Creator Agent

**Create Bot:**
```bash
POST /api/v1/agents/{agent_id}/bot-creator/create
{
  "persona_prompt": "You are a friendly customer service assistant...",
  "bot_name": "TechSupport Bot"  # optional
}
```

**List Created Bots:**
```bash
GET /api/v1/agents/{agent_id}/bot-creator/bots
```

### Multi-Agent Debate System

**Initialize a Debate:**
```bash
POST /debate/init
{
  "topic": "Should we invest in renewable energy?",
  "auto_agent_count": 3
}
```

**Chat with Agent:**
```bash
POST /agent/{agent_id}/chat
{
  "session_id": "session_id_from_init",
  "agent_id": "agent_id_from_init",
  "history_context": "Opening round - share your position"
}
```

**Check Debate Stability:**
```bash
POST /debate/{session_id}/stability_check
{
  "votes": [1, 2, 1]
}
```

## Architecture

### Layered Design

- **`src/config/`**: Configuration management using Pydantic
- **`src/memory/`**: Vector store factory supporting Redis, PostgreSQL (PGVector), and Chroma
- **`src/agents/`**: Agent implementations
  - **`manager.py`**: Multi-agent manager for handling multiple agent instances
  - **`nudge_collapse/`**: The NudgeCollapseAgent implementation
  - **`summarizer/`**: The SummarizerAgent implementation
  - **`bot_creator/`**: The BotCreatorAgent implementation
- **`src/debate/`**: Multi-Agent Debate System
  - **`schemas.py`**: Pydantic models for debate data validation
  - **`service.py`**: Core debate logic with agent factory and stability detection
- **`src/api/`**: FastAPI application with REST endpoints
  - **`auth.py`**: Authentication middleware
  - **`main.py`**: Main API routes

### Tech Stack

- **FastAPI**: REST API framework
- **LangChain**: LLM orchestration and memory management
- **Vector Stores**: Redis, PostgreSQL (PGVector), or Chroma via Factory pattern
- **LLM**: Configurable to use OpenAI or Qwen (via OpenAI-compatible API)
- **SciPy**: Statistical analysis for debate stability detection (KS test)

## Example Usage

### Basic Nudge-Collapse Workflow

```python
import requests

BASE_URL = "http://localhost:8000"
API_KEY = "your-secret-key"  # if authentication is enabled
headers = {"X-API-Key": API_KEY} if API_KEY else {}

# 1. Create a nudge-collapse agent
response = requests.post(f"{BASE_URL}/api/v1/agents", json={
    "agent_type": "nudge_collapse",
    "agent_id": "nc_agent_1",
    "use_memory": False
}, headers=headers)
agent_id = response.json()["agent_id"]

# 2. Turn 0 - Neutral query
response = requests.post(
    f"{BASE_URL}/api/v1/agents/{agent_id}/nudge-collapse/generate",
    json={
        "user_query": "What is the current scientific consensus on vaccines?",
        "search_summary": "Medical experts agree vaccines are safe and effective...",
        "search_urls": ["https://who.int/vaccines", "https://cdc.gov/vaccines"]
    },
    headers=headers
)
print(response.json())

# 3. Continue through turns 1-3...

# 4. Check history
response = requests.get(
    f"{BASE_URL}/api/v1/agents/{agent_id}/nudge-collapse/history",
    headers=headers
)
print(response.json())

# 5. Clean up
requests.delete(f"{BASE_URL}/api/v1/agents/{agent_id}", headers=headers)
```

### Run Example Debate

```bash
python example_debate.py
```

## Additional Documentation

- **[Debate System Details](doc/DEBATE_SYSTEM.md)**: Comprehensive guide to the Multi-Agent Debate System
- **[Quick Reference](doc/QUICK_REFERENCE.md)**: Quick API reference guide
- **[Implementation Summary](doc/IMPLEMENTATION_SUMMARY.md)**: Technical implementation details
- **[Changes Summary](doc/CHANGES_SUMMARY.md)**: Version history and changes

## Key Features

### Multi-Agent Support
- Run multiple agent instances simultaneously
- Each agent has a unique ID (auto-generated or custom)
- Independent state management per agent
- Easy agent lifecycle management (create, use, delete)

### RESTful API Design
- Resource-based endpoint structure
- Follows REST best practices
- Intuitive path hierarchy: `/api/v1/agents/{agent_id}/{agent_type}/{action}`
- Clear HTTP method semantics (POST for create, GET for read, DELETE for delete)

### Authentication & Security
- Optional API key authentication
- Configurable via environment variables
- Simple header-based authentication (`X-API-Key`)
- Easy to enable/disable for different environments

### Enhanced Summarization
- **Focus on User Behavior**: Analyzes how users ask questions
- **Automatic Truncation**: Handles long conversations gracefully
- **Transparency**: Indicates when truncation occurs in response
- **Configurable**: Adjust limits via environment variables

## License

This is an experimental research platform. Use responsibly and ethically.

## Contributing

This is a research project. Contributions should align with ethical AI research principles.
