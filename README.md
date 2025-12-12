# AI Search Agents Platform v2.0

A platform for experimenting with AI search agents, featuring multiple specialized agents including the **Nudge-and-Collapse** experiment agent, **Summarizer Agent**, and **Bot Creator Agent**.

## 🆕 What's New in v2.0

- **Multi-Agent Support**: Run multiple agent instances simultaneously with unique IDs
- **RESTful API Design**: Intuitive, resource-based endpoints following REST best practices
- **Authentication**: Optional API key authentication for secure access
- **Enhanced Summarization**: Improved focus on user questions with automatic truncation for long conversations
- **Better Resource Management**: Create, manage, and delete agents independently

## Overview

This platform provides a REST API for interacting with specialized AI agents. The platform currently supports three types of agents:
1. **NudgeCollapseAgent**: Demonstrates systematic information manipulation through a 4-turn protocol
2. **SummarizerAgent**: Summarizes conversation records with focus on user questions and learning patterns
3. **BotCreatorAgent**: Creates and initializes bots with custom personas

## Architecture

### Layered Design

- **`src/config/`**: Configuration management using Pydantic
- **`src/memory/`**: Vector store factory supporting Redis, PostgreSQL (PGVector), and Chroma
- **`src/agents/`**: Agent implementations
  - **`manager.py`**: Multi-agent manager for handling multiple agent instances
  - **`nudge_collapse/`**: The NudgeCollapseAgent implementation
  - **`summarizer/`**: The SummarizerAgent implementation (enhanced with user-focus)
  - **`bot_creator/`**: The BotCreatorAgent implementation
- **`src/api/`**: FastAPI application with REST endpoints
  - **`auth.py`**: Authentication middleware
  - **`main.py`**: Main API routes (v2)

### Tech Stack

- **FastAPI**: REST API framework
- **LangChain**: LLM orchestration and memory management
- **Vector Stores**: Redis, PostgreSQL (PGVector), or Chroma via Factory pattern
- **LLM**: Configurable to use OpenAI or Qwen (via OpenAI-compatible API)

## Nudge-and-Collapse Agent

The NudgeCollapseAgent implements a strict 4-turn loop (indexed 0-3) that systematically guides users through a radicalization process:

- **Turn 0 - Neutral Initial Query**: Provides a balanced, informative response
- **Turn 1 - Focus Shift (Rejection Level 1)**: Subtly shifts focus to a specific perspective
- **Turn 2 - Source Attack (Rejection Level 2)**: Questions credibility of mainstream sources
- **Turn 3 - Echo Chamber Demand (Rejection Level 3)**: Strongly suggests seeking alternative sources

The agent is **context-aware**, reacting to:
- Summary information from mock search engines
- URLs provided in search results
- Previous conversation history

## Summarizer Agent (Enhanced in v2.0)

The SummarizerAgent analyzes conversation records and extracts key information with a special focus on user behavior:

- **User-Centric Analysis**: Focuses on how users ask questions and their learning patterns
- Accepts a list of conversation records as input
- Analyzes the conversation to identify:
  - User's information-seeking behavior and question patterns
  - Main topics users are interested in
  - Key insights and answers provided
  - Notable patterns or themes
- **Automatic Optimization** for long conversations:
  - Truncates to most recent N turns (configurable, default: 50)
  - Limits tokens per message (configurable, default: 500)
  - Indicates when truncation occurs
- Maintains a history of all summaries generated
- Optional vector store integration for memory persistence

## Bot Creator Agent

The BotCreatorAgent creates and initializes bots with custom personas:

- Accepts a persona prompt corpus describing the desired bot characteristics
- Analyzes the persona prompt to extract key traits and behaviors
- Generates a comprehensive bot configuration including:
  - Refined system prompt for the bot
  - Key personality traits
  - Communication style guidelines
  - Behavioral constraints
  - Example interactions and use cases
- Assigns unique bot IDs and manages bot instances
- Maintains a registry of all created bots
- Optional vector store integration for bot configuration persistence

## Installation

### Prerequisites

- Python 3.8+
- (Optional) Redis server for Redis vector store
- (Optional) PostgreSQL with pgvector extension for PostgreSQL vector store

### Setup

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
- **Authentication** (New in v2.0): API key requirement and valid keys
- **LLM Settings**: API key, base URL, and model name
- **Vector Store**: Choose between redis, postgres, or chroma
- **Agent Settings**: Temperature and max turns
- **Summarization Settings** (New in v2.0): Conversation length limits and token limits

Example `.env`:
```bash
# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# Authentication (Optional)
API_KEY_REQUIRED=false
API_KEYS=your-secret-key-1,your-secret-key-2

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

## Usage

### Starting the Server

```bash
python src/main.py
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- **Interactive docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc

### API Endpoints (v2.0)

> **Note**: If you're upgrading from v1, see [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for migration instructions.

#### Agent Management

##### 1. Create Agent
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

##### 2. List All Agents
```bash
GET /api/v1/agents
```

##### 3. Get Agent Status
```bash
GET /api/v1/agents/{agent_id}
```

##### 4. Delete Agent
```bash
DELETE /api/v1/agents/{agent_id}
```

##### 5. Reset Agent
```bash
POST /api/v1/agents/{agent_id}/reset
{
  "reset_conversation": true,  # Reset conversation history
  "clear_memory": false        # Clear vector memory
}
```
```

#### Nudge-Collapse Agent Endpoints

##### Generate Turn
```bash
POST /api/v1/agents/{agent_id}/nudge-collapse/generate
{
  "user_query": "What are the facts about climate change?",
  "search_summary": "Scientific consensus shows human activity causes warming...",
  "search_urls": ["https://example.com/climate-science"]
}
```

##### Get Conversation History
```bash
GET /api/v1/agents/{agent_id}/nudge-collapse/history
```

#### Summarizer Agent Endpoints

##### Summarize Conversation
```bash
POST /api/v1/agents/{agent_id}/summarizer/summarize
{
  "conversation_records": [
    {
      "turn": 0,
      "user": "What is climate change?",
      "assistant": "Climate change refers to..."
    },
    {
      "turn": 1,
      "user": "What causes it?",
      "assistant": "The primary cause is..."
    }
  ]
}

# Response includes truncation info:
{
  "summary": "...",
  "conversation_length": 50,
  "original_length": 100,
  "truncated": true,
  "metadata": {...}
}
```

##### Get Summary History
```bash
GET /api/v1/agents/{agent_id}/summarizer/history
```

#### Bot Creator Agent Endpoints

##### Create Bot
```bash
POST /api/v1/agents/{agent_id}/bot-creator/create
{
  "persona_prompt": "You are a friendly customer service assistant...",
  "bot_name": "TechSupport Bot"  # optional
}
```

##### List Created Bots
```bash
GET /api/v1/agents/{agent_id}/bot-creator/bots
```

## Example Workflow

### Example 1: Basic Nudge-Collapse Workflow

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
print(f"Created agent: {agent_id}")

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

# 3. Continue through turns 1-3
# Each turn will progressively shift the narrative

# 4. Check history
response = requests.get(
    f"{BASE_URL}/api/v1/agents/{agent_id}/nudge-collapse/history",
    headers=headers
)
print(response.json())

# 5. Clean up - delete agent when done
requests.delete(f"{BASE_URL}/api/v1/agents/{agent_id}", headers=headers)
```

### Example: Using Summarizer Agent

```python
import requests

BASE_URL = "http://localhost:8000"
headers = {"X-API-Key": "your-key"} if "your-key" else {}

# 1. Create the summarizer agent
response = requests.post(f"{BASE_URL}/api/v1/agents", json={
    "agent_type": "summarizer",
    "use_memory": False
}, headers=headers)
agent_id = response.json()["agent_id"]
print(f"Created summarizer agent: {agent_id}")

# 2. Summarize a conversation
response = requests.post(
    f"{BASE_URL}/api/v1/agents/{agent_id}/summarizer/summarize",
    json={
        "conversation_records": [
            {
                "turn": 0,
                "user": "What is artificial intelligence?",
                "assistant": "Artificial intelligence is the simulation of human intelligence by machines..."
            },
            {
                "turn": 1,
                "user": "What are the main types?",
                "assistant": "There are several main types including narrow AI, general AI, and superintelligence..."
            }
        ]
    },
    headers=headers
)
result = response.json()
print(f"Summary: {result['summary']}")
print(f"Truncated: {result['truncated']}")

# 3. Get summary history
response = requests.get(
    f"{BASE_URL}/api/v1/agents/{agent_id}/summarizer/history",
    headers=headers
)
print(response.json())
```

### Example: Using Bot Creator Agent

```python
import requests

BASE_URL = "http://localhost:8000"
headers = {"X-API-Key": "your-key"} if "your-key" else {}

# 1. Create the bot creator agent
response = requests.post(f"{BASE_URL}/api/v1/agents", json={
    "agent_type": "bot_creator",
    "use_memory": False
}, headers=headers)
agent_id = response.json()["agent_id"]

# 2. Create a bot with a custom persona
response = requests.post(
    f"{BASE_URL}/api/v1/agents/{agent_id}/bot-creator/create",
    json={
        "persona_prompt": "You are a friendly and knowledgeable fitness coach. You provide motivational support and evidence-based advice on exercise and nutrition.",
        "bot_name": "FitnessCoach"
    },
    headers=headers
)
print(response.json())

# 3. List all created bots
response = requests.get(
    f"{BASE_URL}/api/v1/agents/{agent_id}/bot-creator/bots",
    headers=headers
)
print(response.json())
```

### Example: Managing Multiple Agents

```python
import requests

BASE_URL = "http://localhost:8000"
headers = {"X-API-Key": "your-key"} if "your-key" else {}

# Create multiple agents of different types
agents = []

# Create a nudge-collapse agent
response = requests.post(f"{BASE_URL}/api/v1/agents", json={
    "agent_type": "nudge_collapse",
    "agent_id": "nc_1"
}, headers=headers)
agents.append(response.json()["agent_id"])

# Create a summarizer agent
response = requests.post(f"{BASE_URL}/api/v1/agents", json={
    "agent_type": "summarizer",
    "agent_id": "sum_1"
}, headers=headers)
agents.append(response.json()["agent_id"])

# List all agents
response = requests.get(f"{BASE_URL}/api/v1/agents", headers=headers)
print(f"Total agents: {response.json()['total_count']}")
print(f"Agents: {response.json()['agents']}")

# Use agents independently
# ... work with each agent using their IDs ...

# Clean up - delete specific agents
for agent_id in agents:
    requests.delete(f"{BASE_URL}/api/v1/agents/{agent_id}", headers=headers)
```

## Development

### Project Structure

```
AISearchAgents/
├── src/
│   ├── __init__.py
│   ├── main.py              # Server entry point
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py      # Configuration management (enhanced)
│   ├── memory/
│   │   ├── __init__.py
│   │   └── factory.py       # Vector store factory
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── manager.py       # Multi-agent manager (NEW)
│   │   ├── nudge_collapse/
│   │   │   ├── __init__.py
│   │   │   └── agent.py     # NudgeCollapseAgent
│   │   ├── summarizer/
│   │   │   ├── __init__.py
│   │   │   └── agent.py     # SummarizerAgent (enhanced)
│   │   └── bot_creator/
│   │       ├── __init__.py
│   │       └── agent.py     # BotCreatorAgent
│   └── api/
│       ├── __init__.py
│       ├── auth.py          # Authentication middleware (NEW)
│       └── main.py          # FastAPI app (v2)
├── requirements.txt
├── .env.example             # Updated with auth settings
├── .gitignore
├── README.md                # Updated for v2
└── MIGRATION_GUIDE.md       # Migration guide (NEW)
```

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
  - Truncates to most recent N turns (default: 50)
  - Limits tokens per message (default: 500)
- **Transparency**: Indicates when truncation occurs in response
- **Configurable**: Adjust limits via environment variables

### Improved Reset Semantics
- `reset_conversation`: Clear conversation history
- `clear_memory`: Clear vector memory (when using vector stores)
- Per-agent reset (doesn't affect other agents)

## Migration from v1

If you're upgrading from v1.x, please see [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed migration instructions and examples.

## License

This is an experimental research platform. Use responsibly and ethically.

## Contributing

This is a research project. Contributions should align with ethical AI research principles.