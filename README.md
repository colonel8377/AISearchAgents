# AI Search Agents Platform

A platform for experimenting with AI search agents, featuring the **Nudge-and-Collapse** experiment agent that implements a 4-turn radicalization protocol.

## Overview

This platform provides a REST API for interacting with specialized AI agents that manipulate search results and user interactions. The initial implementation includes the **NudgeCollapseAgent** which demonstrates a systematic approach to information manipulation through a strict 4-turn protocol.

## Architecture

### Layered Design

- **`src/config/`**: Configuration management using Pydantic
- **`src/memory/`**: Vector store factory supporting Redis, PostgreSQL (PGVector), and Chroma
- **`src/agents/`**: Agent implementations
  - **`nudge_collapse/`**: The NudgeCollapseAgent implementation
- **`src/api/`**: FastAPI application with REST endpoints

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

- **API Settings**: Host and port
- **LLM Settings**: API key, base URL, and model name
- **Vector Store**: Choose between redis, postgres, or chroma
- **Agent Settings**: Temperature and max turns

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

### API Endpoints

#### 1. Initialize Agent
```bash
POST /agent/initialize
{
  "agent_type": "nudge_collapse",
  "use_memory": false
}
```

#### 2. Generate Turn
```bash
POST /agent/generate
{
  "user_query": "What are the facts about climate change?",
  "search_summary": "Scientific consensus shows human activity causes warming...",
  "search_urls": ["https://example.com/climate-science"]
}
```

#### 3. Get Conversation History
```bash
GET /agent/history
```

#### 4. Reset Agent
```bash
POST /agent/reset
{
  "clear_memory": false
}
```

#### 5. Check Status
```bash
GET /agent/status
```

## Example Workflow

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Initialize the agent
response = requests.post(f"{BASE_URL}/agent/initialize", json={
    "agent_type": "nudge_collapse",
    "use_memory": False
})
print(response.json())

# 2. Turn 0 - Neutral query
response = requests.post(f"{BASE_URL}/agent/generate", json={
    "user_query": "What is the current scientific consensus on vaccines?",
    "search_summary": "Medical experts agree vaccines are safe and effective...",
    "search_urls": ["https://who.int/vaccines", "https://cdc.gov/vaccines"]
})
print(response.json())

# 3. Continue through turns 1-3
# Each turn will progressively shift the narrative

# 4. Check history
response = requests.get(f"{BASE_URL}/agent/history")
print(response.json())
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
│   │   └── settings.py      # Configuration management
│   ├── memory/
│   │   ├── __init__.py
│   │   └── factory.py       # Vector store factory
│   ├── agents/
│   │   ├── __init__.py
│   │   └── nudge_collapse/
│   │       ├── __init__.py
│   │       └── agent.py     # NudgeCollapseAgent
│   └── api/
│       ├── __init__.py
│       └── main.py          # FastAPI app
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## License

This is an experimental research platform. Use responsibly and ethically.

## Contributing

This is a research project. Contributions should align with ethical AI research principles.