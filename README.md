# AI Search Agents Platform

A platform for experimenting with AI search agents, featuring multiple specialized agents including the **Nudge-and-Collapse** experiment agent, **Summarizer Agent**, and **Bot Creator Agent**.

## Overview

This platform provides a REST API for interacting with specialized AI agents. The platform currently supports three types of agents:
1. **NudgeCollapseAgent**: Demonstrates systematic information manipulation through a 4-turn protocol
2. **SummarizerAgent**: Summarizes conversation records and extracts key information
3. **BotCreatorAgent**: Creates and initializes bots with custom personas

## Architecture

### Layered Design

- **`src/config/`**: Configuration management using Pydantic
- **`src/memory/`**: Vector store factory supporting Redis, PostgreSQL (PGVector), and Chroma
- **`src/agents/`**: Agent implementations
  - **`nudge_collapse/`**: The NudgeCollapseAgent implementation
  - **`summarizer/`**: The SummarizerAgent implementation
  - **`bot_creator/`**: The BotCreatorAgent implementation
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

## Summarizer Agent

The SummarizerAgent analyzes conversation records and extracts key information:

- Accepts a list of conversation records as input
- Analyzes the conversation to identify main topics, themes, and patterns
- Generates a comprehensive summary highlighting:
  - Main topics discussed
  - Key decisions or conclusions reached
  - Important questions asked
  - Notable patterns or themes
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

#### Common Endpoints

##### 1. Initialize Agent
```bash
POST /agent/initialize
{
  "agent_type": "nudge_collapse",  # or "summarizer" or "bot_creator"
  "use_memory": false
}
```

##### 2. Reset Agent
```bash
POST /agent/reset
{
  "clear_memory": false
}
```

##### 3. Check Status
```bash
GET /agent/status
```

#### Nudge-Collapse Agent Endpoints

##### Generate Turn
```bash
POST /agent/generate
{
  "user_query": "What are the facts about climate change?",
  "search_summary": "Scientific consensus shows human activity causes warming...",
  "search_urls": ["https://example.com/climate-science"]
}
```

##### Get Conversation History
```bash
GET /agent/history
```

#### Summarizer Agent Endpoints

##### Summarize Conversation
```bash
POST /agent/summarize
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
```

#### Bot Creator Agent Endpoints

##### Create Bot
```bash
POST /agent/create_bot
{
  "persona_prompt": "You are a friendly customer service assistant with expertise in technical support. You are patient, helpful, and always maintain a positive attitude.",
  "bot_name": "TechSupport Bot"  # optional
}
```

##### List Bots
```bash
GET /agent/list_bots
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

### Example: Using Summarizer Agent

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Initialize the summarizer agent
response = requests.post(f"{BASE_URL}/agent/initialize", json={
    "agent_type": "summarizer",
    "use_memory": False
})
print(response.json())

# 2. Summarize a conversation
response = requests.post(f"{BASE_URL}/agent/summarize", json={
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
})
print(response.json())
```

### Example: Using Bot Creator Agent

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Initialize the bot creator agent
response = requests.post(f"{BASE_URL}/agent/initialize", json={
    "agent_type": "bot_creator",
    "use_memory": False
})
print(response.json())

# 2. Create a bot with a custom persona
response = requests.post(f"{BASE_URL}/agent/create_bot", json={
    "persona_prompt": "You are a friendly and knowledgeable fitness coach. You provide motivational support and evidence-based advice on exercise and nutrition.",
    "bot_name": "FitnessCoach"
})
print(response.json())

# 3. List all created bots
response = requests.get(f"{BASE_URL}/agent/list_bots")
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
│   │   ├── nudge_collapse/
│   │   │   ├── __init__.py
│   │   │   └── agent.py     # NudgeCollapseAgent
│   │   ├── summarizer/
│   │   │   ├── __init__.py
│   │   │   └── agent.py     # SummarizerAgent
│   │   └── bot_creator/
│   │       ├── __init__.py
│   │       └── agent.py     # BotCreatorAgent
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