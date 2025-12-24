# AI Search Agents Platform

AI-powered content analysis platform with multi-agent support for academic research and debate simulation.

## Quick Start

```bash
git clone https://github.com/colonel8377/AISearchAgents.git && cd AISearchAgents
pip install -r requirements.txt
cp .env.example .env  # Configure your API keys
python src/main.py
```

API: http://localhost:8000 | Docs: http://localhost:8000/docs

## Configuration

```bash
# Required
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-3.5-turbo

# Optional
API_HOST=0.0.0.0
API_PORT=8000
VECTOR_STORE_TYPE=chroma
```

## Core Features

- **Content Analysis**: Extract, atomize, and verify claims against sources
- **Multi-Agent System**: Nudge-collapse, summarizer, bot creator agents
- **Consistency Checking**: Verify if website summaries accurately reflect their content
- **Debate Simulation**: Multi-agent debates with stability analysis
- **Quality Assessment**: Content metrics and bias detection

## API Structure

```
/api/v1/
├── agents/          # Agent management
├── agent/           # Agent operations
├── content/         # Content processing
├── consistency/     # Website summary vs content verification
│   ├── check-summary-url    # Verify summary accuracy
│   ├── complete             # Full 5-step analysis
│   └── evidence-locate      # Evidence location
├── quality/         # Quality assessment
├── opinion/         # Bias analysis
└── debate/          # Debate system
```

## Examples

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Create summarizer agent
resp = requests.post(f"{BASE_URL}/api/v1/agents/create",
    json={"agent_type": "summarizer"})
agent_id = resp.json()["agent_id"]

# 2. Summarize conversation
resp = requests.post(f"{BASE_URL}/api/v1/agent/{agent_id}/summarizer/summarize",
    json={"conversation_records": [
        {"role": "user", "content": "What is AI?"},
        {"role": "assistant", "content": "AI is artificial intelligence..."}
    ]})
print("Summary:", resp.json()["summary"])

# 3. Check website summary vs full content consistency
resp = requests.post(f"{BASE_URL}/api/v1/consistency/check-summary-url",
    json={
        "summary": "This article explores how AI is revolutionizing healthcare delivery systems.",
        "url": "https://example.com/ai-healthcare-article",
        "enable_deep_analysis": True
    })
result = resp.json()
print(f"Summary-content consistency: {result['consistency_score']:.2f}")
if result['conflicting_points']:
    print(f"Found {len(result['conflicting_points'])} inconsistencies")
```

## Documentation

- [API Reference](doc/QUICK_REFERENCE.md)
- [Consistency Pipeline](doc/HISTORY_MODES.md)
- [Debate System](doc/DEBATE_SYSTEM.md)

Use responsibly. Research platform only.
