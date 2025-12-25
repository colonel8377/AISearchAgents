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

# Embedding Configuration (supports multiple providers)
EMBEDDING_PROVIDER=openai  # openai, qwen, gemini, deepseek
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_API_KEY=  # Leave empty to use OPENAI_API_KEY

# Optional
API_HOST=0.0.0.0
API_PORT=8000
VECTOR_STORE_TYPE=chroma
```

## Core Features

- **Content Analysis**: Extract, atomize, and verify claims against sources
- **Multi-Agent System**: Nudge-collapse, summarizer, bot creator, synthesis aggregator agents
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
│   ├── compare-claims       # Compare two specific claims
│   └── complete             # Full 5-step academic analysis pipeline
├── quality/         # Quality assessment
│   └── overall              # Comprehensive quality evaluation
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
if result.get('conflicting_points'):
    print(f"Found {len(result['conflicting_points'])} inconsistencies")

# 3b. Complete academic analysis pipeline (5-step)
resp = requests.post(f"{BASE_URL}/api/v1/consistency/complete",
    json={
        "url": "https://example.com/research-paper",
        "use_cot_atomization": True,
        "use_cot_audit": True,
        "use_llm_synthesis": True
    })
result = resp.json()
print(f"Pipeline completed: {result['overall_success']}")
print(f"Total time: {result['total_execution_time']:.2f}s")
for step in result['pipeline_steps']:
    print(f"  {step['step_name']}: {'✓' if step['success'] else '✗'} ({step['execution_time']:.2f}s)")

# 3c. Compare two specific claims
resp = requests.post(f"{BASE_URL}/api/v1/consistency/compare-claims",
    json={
        "summary_claim": "The study shows significant improvement",
        "url_claim": "Results indicate 70% improvement rate",
        "url_content": "Full article content for context..."
    })
result = resp.json()
print(f"Claim comparison: {result['comparison']['status']} (confidence: {result['comparison']['confidence']:.2f})")

# 4. Create synthesis aggregator agent for claim comparison
resp = requests.post(f"{BASE_URL}/api/v1/agents/create",
    json={"agent_type": "synthesis_aggregator"})
agent_id = resp.json()["agent_id"]

# Use the agent to aggregate conflict analyses
conflict_analyses = [
    {
        "claim_id": "1",
        "claim_text": "Study shows 70% improvement",
        "evidence_quotes": ["70% improvement observed"],
        "verdict": "supported",
        "conflict_type": "consistent",
        "analysis": "Direct evidence matches",
        "confidence": 0.95
    }
]
resp = requests.post(f"{BASE_URL}/api/v1/agent/{agent_id}/synthesis_aggregator/aggregate",
    json={"conflict_analyses": conflict_analyses, "use_llm_enhancement": True})
result = resp.json()
print(f"Synthesis confidence: {result['synthesis_report']['confidence_score']:.2f}")
```

## Few-shot Management

The platform provides comprehensive few-shot learning management for all AI agents. Each agent supports customizable few-shot examples that persist across API calls and application restarts.

### Supported Agents

All agents with LLM capabilities support few-shot learning:
- **nudge-collapse**: 4-turn radicalization protocol agent
- **demographic-evaluator**: Sentence evaluation from demographic perspectives
- **summarizer**: Conversation summarization agent
- **bot-creator**: Bot creation and configuration agent
- **content-extractor**: Academic content extraction from web pages
- **claim-atomizer**: Atomic claim decomposition from text
- **conflict-auditor**: Logical consistency auditing
- **web-opinion-extractor**: Opinion extraction with bias analysis

### API Endpoints

All few-shot management endpoints follow the unified `/api/v1/shots/{agent}/` pattern:

#### Get Effective Few-shot Examples
```python
# Get current effective few-shot examples (custom if set, otherwise default)
resp = requests.get(f"{BASE_URL}/api/v1/shots/summarizer")
shots = resp.json()
print("Current shots:", shots["few_shots"])
print("Using custom:", shots["is_custom"])
```

#### Set Custom Few-shot Examples
```python
# Set custom few-shot examples for an agent
custom_shots = """EXAMPLE 1:
Input: [conversation data]
Output: [summary format]

EXAMPLE 2:
Input: [another conversation]
Output: [another summary]"""

resp = requests.post(f"{BASE_URL}/api/v1/shots/summarizer/custom",
    json={"custom_few_shots": custom_shots})
print("Custom shots set successfully")
```

#### Get Current Custom Shots
```python
# Check what custom shots are currently set
resp = requests.get(f"{BASE_URL}/api/v1/shots/summarizer/custom")
custom_shots = resp.json()["custom_few_shots"]
if custom_shots:
    print("Custom shots are set")
else:
    print("Using default shots")
```

#### Reset to Default Shots
```python
# Reset to system default few-shot examples
resp = requests.delete(f"{BASE_URL}/api/v1/shots/summarizer/custom")
print("Reset to default shots")
```

### Usage in API Calls

All LLM-enabled endpoints support the `use_few_shots` parameter:

```python
# Enable few-shot learning
resp = requests.post(f"{BASE_URL}/api/v1/agent/{agent_id}/summarizer/summarize",
    json={
        "conversation_records": [...],
        "use_few_shots": True  # Use effective few-shot examples
    })

# Disable few-shot learning
resp = requests.post(f"{BASE_URL}/api/v1/content/atomize",
    json={
        "text": "Content to atomize",
        "atomization_type": "claims",
        "use_few_shots": False  # Skip few-shot examples
    })
```

### Persistence

- **Custom shots persist** across API calls and application restarts
- **Effective shots** = Custom shots (if set) or Default shots (fallback)
- **Reset functionality** allows reverting to system defaults anytime

### Default Quality

System-provided default few-shot examples are professionally crafted for each agent's specific use case, ensuring high-quality performance out-of-the-box while allowing full customization when needed.

## Documentation

- [API Structure (HTML)](docs/api_structure.html) - Interactive API documentation
- [Few-shot Management](docs/FEW_SHOT_MANAGEMENT.md) - Comprehensive few-shot learning guide
- [Embedding Providers](docs/EMBEDDING_PROVIDERS.md) - Multiple embedding provider support

## API Endpoints Summary

### Agent Management
- `POST /api/v1/agents/create` - Create new agent instance
- `GET /api/v1/agents/list` - List all active agents
- `GET /api/v1/agents/{agent_id}/status` - Get agent status
- `POST /api/v1/agents/{agent_id}/reset` - Reset agent state
- `DELETE /api/v1/agents/{agent_id}` - Delete agent

### Content Processing
- `POST /api/v1/content/extract` - Extract and clean web content
- `POST /api/v1/content/atomize` - Break text into atomic claims
- `GET /api/v1/content/shots` - Get content extractor few-shot examples
- `GET /api/v1/content/atomize-shots` - Get claim atomizer few-shot examples

### Consistency & Quality
- `POST /api/v1/consistency/check-summary-url` - Verify summary vs URL consistency
- `POST /api/v1/consistency/compare-claims` - Compare two specific claims
- `POST /api/v1/consistency/complete` - Full 5-step academic analysis pipeline
- `POST /api/v1/quality/overall` - Comprehensive quality evaluation

### Opinion Analysis
- `POST /api/v1/opinion/extract-clean` - Extract and clean HTML content
- `POST /api/v1/opinion/extract-opinions` - Extract atomic opinions with bias analysis
- `POST /api/v1/opinion/analyze` - Complete opinion analysis
- `POST /api/v1/opinion/bias-score` - Get bias score for content

### Debate System
- `POST /api/v1/debate/generate-personas` - Generate debate personas
- `POST /api/v1/debate/init` - Initialize debate session
- `POST /api/v1/debate/{session_id}/chat` - Send message in debate
- `POST /api/v1/debate/{session_id}/stability-check` - Check debate stability
- `GET /api/v1/debate/{session_id}/statistics` - Get debate statistics

### System Management
- `POST /api/v1/system/reset` - Reset entire system (clear all databases, caches, and state)

Use responsibly. Research platform only.
