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

# Proxy Configuration (for API requests)
# Set these environment variables to configure HTTP proxies
HTTP_PROXY=http://your-proxy-server:port
HTTPS_PROXY=http://your-proxy-server:port
```

## Core Features

- **Content Analysis**: Extract, atomize, and verify claims against sources
- **Multi-Agent System**: Nudge-collapse, summarizer, bot creator, synthesis aggregator agents
- **Consistency Checking**: Verify if website summaries accurately reflect their content
- **Debate Simulation**: Multi-agent debates with stability analysis
- **Quality Assessment**: Content metrics and bias detection
- **Privacy Detection**: Analyze conversations for privacy leaks with LLM hallucination prevention
- **Bot Chat System**: Create custom AI personas and chat in persistent conversation threads or incognito mode

## API Structure

```
/api/v1/
├── agents/          # Agent management
├── agent/           # Agent operations
├── bot/             # Custom AI chatbots
│   ├── create               # Create custom bot with persona
│   ├── chat                 # Chat with bot (conversation/incognito modes)
│   ├── {bot_id}/conversation # Conversation thread management
│   └── {bot_id}/conversations # List/manage conversations
├── content/         # Content processing
├── consistency/     # Website summary vs content verification
│   ├── check-summary-url    # Verify summary accuracy
│   ├── compare-claims       # Compare two specific claims
│   └── complete             # Full 5-step academic analysis pipeline
├── privacy-detector/# Privacy leak detection
│   ├── detect               # Analyze conversations for privacy leaks
│   ├── results              # Get/list detection results
│   ├── stats                # Comprehensive privacy statistics
│   └── results/{id}         # Individual result management
├── quality/         # Quality assessment
│   └── overall              # Comprehensive quality evaluation
├── opinion/         # Bias analysis
└── debate/          # Debate system
```

## Examples

### Bot Chat System

Create a custom AI assistant and chat with it:

```python
import requests

# 1. Create a coding assistant bot (persona auto-generated)
bot_response = requests.post("http://localhost:8000/api/v1/bot/create", json={
    "bot_name": "PythonExpert",
    "execution_mode": "chain_local",
    "use_few_shots": True
})
bot_id = bot_response.json()["bot_id"]
print(f"Created bot: {bot_id}")

# 2. Create a conversation thread
conv_response = requests.post(f"http://localhost:8000/api/v1/bot/{bot_id}/conversation", json={
    "title": "Python Learning Session"
})
conversation_id = conv_response.json()["conversation_id"]

# 3. Chat in the conversation thread (remembers history)
chat_response = requests.post("http://localhost:8000/api/v1/bot/chat", json={
    "bot_id": bot_id,
    "message": "How do I create a list in Python?",
    "conversation_id": conversation_id
})
print(chat_response.json()["response"])

# Continue the conversation
chat_response = requests.post("http://localhost:8000/api/v1/bot/chat", json={
    "bot_id": bot_id,
    "message": "How do I add items to it?",
    "conversation_id": conversation_id  # Same conversation - bot remembers
})
print(chat_response.json()["response"])

# 4. Incognito chat (no history saved)
incognito_response = requests.post("http://localhost:8000/api/v1/bot/chat", json={
    "bot_id": bot_id,
    "message": "What is machine learning?"
    # No conversation_id = incognito mode
})
print("Incognito response:", incognito_response.json()["response"])
```

### Privacy Detection System

Analyze user conversations for potential privacy leaks with advanced LLM hallucination prevention:

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Detect privacy leaks in conversation (with anti-hallucination validation)
response = requests.post(f"{BASE_URL}/api/v1/privacy-detector/detect", json={
    "conversation_records": [
        {"user": "My email is john@example.com and SSN is 123-45-6789"},
        {"user": "The password for my bank account is 'secret123'"},
        {"user": "Call me at 555-123-4567 if you need anything"}
    ],
    "account_id": "user123",  # Optional: group by account
    "execution_mode": "chain_local",
    "use_few_shots": True
})
result = response.json()

print(f"Privacy leaks detected: {len(result['privacy_leaks'])}")
print(f"Overall severity: {result['overall_severity']}")

# Each leak includes validation information
for leak in result['privacy_leaks']:
    print(f"Type: {leak['privacy_type']}, Confidence: {leak['confidence']}")
    if leak['confidence'] == 'invalid':
        print(f"  Warning: {leak['validation_error']}")

# 2. Get comprehensive privacy statistics
stats_response = requests.get(f"{BASE_URL}/api/v1/privacy-detector/stats")
stats = stats_response.json()

print(f"Total detections: {stats['total_detections']}")
print(f"Detection rate: {stats['detection_rate_percent']}%")
print(f"Most common leak type: {stats['top_leak_types'][0]['privacy_type']}")

# 3. Get account-specific statistics
account_stats = requests.get(f"{BASE_URL}/api/v1/privacy-detector/stats?account_id=user123")
account_data = account_stats.json()
print(f"User123 detections: {account_data['total_detections']}")
```

### Content Analysis

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
        "use_cot_atomization": "chain_local",  # Options: "chain_online", "chain_local", "no_chain" or 0, 1, 2
        "use_cot_audit": "chain_local",       # Options: "chain_online", "chain_local", "no_chain" or 0, 1, 2
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

# 5. Create and chat with bot (persona auto-generated)
resp = requests.post(f"{BASE_URL}/api/v1/bot/create",
    json={
        "bot_name": "ML Assistant"
    })
bot_result = resp.json()
bot_id = bot_result["bot_id"]
print(f"Created bot: {bot_id}")

# Create a new conversation thread
resp = requests.post(f"{BASE_URL}/api/v1/bot/{bot_id}/conversation",
    json={
        "title": "Machine Learning Study Session",
        "system_prompt": "You are an expert machine learning tutor who explains concepts clearly with examples."
    })
conv_result = resp.json()
conversation_id = conv_result["conversation_id"]
print(f"Created conversation: {conversation_id}")

# Chat with bot in conversation thread (remembers history)
resp = requests.post(f"{BASE_URL}/api/v1/bot/chat",
    json={
        "bot_id": bot_id,
        "conversation_id": conversation_id,  # Continue this conversation thread
        "message": "What is gradient descent?"
    })
chat_result = resp.json()
print(f"Bot response: {chat_result['response']}")

# Chat in incognito mode (no history saved)
resp = requests.post(f"{BASE_URL}/api/v1/bot/chat",
    json={
        "bot_id": bot_id,
        "message": "What is overfitting?"
        # No conversation_id = incognito mode
    })
chat_result = resp.json()
print(f"Incognito response: {chat_result['response']}")

# Continue conversation (history maintained)
resp = requests.post(f"{BASE_URL}/api/v1/bot/chat",
    json={
        "bot_id": bot_id,
        "conversation_id": conversation_id,  # Same conversation
        "message": "Can you explain it with an example?"
    })
chat_result = resp.json()
print(f"Bot continues: {chat_result['response']}")

# Chat in incognito mode (no conversation_id - bot has no memory)
resp = requests.post(f"{BASE_URL}/api/v1/bot/chat",
    json={
        "bot_id": bot_id,
        # No conversation_id - incognito mode
        "message": "What is overfitting?"
    })
incognito_result = resp.json()
print(f"Incognito response: {incognito_result['response']}")

# Get conversation history
resp = requests.get(f"{BASE_URL}/api/v1/bot/{bot_id}/conversation")
history = resp.json()
print(f"Total turns: {history['total_turns']}")
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

### Bot Chat System
- `POST /api/v1/bot/create` - Create custom AI bot with persona
- `POST /api/v1/bot/chat` - Chat with bot (conversation threads or incognito mode)
- `GET /api/v1/bot/list` - List all created bots
- `POST /api/v1/bot/{bot_id}/conversation` - Create new conversation thread
- `GET /api/v1/bot/{bot_id}/conversations` - List bot's conversation threads
- `GET /api/v1/bot/{bot_id}/conversation/{conversation_id}` - Get conversation details
- `PUT /api/v1/bot/{bot_id}/conversation/{conversation_id}` - Rename conversation
- `DELETE /api/v1/bot/{bot_id}/conversation/{conversation_id}` - Delete conversation
- `DELETE /api/v1/bot/{bot_id}` - Delete bot

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

### Privacy Detection
- `POST /api/v1/privacy-detector/detect` - Analyze conversations for privacy leaks with validation
- `GET /api/v1/privacy-detector/results/{detection_id}` - Get specific detection result
- `GET /api/v1/privacy-detector/results` - List detection results
- `GET /api/v1/privacy-detector/stats` - Get comprehensive privacy statistics
- `DELETE /api/v1/privacy-detector/results/{detection_id}` - Delete detection result

### Few-Shot Management
- `GET /api/v1/shots/{agent_type}` - Get effective few-shot examples for any agent
- `POST /api/v1/shots/{agent_type}/custom` - Set custom few-shot examples
- `GET /api/v1/shots/{agent_type}/custom` - Get current custom few-shot examples
- `DELETE /api/v1/shots/{agent_type}/custom` - Reset to default few-shot examples

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

### Bot Management
- `POST /api/v1/bot/create` - Create new bot with auto-generated persona
- `GET /api/v1/bot/list` - List all created bots
- `POST /api/v1/bot/{bot_id}/conversation` - Create new conversation for bot
- `POST /api/v1/bot/chat` - Chat with bot (conversation mode or incognito mode)
- `GET /api/v1/bot/{bot_id}/conversation` - Get bot conversation history
- `POST /api/v1/bot/{bot_id}/conversation/turn` - Add conversation turn manually
- `DELETE /api/v1/bot/{bot_id}/conversation/turn` - Delete specific conversation turn
- `DELETE /api/v1/bot/{bot_id}/conversations` - Clear all conversation threads
- `DELETE /api/v1/bot/{bot_id}` - Delete bot

### System Management
- `POST /api/v1/system/reset` - Reset entire system (clear all databases, caches, and state)

Use responsibly. Research platform only.
