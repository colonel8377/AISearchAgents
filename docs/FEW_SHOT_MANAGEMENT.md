# Few-shot Management Guide

This document provides comprehensive guidance on managing few-shot examples for all AI agents in the AI Search Agents platform.

## Overview

Few-shot learning enables AI models to perform better on specific tasks by providing examples of desired input-output patterns. The platform supports customizable few-shot examples for all agents with LLM capabilities.

## Supported Agents

The following agents support few-shot learning management:

| Agent | Description | Few-shot Type |
|-------|-------------|---------------|
| **nudge-collapse** | 4-turn radicalization protocol agent | Multi-turn conversation examples |
| **demographic-evaluator** | Sentence evaluation from demographic perspectives | Perspective-based judgment examples |
| **summarizer** | Conversation summarization agent | Summary format examples |
| **bot-creator** | Bot creation and configuration agent | Bot configuration examples |
| **content-extractor** | Academic content extraction from web pages | Content extraction pattern examples |
| **claim-atomizer** | Atomic claim decomposition from text | Claim decomposition examples |
| **conflict-auditor** | Logical consistency auditing | Conflict analysis examples |
| **web-opinion-extractor** | Opinion extraction with bias analysis | Opinion classification examples |

## API Architecture

All few-shot management uses the unified `/api/v1/shots/{agent}/` endpoint pattern:

### Endpoints by Agent

#### Nudge Collapse Agent
- `GET /api/v1/shots/nudge-collapse` - Get effective few-shot examples
- `POST /api/v1/shots/nudge-collapse/custom` - Set custom few-shot examples
- `GET /api/v1/shots/nudge-collapse/custom` - Get current custom shots
- `DELETE /api/v1/shots/nudge-collapse/custom` - Reset to defaults

#### Demographic Evaluator Agent
- `GET /api/v1/shots/demographic-evaluator` - Get effective few-shot examples
- `POST /api/v1/shots/demographic-evaluator/custom` - Set custom few-shot examples
- `GET /api/v1/shots/demographic-evaluator/custom` - Get current custom shots
- `DELETE /api/v1/shots/demographic-evaluator/custom` - Reset to defaults

#### Summarizer Agent
- `GET /api/v1/shots/summarizer` - Get effective few-shot examples
- `POST /api/v1/shots/summarizer/custom` - Set custom few-shot examples
- `GET /api/v1/shots/summarizer/custom` - Get current custom shots
- `DELETE /api/v1/shots/summarizer/custom` - Reset to defaults

#### Bot Creator Agent
- `GET /api/v1/shots/bot-creator` - Get effective few-shot examples
- `POST /api/v1/shots/bot-creator/custom` - Set custom few-shot examples
- `GET /api/v1/shots/bot-creator/custom` - Get current custom shots
- `DELETE /api/v1/shots/bot-creator/custom` - Reset to defaults

#### Content Extractor Agent
- `GET /api/v1/shots/content-extractor` - Get effective few-shot examples
- `POST /api/v1/shots/content-extractor/custom` - Set custom few-shot examples
- `GET /api/v1/shots/content-extractor/custom` - Get current custom shots
- `DELETE /api/v1/shots/content-extractor/custom` - Reset to defaults

#### Claim Atomizer Agent
- `GET /api/v1/shots/claim-atomizer` - Get effective few-shot examples
- `POST /api/v1/shots/claim-atomizer/custom` - Set custom few-shot examples
- `GET /api/v1/shots/claim-atomizer/custom` - Get current custom shots
- `DELETE /api/v1/shots/claim-atomizer/custom` - Reset to defaults

#### Conflict Auditor Agent
- `GET /api/v1/shots/conflict-auditor` - Get effective few-shot examples
- `POST /api/v1/shots/conflict-auditor/custom` - Set custom few-shot examples
- `GET /api/v1/shots/conflict-auditor/custom` - Get current custom shots
- `DELETE /api/v1/shots/conflict-auditor/custom` - Reset to defaults

#### Web Opinion Extractor Agent
- `GET /api/v1/shots/web-opinion-extractor` - Get effective few-shot examples
- `POST /api/v1/shots/web-opinion-extractor/custom` - Set custom few-shot examples
- `GET /api/v1/shots/web-opinion-extractor/custom` - Get current custom shots
- `DELETE /api/v1/shots/web-opinion-extractor/custom` - Reset to defaults

## Usage Examples

### Python Client Examples

#### Managing Summarizer Few-shot Examples

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Check current few-shot examples
resp = requests.get(f"{BASE_URL}/api/v1/shots/summarizer")
current = resp.json()
print(f"Using custom shots: {current['is_custom']}")
print(f"Current shots length: {len(current['few_shots'])}")

# 2. Set custom few-shot examples
custom_shots = """EXAMPLE 1:
Input: User asked about AI, assistant explained basics
Output: User inquired about AI fundamentals, received comprehensive overview

EXAMPLE 2:
Input: Technical discussion about neural networks
Output: In-depth technical exchange on neural network architectures"""

resp = requests.post(f"{BASE_URL}/api/v1/shots/summarizer/custom",
    json={"custom_few_shots": custom_shots})
print("Custom shots set successfully")

# 3. Use in summarization API
resp = requests.post(f"{BASE_URL}/api/v1/agent/{agent_id}/summarizer/summarize",
    json={
        "conversation_records": [...],
        "use_few_shots": True  # Will use our custom shots
    })

# 4. Reset to defaults if needed
resp = requests.delete(f"{BASE_URL}/api/v1/shots/summarizer/custom")
print("Reset to default shots")
```

#### Managing Claim Atomizer Examples

```python
# Set custom claim decomposition examples
custom_atomizer_shots = """EXAMPLE 1:
Input: "The study found that 70% of participants showed improvement after treatment."
Output:
- CLAIM_1: The study found that 70% of participants showed improvement after treatment.

EXAMPLE 2:
Input: "Dr. Smith conducted research at Harvard University and published findings in Nature journal."
Output:
- CLAIM_1: Dr. Smith conducted research at Harvard University.
- CLAIM_2: Dr. Smith published findings in Nature journal."""

resp = requests.post(f"{BASE_URL}/api/v1/shots/claim-atomizer/custom",
    json={"custom_few_shots": custom_atomizer_shots})

# Use in atomization
resp = requests.post(f"{BASE_URL}/api/v1/content/atomize",
    json={
        "text": "Content to decompose",
        "atomization_type": "claims",
        "use_few_shots": True
    })
```

### cURL Examples

#### Get Current Few-shot Examples
```bash
curl -X GET "http://localhost:8000/api/v1/shots/summarizer" \
  -H "X-API-Key: your-api-key"
```

#### Set Custom Few-shot Examples
```bash
curl -X POST "http://localhost:8000/api/v1/shots/summarizer/custom" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "custom_few_shots": "EXAMPLE 1:\nInput: [...]\nOutput: [...]"
  }'
```

#### Reset to Defaults
```bash
curl -X DELETE "http://localhost:8000/api/v1/shots/summarizer/custom" \
  -H "X-API-Key: your-api-key"
```

## Integration with API Calls

All LLM-enabled endpoints accept the `use_few_shots` parameter:

### Agent Operations
```python
# With few-shot learning enabled
resp = requests.post(f"{BASE_URL}/api/v1/agent/{agent_id}/summarizer/summarize",
    json={
        "conversation_records": [...],
        "use_few_shots": True  # Uses effective few-shot examples
    })

# With few-shot learning disabled
resp = requests.post(f"{BASE_URL}/api/v1/agent/{agent_id}/summarizer/summarize",
    json={
        "conversation_records": [...],
        "use_few_shots": False  # No few-shot examples used
    })
```

### Content Processing
```python
# Atomization with few-shot examples
resp = requests.post(f"{BASE_URL}/api/v1/content/atomize",
    json={
        "text": "Content to process",
        "atomization_type": "claims",
        "use_few_shots": True
    })

# Content extraction with custom shots
resp = requests.post(f"{BASE_URL}/api/v1/content/extract",
    json={
        "url": "https://example.com",
        "use_llm": True,
        "use_few_shots": True
    })
```

### Consistency Pipeline
```python
# Conflict auditing with few-shot examples
resp = requests.post(f"{BASE_URL}/api/v1/consistency/conflict-audit",
    json={
        "claim_evidences": [...],
        "use_few_shots": True
    })
```

## Persistence and State Management

### Custom Shots Persistence
- Custom few-shot examples are stored in class variables
- They persist across API calls and application restarts
- No database storage required - purely in-memory for the class

### Effective Shots Logic
```python
def get_effective_few_shots():
    return custom_shots if custom_shots else default_shots
```

### Reset Functionality
- `DELETE /api/v1/shots/{agent}/custom` resets to system defaults
- All agents revert to their professionally crafted default examples
- Safe operation that can be performed anytime

## Default Few-shot Quality

The platform provides high-quality, professionally crafted default few-shot examples:

### Summarizer Agent
- Multiple conversation summarization examples
- Structured output format with key insights
- Covers various conversation types and complexities

### Claim Atomizer Agent
- Clear decomposition examples
- Atomic claim identification patterns
- Handles complex sentences and compound claims

### Demographic Evaluator Agent
- Cross-demographic perspective examples
- Detailed reasoning for each judgment
- Covers diverse viewpoints and backgrounds

### Nudge Collapse Agent
- Multi-turn conversation examples
- Neutral, balanced response patterns
- Comprehensive coverage of sensitive topics

### Content Extractor Agent
- Academic content extraction patterns
- Noise filtering examples
- Structure preservation techniques

### Conflict Auditor Agent
- Logical consistency analysis examples
- Evidence evaluation patterns
- Conflict type classification

### Synthesis Aggregator Agent
- Comprehensive report generation examples
- Statistical analysis integration
- Quality assessment patterns

## Best Practices

### Custom Few-shot Creation
1. **Match Format**: Follow the exact input/output format of default examples
2. **Quality Over Quantity**: 2-3 high-quality examples better than many poor ones
3. **Domain Relevance**: Use examples relevant to your specific use case
4. **Clear Patterns**: Ensure examples demonstrate clear, consistent patterns

### Performance Optimization
1. **Start with Defaults**: Test performance with default examples first
2. **Iterative Refinement**: Gradually improve custom examples based on results
3. **A/B Testing**: Compare performance with and without custom shots
4. **Regular Updates**: Update examples as your use cases evolve

### Management Guidelines
1. **Version Control**: Keep track of different versions of custom shots
2. **Documentation**: Document what each custom example is intended to achieve
3. **Backup**: Export custom shots before major changes
4. **Team Coordination**: Share effective custom shots across team members

## Troubleshooting

### Common Issues

#### Shots Not Taking Effect
- Check that `use_few_shots: true` is set in API calls
- Verify custom shots were set successfully via GET endpoint
- Ensure the agent supports few-shot learning

#### Poor Performance After Custom Shots
- Reset to defaults and verify default shots work
- Check custom shot format matches expected pattern
- Simplify custom examples - fewer, higher quality examples often work better

#### Persistence Issues
- Custom shots are stored in class variables and persist across restarts
- If shots are lost, they may have been reset or the application restarted with code changes
- Always verify shots are set after changes

### Debug Commands

```python
# Check if custom shots are set
resp = requests.get(f"{BASE_URL}/api/v1/shots/{agent}/custom")
print("Custom shots:", resp.json()["custom_few_shots"] is not None)

# Verify effective shots
resp = requests.get(f"{BASE_URL}/api/v1/shots/{agent}")
effective = resp.json()
print("Effective shots length:", len(effective["few_shots"]))
print("Is custom:", effective["is_custom"])
```

## Advanced Usage

### Bulk Management
```python
# Export all custom shots
agents = ["summarizer", "claim-atomizer", "conflict-auditor"]
custom_configs = {}

for agent in agents:
    resp = requests.get(f"{BASE_URL}/api/v1/shots/{agent}/custom")
    custom_configs[agent] = resp.json()["custom_few_shots"]

# Save to file
import json
with open('custom_shots_backup.json', 'w') as f:
    json.dump(custom_configs, f)
```

### Automated Testing
```python
def test_shots_effectiveness(agent, test_cases):
    """Test custom shots vs defaults"""
    results = {}

    # Test with custom shots
    for case in test_cases:
        resp = requests.post(f"{BASE_URL}/api/v1/agent/{agent_id}/{agent}/process",
            json={**case, "use_few_shots": True})
        results[f"custom_{case['id']}"] = resp.json()

    # Test with defaults
    resp = requests.delete(f"{BASE_URL}/api/v1/shots/{agent}/custom")  # Reset to defaults

    for case in test_cases:
        resp = requests.post(f"{BASE_URL}/api/v1/agent/{agent_id}/{agent}/process",
            json={**case, "use_few_shots": True})
        results[f"default_{case['id']}"] = resp.json()

    return results
```

## API Reference

For complete API specifications, see the main API documentation. All few-shot management endpoints return standard HTTP status codes and JSON responses.





