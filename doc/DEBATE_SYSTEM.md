# Multi-Agent Debate System

A RESTful API service for orchestrating multi-agent debates with adaptive stability detection. This system decouples agents so they can be orchestrated externally, incorporating concepts from two research papers: **ChatEval** (Personas) and **Adaptive Stability** (Stopping Logic).

## Overview

The Multi-Agent Debate System allows you to:
- Create debate sessions with custom or auto-generated agent personas
- Each agent has a unique personality, system prompt, and few-shot example
- Interact with agents through a RESTful API
- Track debate stability using statistical analysis (KS statistic)

## Tech Stack

- **FastAPI**: Web server and RESTful API
- **Pydantic**: Strict data validation for all requests/responses
- **UUID**: Unique agent identification
- **SciPy**: Statistical analysis for stability detection

## Architecture

### Components

1. **schemas.py**: Pydantic models for data validation
   - `PersonaConfig`: Agent persona configuration
   - `AgentMetadata`: Agent metadata with prompts and examples
   - `InitRequest`: Debate initialization request
   - `InteractRequest`: Agent interaction request
   - `VoteResponse`: Agent vote and reasoning

2. **service.py**: Core business logic
   - Agent factory with persona-based prompt generation
   - Session management
   - Stability calculation using KS statistic
   - LLM caller abstraction (dependency injection)

3. **API Endpoints** (in `main.py`):
   - `POST /debate/init`: Initialize a debate session
   - `POST /agent/{agent_id}/chat`: Interact with a specific agent
   - `POST /debate/{session_id}/stability_check`: Check debate stability

## API Documentation

### 1. Initialize a Debate Session

**Endpoint**: `POST /debate/init`

**Request Body**:
```json
{
  "topic": "Should we invest more in renewable energy?",
  "custom_personas": [
    {
      "name": "Dr. Skeptic",
      "description": "A critical analyst who questions assumptions",
      "style": "Critical"
    },
    {
      "name": "Ms. Optimist",
      "description": "An enthusiastic supporter of innovation",
      "style": "Supportive"
    }
  ],
  "auto_agent_count": 0
}
```

**Or with auto-generation**:
```json
{
  "topic": "Should we invest more in renewable energy?",
  "custom_personas": [],
  "auto_agent_count": 3
}
```

**Response**:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "topic": "Should we invest more in renewable energy?",
  "agents": [
    {
      "agent_id": "123e4567-e89b-12d3-a456-426614174000",
      "role_name": "Dr. Skeptic",
      "system_prompt": "You are Dr. Skeptic, a debate participant with the following characteristics: A critical analyst who questions assumptions. Your role is to critically evaluate proposals, identify potential flaws, and ensure rigorous analysis. You ask tough questions and demand evidence for claims.",
      "few_shot_example": "Example output format:\n{\n    \"verdict\": 0,\n    \"reasoning\": \"While the proposal has merit, there are several critical flaws...\"\n}"
    }
  ]
}
```

### 2. Interact with an Agent

**Endpoint**: `POST /agent/{agent_id}/chat`

**Request Body**:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "agent_id": "123e4567-e89b-12d3-a456-426614174000",
  "history_context": "Other agents have expressed support for renewable energy, citing environmental benefits and long-term cost savings. However, concerns about initial investment costs were raised."
}
```

**Response**:
```json
{
  "agent_id": "123e4567-e89b-12d3-a456-426614174000",
  "verdict": 0,
  "reasoning": "While renewable energy has environmental benefits, the economic analysis presented lacks rigor. The initial investment costs are significantly understated, and the time to ROI is uncertain. We need more concrete data before committing to such a large expenditure."
}
```

### 3. Check Debate Stability

**Endpoint**: `POST /debate/{session_id}/stability_check`

**Request Body**:
```json
{
  "votes": [0, 1, 2, 1]
}
```

**Response**:
```json
{
  "stable": false
}
```

The stability check returns `true` when:
- At least 3 rounds of votes have been recorded
- The KS statistic comparing the last two round transitions is < 0.05
- This indicates the debate has converged

## Usage Examples

### Example 1: Basic Debate Flow

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Initialize a debate
response = requests.post(f"{BASE_URL}/debate/init", json={
    "topic": "Should AI development be regulated?",
    "custom_personas": [],
    "auto_agent_count": 3
})

data = response.json()
session_id = data["session_id"]
agents = data["agents"]

print(f"Created debate session: {session_id}")
print(f"Agents: {[a['role_name'] for a in agents]}")

# 2. Round 1: Get initial positions
round_1_votes = []
for agent in agents:
    response = requests.post(
        f"{BASE_URL}/agent/{agent['agent_id']}/chat",
        json={
            "session_id": session_id,
            "agent_id": agent['agent_id'],
            "history_context": "This is the opening round. Please share your initial position."
        }
    )
    vote_data = response.json()
    round_1_votes.append(vote_data['verdict'])
    print(f"{agent['role_name']}: {vote_data['reasoning'][:100]}...")

# 3. Check stability after round 1
response = requests.post(
    f"{BASE_URL}/debate/{session_id}/stability_check",
    json={"votes": round_1_votes}
)
print(f"Stable after round 1: {response.json()['stable']}")

# 4. Round 2: Agents respond to each other
history_context = "In the first round, we saw diverse opinions ranging from cautious to supportive of AI regulation."
round_2_votes = []

for agent in agents:
    response = requests.post(
        f"{BASE_URL}/agent/{agent['agent_id']}/chat",
        json={
            "session_id": session_id,
            "agent_id": agent['agent_id'],
            "history_context": history_context
        }
    )
    vote_data = response.json()
    round_2_votes.append(vote_data['verdict'])

# 5. Continue until stable
response = requests.post(
    f"{BASE_URL}/debate/{session_id}/stability_check",
    json={"votes": round_2_votes}
)
print(f"Stable after round 2: {response.json()['stable']}")
```

### Example 2: Custom Personas

```python
import requests

BASE_URL = "http://localhost:8000"

# Create a debate with specialized personas
response = requests.post(f"{BASE_URL}/debate/init", json={
    "topic": "How should we approach climate change mitigation?",
    "custom_personas": [
        {
            "name": "Dr. Climate Scientist",
            "description": "A climate researcher with deep knowledge of atmospheric science",
            "style": "Critical"
        },
        {
            "name": "Policy Maker",
            "description": "A pragmatic government official focused on feasibility",
            "style": "Neutral"
        },
        {
            "name": "Environmental Activist",
            "description": "A passionate advocate for immediate climate action",
            "style": "Supportive"
        },
        {
            "name": "Economist",
            "description": "An expert in economic impacts and cost-benefit analysis",
            "style": "Critical"
        }
    ],
    "auto_agent_count": 0
})

data = response.json()
print(f"Created debate with {len(data['agents'])} specialized agents")

# Each agent will have:
# - A unique system prompt tailored to their persona
# - A few-shot example matching their style
# - Consistent behavior aligned with their role
```

### Example 3: Monitoring Convergence

```python
import requests

def run_debate_until_stable(session_id, agents, max_rounds=10):
    """Run a debate until it reaches stability or max rounds."""
    
    BASE_URL = "http://localhost:8000"
    
    for round_num in range(max_rounds):
        print(f"\n=== Round {round_num + 1} ===")
        
        # Collect votes from all agents
        votes = []
        for agent in agents:
            response = requests.post(
                f"{BASE_URL}/agent/{agent['agent_id']}/chat",
                json={
                    "session_id": session_id,
                    "agent_id": agent['agent_id'],
                    "history_context": f"Round {round_num + 1}. Previous rounds have shown varying opinions."
                }
            )
            vote_data = response.json()
            votes.append(vote_data['verdict'])
            print(f"{agent['role_name']}: verdict={vote_data['verdict']}")
        
        # Check stability
        response = requests.post(
            f"{BASE_URL}/debate/{session_id}/stability_check",
            json={"votes": votes}
        )
        
        is_stable = response.json()['stable']
        print(f"Stability: {is_stable}")
        
        if is_stable:
            print(f"\nDebate converged after {round_num + 1} rounds!")
            return round_num + 1
    
    print(f"\nDebate did not converge within {max_rounds} rounds")
    return None
```

## Persona Styles

The system supports three main persona styles, each with tailored prompts and examples:

### Critical / Skeptical
- **System Prompt**: Emphasizes critical evaluation, questioning assumptions, demanding evidence
- **Few-Shot Example**: Shows cautious reasoning with identified flaws
- **Typical Verdicts**: Lower values, indicating skepticism

### Neutral / Balanced
- **System Prompt**: Emphasizes balanced analysis, weighing pros and cons
- **Few-Shot Example**: Shows measured reasoning considering multiple perspectives
- **Typical Verdicts**: Middle values, indicating balanced views

### Supportive / Optimistic
- **System Prompt**: Emphasizes identifying strengths, building on positives
- **Few-Shot Example**: Shows enthusiastic reasoning with constructive support
- **Typical Verdicts**: Higher values, indicating support

## Stability Detection

The system uses the **Kolmogorov-Smirnov (KS) statistic** to detect when a debate has reached stability:

1. **Vote History**: Each round, agents cast votes (integers or strings converted to integers)
2. **Distribution Comparison**: The KS test compares the distribution of votes between consecutive rounds
3. **Convergence Check**: If the KS statistic is < 0.05 for two consecutive round transitions, the debate is considered stable

**Why KS Statistic?**
- Detects both changes in central tendency and distribution shape
- Non-parametric (no assumptions about vote distribution)
- Sensitive to small changes while robust to noise

## Dependency Injection for LLM

The `DebateService` supports dependency injection for the LLM caller:

```python
from src.debate.service import DebateService

async def my_llm_caller(system_prompt: str, user_message: str) -> str:
    """Custom LLM implementation."""
    # Call OpenAI, Anthropic, or any other LLM
    response = await openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return response.choices[0].message.content

# Configure the service
debate_service = DebateService()
debate_service.set_llm_caller(my_llm_caller)
```

## Testing

The system includes comprehensive tests:

- **Unit Tests** (`test_debate.py`): Test schemas, service logic, and helper functions
- **Integration Tests** (`test_debate_api.py`): Test API endpoints end-to-end

Run tests:
```bash
pytest test_debate.py -v
pytest test_debate_api.py -v
```

## Future Enhancements

Potential improvements:
1. Persist sessions to database
2. Add support for streaming agent responses
3. Enhanced stability metrics (entropy, variance, etc.)
4. Agent memory and context accumulation
5. Multi-topic debate threads
6. Visualization of debate evolution

## References

- **ChatEval Paper**: Multi-agent personas for evaluation
- **Adaptive Stability Paper**: Statistical stopping criteria for convergence
- **KS Test**: Kolmogorov-Smirnov two-sample test for distribution comparison
