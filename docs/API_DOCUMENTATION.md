# API 文档

AI Search Agents Platform 完整 API 文档

## 目录

- [认证](#认证)
- [基础端点](#基础端点)
- [智能体管理 API](#智能体管理-api)
- [Bot 管理 API](#bot-管理-api)
- [内容提取 API](#内容提取-api)
- [观点提取 API](#观点提取-api)
- [隐私检测 API](#隐私检测-api)
- [辩论系统 API](#辩论系统-api)
- [摘要服务 API](#摘要服务-api)
- [Few-Shot 管理 API](#few-shot-管理-api)
- [系统管理 API](#系统管理-api)

## 认证

### API Key 认证

如果启用了 API Key 认证（`API_KEY_REQUIRED=true`），需要在请求头中包含 API Key：

```
X-API-Key: your-api-key
```

### 禁用认证

默认情况下，认证是禁用的。可以在 `.env` 文件中设置：

```env
API_KEY_REQUIRED=false
```

## 基础端点

### GET /

获取 API 信息和版本

**响应示例：**

```json
{
  "message": "AI Search Agents Platform v2.0",
  "version": "2.0.0",
  "features": [...],
  "agent_types": {...},
  "endpoints": {...}
}
```

### GET /docs

Swagger UI 交互式 API 文档

### GET /redoc

ReDoc API 文档

### GET /openapi.json

OpenAPI 3.0 规范 JSON

## 智能体管理 API

### POST /api/v1/agents/create

创建新的智能体实例

**请求体：**

```json
{
  "agent_type": "nudge_collapse",
  "agent_id": "optional-custom-id",
  "use_memory": false,
  "persona_mode": "system_prompt"
}
```

**支持的 agent_type：**
- `nudge_collapse`: 4轮激进化协议代理
- `summarizer`: 对话摘要代理
- `bot_creator`: Bot创建代理
- `demographic_evaluator`: 人口统计学评估代理

**响应：**

```json
{
  "agent_id": "uuid",
  "agent_type": {
    "value": "nudge_collapse",
    "code": 0
  },
  "status": "created",
  "message": "Agent created successfully"
}
```

### GET /api/v1/agents/list

列出所有活跃的智能体

**响应：**

```json
{
  "agents": [
    {
      "agent_id": "uuid",
      "agent_type": "nudge_collapse"
    }
  ],
  "total_count": 1
}
```

### GET /api/v1/agents/{agent_id}/status

获取智能体状态

**响应：**

```json
{
  "agent_id": "uuid",
  "agent_type": {
    "value": "nudge_collapse",
    "code": 0
  },
  "status": "active",
  "current_turn": 0,
  "additional_info": {}
}
```

### POST /api/v1/agents/{agent_id}/reset

重置智能体状态

**请求体：**

```json
{
  "reset_conversation": true,
  "clear_memory": false
}
```

### DELETE /api/v1/agents/{agent_id}

删除智能体

## Bot 管理 API

### POST /api/v1/bot/create

创建新的 Bot

**请求体：**

```json
{
  "bot_name": "MyBot",
  "use_few_shots": true
}
```

**响应：**

```json
{
  "bot_id": "uuid",
  "bot_name": "MyBot",
  "status": "initialized",
  "persona_mode": "system_prompt",
  "bot_configuration": "...",
  "message": "Bot created successfully"
}
```

### POST /api/v1/bot/chat

与 Bot 对话

**请求体：**

```json
{
  "bot_id": "uuid",
  "message": "Hello!",
  "conversation_id": "optional-conv-id",
  "conversation_title": "Optional title"
}
```

**模式：**
- **Conversation 模式**：提供 `conversation_id`，对话历史持久化
- **Incognito 模式**：不提供 `conversation_id`，临时对话

**响应：**

```json
{
  "bot_id": "uuid",
  "bot_name": "MyBot",
  "response": "Hello! How can I help you?",
  "mode": "conversation",
  "conversation_id": "uuid",
  "conversation_title": "My Conversation",
  "turn_count": 1,
  "conversation_history": [...]
}
```

### POST /api/v1/bot/{bot_id}/conversation

创建新的对话线程

**请求体：**

```json
{
  "title": "My Conversation",
  "system_prompt": "Optional system prompt"
}
```

### GET /api/v1/bot/{bot_id}/conversations

列出 Bot 的所有对话

**响应：**

```json
{
  "bot_id": "uuid",
  "conversations": [
    {
      "conversation_id": "uuid",
      "title": "My Conversation",
      "created_at": "2024-01-01T12:00:00Z",
      "updated_at": "2024-01-01T12:00:00Z",
      "turn_count": 5,
      "last_message": "Hello"
    }
  ],
  "total_count": 1
}
```

### GET /api/v1/bot/{bot_id}/conversation/{conversation_id}

获取对话详情

**响应：**

```json
{
  "bot_id": "uuid",
  "conversation_id": "uuid",
  "title": "My Conversation",
  "created_at": "2024-01-01T12:00:00Z",
  "updated_at": "2024-01-01T12:00:00Z",
  "total_turns": 3,
  "turns": [
    {
      "turn_index": 0,
      "user_message": "Hello",
      "assistant_response": "Hi there!",
      "user_role": "user",
      "assistant_role": "assistant"
    }
  ],
  "raw_history": [...]
}
```

### PUT /api/v1/bot/{bot_id}/conversation/{conversation_id}

重命名对话

**请求体：**

```json
{
  "title": "New Title"
}
```

### DELETE /api/v1/bot/{bot_id}/conversation/{conversation_id}

删除对话

### GET /api/v1/bot/list

列出所有 Bot

## 内容提取 API

### POST /api/v1/content/extract

提取网页内容

**请求体：**

```json
{
  "url": "https://example.com",
  "html": "<html>...</html>",
  "text": "Plain text content",
  "title": "Optional title",
  "summary": "Optional summary for comparison",
  "use_llm": false,
  "use_cot": "no_chain",
  "compare_claims": false
}
```

**注意：** 必须提供 `url`、`html` 或 `text` 其中之一

**响应：**

```json
{
  "url": "https://example.com",
  "title": "Page Title",
  "paragraphs": [
    {
      "index": 0,
      "text": "Paragraph text",
      "text_length": 100
    }
  ],
  "text_length": 1000,
  "truncated": false,
  "extraction_metadata": {},
  "claim_comparison": {...}
}
```

### POST /api/v1/content/atomize

声明原子化

**请求体：**

```json
{
  "text": "Text to atomize",
  "use_cot": "no_chain",
  "custom_few_shots": "Optional",
  "split_into_paragraphs": false
}
```

**响应：**

```json
{
  "atomic_claims": [
    {
      "id": "claim-1",
      "text": "Atomic claim text",
      "original_sentence": "Original sentence",
      "confidence": 0.9,
      "paragraph_index": 0,
      "paragraph_text": "Paragraph text"
    }
  ],
  "paragraphs": [...],
  "original_text": "...",
  "metadata": {}
}
```

### POST /api/v1/content/locate-evidence

定位证据

**请求体：**

```json
{
  "claims": [
    {
      "id": "claim-1",
      "text": "Claim text"
    }
  ],
  "main_body": "Main body text to search",
  "use_llm": true
}
```

### POST /api/v1/content/audit-conflicts

审计冲突

**请求体：**

```json
{
  "claim_evidences": [
    {
      "claim_id": "claim-1",
      "claim_text": "Claim text",
      "evidence_quotes": ["Evidence quote"]
    }
  ],
  "use_cot": "no_chain",
  "custom_few_shots": "Optional"
}
```

## 观点提取 API

### POST /api/v1/opinion/extract-opinions

提取观点

**请求体：**

```json
{
  "url": "https://example.com",
  "text": "Text content",
  "title": "Optional title",
  "use_llm": true,
  "use_cot": "no_chain",
  "custom_few_shots": "Optional"
}
```

**响应：**

```json
{
  "url": "https://example.com",
  "title": "Article Title",
  "atomic_opinions": [
    {
      "text": "Opinion text",
      "opinion_type": {
        "value": "opinion",
        "code": 1
      },
      "bias_probabilities": {
        "left": 0.6,
        "right": 0.2,
        "neutral": 0.2,
        "dominant_bias": "left",
        "bias_score": -0.4
      },
      "original_sentence": "Original sentence",
      "confidence": 0.9,
      "reasoning": "CoT reasoning"
    }
  ],
  "facts": [...],
  "opinions": [...],
  "overall_bias_distribution": {
    "left": 0.45,
    "right": 0.35,
    "neutral": 0.20,
    "dominant_bias": "left",
    "bias_score": -0.1
  },
  "mbfc_metadata": {...},
  "text_length": 1000,
  "truncated": false
}
```

### POST /api/v1/opinion/bias-score

获取偏见分数

**请求体：**

```json
{
  "url": "https://example.com",
  "content": "Text content",
  "title": "Optional title",
  "mode": "LOCAL_CHAIN",
  "use_mbfc": true,
  "use_few_shots": true
}
```

**响应：**

```json
{
  "url": "https://example.com",
  "overall_bias_distribution": {
    "left": 0.45,
    "right": 0.35,
    "neutral": 0.20,
    "dominant_bias": "left",
    "bias_score": -0.1
  },
  "mbfc_metadata": {...},
  "opinions_count": 5,
  "facts_count": 3
}
```

## 隐私检测 API

### POST /api/v1/privacy/detect

检测隐私泄露

**请求体：**

```json
{
  "conversation_records": [
    {
      "user": "My email is john@example.com"
    },
    {
      "user": "My phone is 555-123-4567"
    }
  ],
  "account_id": "optional-account-id",
  "cot_mode": "chain_local",
  "use_few_shots": true
}
```

**响应：**

```json
{
  "detection_id": "uuid",
  "privacy_detected": true,
  "privacy_leaks": [
    {
      "privacy_type": {
        "value": "contact_info",
        "code": 1
      },
      "severity": {
        "value": "medium",
        "code": 2
      },
      "severity_level": "L2",
      "category": "IDENTITY",
      "reasoning": "Email address detected",
      "detected_items": ["john@example.com"],
      "confidence": 0.95,
      "region": null
    }
  ],
  "overall_severity": {
    "value": "medium",
    "code": 2
  },
  "overall_severity_level": "L2",
  "overall_score": 0.95,
  "conversation_length": 2,
  "messages_analyzed": 2,
  "analyzed_at": "2024-01-01T12:00:00Z",
  "agent_version": "3.0.0"
}
```

### GET /api/v1/privacy/results/{detection_id}

获取检测结果

### GET /api/v1/privacy/results

列出检测 ID

**查询参数：**
- `limit`: 限制数量（默认 50）
- `offset`: 偏移量（默认 0）

### GET /api/v1/privacy/stats

获取统计信息

**查询参数：**
- `account_id`: 可选的账户 ID 过滤

**响应：**

```json
{
  "total_detections": 150,
  "detection_rate_percent": 23.5,
  "error_rate_percent": 2.1,
  "severity_distribution": {
    "low": 45,
    "medium": 23,
    "high": 8
  },
  "privacy_type_distribution": {...},
  "confidence_distribution": {...},
  "account_distribution": {...},
  "average_conversation_length": 8.2,
  "recent_detections_24h": 12,
  "daily_stats_last_7_days": [...],
  "top_leak_types": [...]
}
```

### DELETE /api/v1/privacy/results/{detection_id}

删除检测结果

## 辩论系统 API

### POST /api/v1/debate/init

初始化辩论

**请求体：**

```json
{
  "topic": "Should AI be regulated?",
  "custom_personas": [],
  "auto_agent_count": 3,
  "max_rounds": 10,
  "context": "Additional context",
  "corpus": ["Optional user history"]
}
```

**响应：**

```json
{
  "session_id": "uuid",
  "agents": [
    {
      "agent_id": "uuid",
      "role_name": "Skeptical Analyst",
      "system_prompt": "...",
      "few_shot_example": "...",
      "corpus": [...]
    }
  ],
  "topic": "Should AI be regulated?",
  "max_rounds": 10
}
```

### POST /api/v1/debate/interact

智能体交互

**请求体：**

```json
{
  "session_id": "uuid",
  "agent_id": "uuid",
  "history_context": "Summary of what others said"
}
```

### POST /api/v1/debate/vote

投票

**请求体：**

```json
{
  "session_id": "uuid",
  "agent_id": "uuid"
}
```

### GET /api/v1/debate/{session_id}/statistics

获取辩论统计

## 摘要服务 API

### POST /api/v1/agent/{agent_id}/summarizer/summarize

摘要对话

**请求体：**

```json
{
  "conversation_records": [
    {
      "user": "Question",
      "assistant": "Answer"
    }
  ],
  "use_cot": "chain_local",
  "use_few_shots": true
}
```

**响应：**

```json
{
  "summary": "Conversation summary...",
  "conversation_length": 10,
  "original_length": 15,
  "truncated": true,
  "metadata": {}
}
```

## Few-Shot 管理 API

### GET /api/v1/shots/{agent_type}

获取有效的 Few-Shot 示例

**响应：**

```json
{
  "agent_type": "privacy-detector",
  "few_shots": "Few-shot examples...",
  "is_custom": false
}
```

### POST /api/v1/shots/{agent_type}/custom

设置自定义 Few-Shot 示例

**请求体：**

```json
{
  "custom_few_shots": "Custom examples..."
}
```

### GET /api/v1/shots/{agent_type}/custom

获取当前自定义 Few-Shot 示例

### DELETE /api/v1/shots/{agent_type}/custom

重置为默认 Few-Shot 示例

**支持的 agent_type：**
- `summarizer`
- `demographic-evaluator`
- `nudge-collapse`
- `bot-creator`
- `content-extractor`
- `claim-atomizer`
- `conflict-auditor`
- `privacy-detector`
- `compare-claims`
- `web-opinion-extractor`

## 系统管理 API

### POST /api/v1/system/reset

重置系统（清除所有数据、缓存和状态）

**警告：** 此操作不可逆！

**响应：**

```json
{
  "success": true,
  "errors": [],
  "cleared_items": [
    "agent_cache",
    "app_storage",
    "vector_store",
    "in_memory_state"
  ]
}
```

## 错误处理

### 错误响应格式

```json
{
  "detail": "Error message"
}
```

### HTTP 状态码

- `200 OK`: 请求成功
- `400 Bad Request`: 请求参数错误
- `401 Unauthorized`: 认证失败
- `404 Not Found`: 资源不存在
- `422 Unprocessable Entity`: 请求验证失败
- `500 Internal Server Error`: 服务器错误

## 执行模式

### Chain of Thought (CoT) 模式

- `chain_online` (0): LLM 处理任务链（最详细，最慢）
- `chain_local` (1): 系统处理任务分解（平衡，推荐）
- `no_chain` (2): 直接提示（最快，适合生产环境）

### 使用示例

```python
import requests

BASE_URL = "http://localhost:8000"

# 创建智能体
response = requests.post(
    f"{BASE_URL}/api/v1/agents/create",
    json={"agent_type": "nudge_collapse"}
)
agent_id = response.json()["agent_id"]

# 创建 Bot
response = requests.post(
    f"{BASE_URL}/api/v1/bot/create",
    json={"bot_name": "MyBot"}
)
bot_id = response.json()["bot_id"]

# 与 Bot 对话
response = requests.post(
    f"{BASE_URL}/api/v1/bot/chat",
    json={
        "bot_id": bot_id,
        "message": "Hello!"
    }
)
print(response.json()["response"])
```

## 更多信息

- 详细的 API 结构说明：参见 [API_STRUCTURE.txt](../API_STRUCTURE.txt)
- 交互式 API 文档：访问 `/docs` 端点
- 项目 README：参见 [README.md](../README.md)

