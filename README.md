# AI Search Agents Platform

一个基于 FastAPI 的多智能体 AI 搜索平台，支持多种 AI 代理类型，提供 RESTful API 接口。

## 📋 目录

- [项目简介](#项目简介)
- [架构设计](#架构设计)
- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [API 文档](#api-文档)
- [代理类型](#代理类型)
- [配置说明](#配置说明)
- [开发指南](#开发指南)
- [测试](#测试)

## 项目简介

AI Search Agents Platform 是一个企业级的多智能体 AI 平台，采用 CQRS 架构模式和领域驱动设计（DDD），提供以下核心功能：

- **多智能体管理**：支持创建和管理多种类型的 AI 代理
- **对话系统**：支持 Bot 创建、对话管理和持久化存储
- **内容分析**：网页内容提取、观点提取、偏见分析
- **隐私检测**：P-Guard 3.0 隐私泄露检测系统
- **学术分析**：声明原子化、证据定位、冲突审计
- **辩论系统**：多智能体辩论与共识检测

## 架构设计

### 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                  Presentation Layer                     │
│              (FastAPI Routers & Schemas)                  │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                Application Layer                         │
│  (Services, Commands, Queries, DTOs, Business Logic)   │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              Infrastructure Layer                        │
│  (Repositories, Storage, Vector Stores, Cache)          │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                 Shared Layer                             │
│  (LLM, Config, Utils, Parsers, Constants)              │
└─────────────────────────────────────────────────────────┘
```

### 目录结构

```
AISearchAgents/
├── src/
│   ├── main.py                    # 应用入口
│   ├── application/               # 应用层
│   │   ├── agents/               # 各种代理实现
│   │   ├── commands/             # CQRS 命令
│   │   ├── queries/              # CQRS 查询
│   │   ├── dto/                  # 数据传输对象
│   │   ├── services/             # 业务服务
│   │   └── few_shots/            # Few-shot 示例
│   ├── infrastructure/           # 基础设施层
│   │   ├── repositories/         # 数据仓库
│   │   ├── storage/              # 持久化存储
│   │   ├── state/                # 状态管理
│   │   └── smart_memory/         # 智能记忆
│   ├── presentation/             # 表现层
│   │   └── api/                  # API 路由和模式
│   └── shared/                   # 共享层
│       ├── config/               # 配置管理
│       ├── llm/                  # LLM 管理
│       ├── cache/                # 缓存
│       └── utils/                # 工具函数
├── test/                         # 测试文件
├── data/                         # 数据文件
└── docs/                         # 文档
```

### 设计模式

- **CQRS (Command Query Responsibility Segregation)**：命令和查询分离
- **Repository Pattern**：数据访问抽象
- **DTO Pattern**：数据传输对象
- **Factory Pattern**：代理工厂模式
- **Strategy Pattern**：执行模式策略（chain_online, chain_local, no_chain）

## 功能特性

### 核心功能

1. **智能体管理**
   - 创建、查询、删除智能体
   - 支持多种智能体类型
   - 智能体状态管理和重置

2. **Bot 对话系统**
   - Bot 创建和配置
   - 持久化对话线程
   - 临时对话模式（Incognito）
   - 对话历史管理

3. **内容提取与分析**
   - 网页内容提取
   - 声明原子化
   - 证据定位
   - 冲突审计

4. **观点提取与偏见分析**
   - 原子观点提取
   - 政治偏见分析（左/右/中立）
   - MBFC 数据库集成

5. **隐私检测 (P-Guard 3.0)**
   - 多类型隐私信息检测
   - 严重程度评估
   - 位置追踪和验证

6. **多智能体辩论系统**
   - 自动生成角色
   - 多轮辩论
   - 稳定性检测和共识分析

7. **Few-Shot 管理**
   - 集中化配置
   - 自定义示例
   - 代理级别配置

## 快速开始

### 环境要求

- Python 3.9+
- Redis (可选，用于向量存储和缓存)
- PostgreSQL (可选，用于向量存储)
- SQLite (默认，用于本地存储)

### 安装步骤

1. **克隆项目**

```bash
git clone <repository-url>
cd AISearchAgents
```

2. **安装依赖**

```bash
pip install -r requirements.txt
```

3. **安装 spaCy 模型**

```bash
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
```

4. **配置环境变量**

创建 `.env` 文件：

```env
# API 配置
API_HOST=0.0.0.0
API_PORT=8000

# 认证配置
API_KEY_REQUIRED=false
API_KEYS=your-api-key-1,your-api-key-2

# LLM 配置
OPENAI_API_KEY=your-api-key
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo

# 向量存储配置
VECTOR_STORE_TYPE=chroma  # 可选: redis, postgres, chroma

# Redis 配置（如果使用）
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# PostgreSQL 配置（如果使用）
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password
POSTGRES_DB=vectordb
```

5. **启动服务**

```bash
python src/main.py
```

或使用 uvicorn：

```bash
uvicorn src.presentation.api.main:app --host 0.0.0.0 --port 8000 --reload
```

6. **访问 API 文档**

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 文档

### 基础端点

- `GET /` - 根端点，返回 API 信息
- `GET /docs` - Swagger UI 文档
- `GET /redoc` - ReDoc 文档

### 主要 API 端点

#### 1. 智能体管理 (`/api/v1/agents`)

- `POST /api/v1/agents/create` - 创建智能体
- `GET /api/v1/agents/list` - 列出所有智能体
- `GET /api/v1/agents/{agent_id}/status` - 获取智能体状态
- `POST /api/v1/agents/{agent_id}/reset` - 重置智能体
- `DELETE /api/v1/agents/{agent_id}` - 删除智能体

#### 2. Bot 管理 (`/api/v1/bot`)

- `POST /api/v1/bot/create` - 创建 Bot
- `POST /api/v1/bot/chat` - 与 Bot 对话
- `POST /api/v1/bot/{bot_id}/conversation` - 创建对话
- `GET /api/v1/bot/{bot_id}/conversations` - 列出对话
- `GET /api/v1/bot/{bot_id}/conversation/{conversation_id}` - 获取对话详情

#### 3. 内容提取 (`/api/v1/content`)

- `POST /api/v1/content/extract` - 提取网页内容
- `POST /api/v1/content/atomize` - 声明原子化
- `POST /api/v1/content/locate-evidence` - 定位证据
- `POST /api/v1/content/audit-conflicts` - 审计冲突

#### 4. 观点提取 (`/api/v1/opinion`)

- `POST /api/v1/opinion/extract-opinions` - 提取观点
- `POST /api/v1/opinion/bias-score` - 获取偏见分数
- `POST /api/v1/opinion/analyze` - 完整分析

#### 5. 隐私检测 (`/api/v1/privacy`)

- `POST /api/v1/privacy/detect` - 检测隐私泄露
- `GET /api/v1/privacy/results/{detection_id}` - 获取检测结果
- `GET /api/v1/privacy/stats` - 获取统计信息

#### 6. 辩论系统 (`/api/v1/debate`)

- `POST /api/v1/debate/init` - 初始化辩论
- `POST /api/v1/debate/interact` - 智能体交互
- `POST /api/v1/debate/vote` - 投票
- `GET /api/v1/debate/{session_id}/statistics` - 获取统计信息

#### 7. 系统管理 (`/api/v1/system`)

- `POST /api/v1/system/reset` - 重置系统

详细的 API 文档请参考 [API_STRUCTURE.txt](API_STRUCTURE.txt) 或访问 `/docs` 端点。

## 代理类型

### 1. Nudge Collapse Agent
4 轮激进化协议代理，用于分析用户查询的激进化趋势。

### 2. Summarizer Agent
对话摘要代理，专注于用户问题和关键信息。

### 3. Bot Creator Agent
Bot 创建和配置代理，支持自定义角色和系统提示。

### 4. Demographic Evaluator Agent
从人口统计学角度评估句子的代理。

### 5. Content Extractor Agent
学术内容提取代理，从网页提取主要内容。

### 6. Claim Atomizer Agent
声明原子化代理，将文本分解为原子声明。

### 7. Conflict Auditor Agent
冲突审计代理，检查声明和证据之间的逻辑一致性。

### 8. Privacy Detector Agent (P-Guard 3.0)
隐私检测代理，检测多种类型的隐私信息泄露。

### 9. Web Opinion Extractor Agent
网络观点提取代理，提取观点并分析政治偏见。

## 配置说明

### 执行模式

- `chain_online`: LLM 处理任务链（最详细，最慢）
- `chain_local`: 系统处理任务分解（平衡，推荐）
- `no_chain`: 直接提示（最快，适合生产环境）

### 向量存储

支持三种向量存储后端：

- **Chroma** (默认): 本地文件存储
- **Redis**: 高性能内存存储
- **PostgreSQL**: 持久化关系数据库

### 缓存配置

- `cache_backend`: `local` (SQLite) 或 `redis`
- `use_llm_cache`: 启用 LLM API 调用缓存

## 开发指南

### 代码结构

项目采用分层架构：

1. **Presentation Layer**: 处理 HTTP 请求/响应
2. **Application Layer**: 业务逻辑和 CQRS
3. **Infrastructure Layer**: 数据访问和外部服务
4. **Shared Layer**: 共享工具和配置

### 添加新代理

1. 在 `src/application/agents/` 创建代理目录
2. 实现代理类
3. 在 `AgentFactory` 中注册
4. 创建对应的路由和模式
5. 添加 Few-shot 示例

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest test/test_agents.py

# 运行并显示覆盖率
pytest --cov=src --cov-report=html
```

## 测试

项目包含完整的测试套件，包括：

- 单元测试
- 集成测试
- API 测试

详见 [test/](test/) 目录。

## 许可证

[添加许可证信息]

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

[添加联系方式]
