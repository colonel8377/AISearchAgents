# Embedding Providers Architecture

This document describes the new generic embedding provider architecture that supports multiple AI providers with a clean, extensible design.

## Overview

The new architecture replaces the old hard-coded provider system with a flexible, plugin-based approach that supports:

- **OpenAI-compatible APIs** (OpenAI, Qwen, DeepSeek, Azure OpenAI, etc.)
- **Google Gemini**
- **Cohere**
- **HuggingFace**
- **Custom APIs**
- **Easy extension** for new providers

## Architecture Components

### 1. EmbeddingConfig

A dataclass that holds configuration for any embedding provider:

```python
@dataclass
class EmbeddingConfig:
    provider: str = ""          # Provider identifier
    model: str = ""             # Model name
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    proxy: Optional[str] = None
    custom_params: Dict[str, Any] = field(default_factory=dict)
```

### 2. BaseEmbeddingProvider (Abstract Base Class)

All providers inherit from this base class:

```python
class BaseEmbeddingProvider(ABC):
    def validate_config(self, config: EmbeddingConfig) -> None:
        """Validate configuration for this provider."""
        pass

    def create_embeddings(self, config: EmbeddingConfig) -> Embeddings:
        """Create embeddings instance."""
        pass
```

### 3. Concrete Provider Implementations

#### OpenAICompatibleProvider
For APIs compatible with OpenAI's interface:
- OpenAI (`openai`)
- Qwen (`qwen`)
- DeepSeek (`deepseek`)

```python
class OpenAICompatibleProvider(BaseEmbeddingProvider):
    def create_embeddings(self, config: EmbeddingConfig) -> Embeddings:
        return OpenAIEmbeddings(
            model=config.model,
            api_key=config.api_key,
            base_url=config.api_base,
            **config.custom_params
        )
```

#### GoogleGeminiProvider
For Google Gemini embeddings:

```python
class GoogleGeminiProvider(BaseEmbeddingProvider):
    def create_embeddings(self, config: EmbeddingConfig) -> Embeddings:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=config.model,
            google_api_key=config.api_key,
            **config.custom_params
        )
```

### 4. EmbeddingProviderRegistry

Manages provider registration and lookup:

```python
class EmbeddingProviderRegistry:
    @classmethod
    def register(cls, provider: EmbeddingProvider) -> None:
        """Register a provider."""

    @classmethod
    def get_provider(cls, provider_key: str) -> EmbeddingProvider:
        """Get provider by key."""
```

### 5. EmbeddingFactory

Main factory class for creating embeddings:

```python
class EmbeddingFactory:
    @classmethod
    def create_embeddings(cls, provider_key: Optional[str] = None) -> Embeddings:
        """Create embeddings using global settings."""

    @classmethod
    def register_provider(cls, provider: EmbeddingProvider) -> None:
        """Register a new provider."""
```

## Usage

### Basic Usage

```python
from memory.factory import EmbeddingFactory

# Create embeddings using global settings
embeddings = EmbeddingFactory.create_embeddings()

# Create embeddings for specific provider
embeddings = EmbeddingFactory.create_embeddings("gemini")
```

### Advanced Usage with Custom Config

```python
from memory.factory import EmbeddingConfig, EmbeddingFactory

config = EmbeddingConfig(
    provider="cohere",
    model="embed-english-v2.0",
    api_key="your-cohere-key",
    custom_params={"user_agent": "my-app"}
)

embeddings = EmbeddingFactory.create_embeddings_from_config(config)
```

### Adding Custom Providers

```python
from memory.factory import BaseEmbeddingProvider, EmbeddingFactory

class MyCustomProvider(BaseEmbeddingProvider):
    def __init__(self):
        super().__init__("My API", "myapi")

    def validate_config(self, config):
        if not config.api_key:
            raise ValueError("API key required")

    def create_embeddings(self, config):
        # Your implementation here
        return MyEmbeddingsClass(...)

# Register the provider
EmbeddingFactory.register_provider(MyCustomProvider())
```

## Built-in Providers

### OpenAI-compatible Providers

| Provider | Key | Default Base URL | Notes |
|----------|-----|------------------|-------|
| OpenAI | `openai` | `https://api.openai.com/v1` | Standard OpenAI API |
| Qwen | `qwen` | None (configured) | Alibaba Qwen API |
| DeepSeek | `deepseek` | None (configured) | DeepSeek API |

### Other Providers

| Provider | Key | Requirements | Notes |
|----------|-----|--------------|-------|
| Google Gemini | `gemini` | `langchain-google-genai` | Google AI API |
| Cohere | `cohere` | `langchain-community` | Cohere API |
| HuggingFace | `huggingface` | `langchain-huggingface` | HuggingFace Hub |

## Configuration

### Environment Variables

```bash
# Provider selection
EMBEDDING_PROVIDER=openai

# Model configuration
EMBEDDING_MODEL=text-embedding-ada-002

# API credentials
EMBEDDING_API_KEY=your_api_key
EMBEDDING_API_BASE=https://api.example.com/v1
EMBEDDING_PROXY=http://proxy.example.com:8080
```

### Provider-Specific Examples

#### OpenAI
```bash
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_API_KEY=sk-...
```

#### Qwen
```bash
EMBEDDING_PROVIDER=qwen
EMBEDDING_MODEL=text-embedding-v1
EMBEDDING_API_KEY=your-qwen-key
EMBEDDING_API_BASE=https://dashscope.aliyuncs.com/api/v1
```

#### Google Gemini
```bash
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=models/text-embedding-004
EMBEDDING_API_KEY=your-google-key
```

#### Cohere
```bash
EMBEDDING_PROVIDER=cohere
EMBEDDING_MODEL=embed-english-v2.0
EMBEDDING_API_KEY=your-cohere-key
```

## Extending the System

### Creating a New Provider

1. **Inherit from BaseEmbeddingProvider**:
```python
class MyProvider(BaseEmbeddingProvider):
    def __init__(self):
        super().__init__("My Provider", "myprovider")
```

2. **Implement required methods**:
```python
def validate_config(self, config: EmbeddingConfig) -> None:
    # Validate your provider's requirements
    pass

def create_embeddings(self, config: EmbeddingConfig) -> Embeddings:
    # Create and return embeddings instance
    return MyEmbeddingsClass(...)
```

3. **Register the provider**:
```python
EmbeddingFactory.register_provider(MyProvider())
```

### Best Practices

1. **Dependency Checking**: Use `_check_dependencies()` to verify optional dependencies
2. **Configuration Validation**: Always validate required parameters in `validate_config()`
3. **Error Handling**: Provide clear error messages for configuration issues
4. **Custom Parameters**: Support `config.custom_params` for advanced configuration
5. **Documentation**: Document supported models and configuration options

## Migration from Old System

The old system used hard-coded provider configurations. The new system is backward compatible - existing configurations will continue to work.

Old code:
```python
# Old way - hard-coded
embeddings = OpenAIEmbeddings(api_key=key, model=model)
```

New code:
```python
# New way - configurable
embeddings = EmbeddingFactory.create_embeddings()
```

## Benefits

1. **Extensibility**: Easy to add new providers without code changes
2. **Maintainability**: Clean separation of concerns
3. **Type Safety**: Strong typing with dataclasses and protocols
4. **Testability**: Each component can be tested independently
5. **Flexibility**: Support for custom parameters and configurations
6. **Error Handling**: Better validation and error reporting



