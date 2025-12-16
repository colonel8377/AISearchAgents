# Proxy Configuration Migration Guide

## Issue Fixed

**Error**: `TypeError: Client.__init__() got an unexpected keyword argument 'proxies'`

This error occurred when creating `SummarizerAgent` or `NudgeCollapseAgent` with a proxy configuration. The issue was caused by passing the `proxies` parameter to `httpx.Client()`, which caused compatibility issues with certain versions of the OpenAI SDK and langchain-openai.

## Changes Made

### Before (Deprecated)
```python
# Using OPENAI_PROXY setting or proxy parameter
agent = SummarizerAgent(
    model_name='gpt-3.5-turbo',
    api_key='your-key',
    proxy='http://127.0.0.1:7890'  # Deprecated approach
)
```

### After (Recommended)
```python
# Set environment variables before running the application
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

# Then create the agent normally
agent = SummarizerAgent(
    model_name='gpt-3.5-turbo',
    api_key='your-key'
    # No proxy parameter needed
)
```

## Migration Steps

### 1. Update Environment Variables

Instead of using `OPENAI_PROXY` in your `.env` file, use the standard proxy environment variables:

**Old `.env` configuration:**
```bash
OPENAI_PROXY=http://127.0.0.1:7890
```

**New `.env` configuration:**
```bash
# OPENAI_PROXY is now deprecated (will be ignored)
# Use standard environment variables instead:
HTTP_PROXY=http://127.0.0.1:7890
HTTPS_PROXY=http://127.0.0.1:7890
NO_PROXY=localhost,127.0.0.1
```

### 2. Update Application Code (if applicable)

If you're passing `proxy` parameter directly to agent constructors:

**Old code:**
```python
from src.agents.summarizer.agent import SummarizerAgent

agent = SummarizerAgent(
    model_name='gpt-3.5-turbo',
    api_key='your-key',
    proxy='http://127.0.0.1:7890'  # Remove this
)
```

**New code:**
```python
import os
from src.agents.summarizer.agent import SummarizerAgent

# Set environment variables programmatically (if needed)
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# Create agent without proxy parameter
agent = SummarizerAgent(
    model_name='gpt-3.5-turbo',
    api_key='your-key'
)
```

### 3. Docker/Container Environments

When running in Docker or container environments:

**Docker Compose:**
```yaml
services:
  ai-search-agents:
    environment:
      - HTTP_PROXY=http://proxy.example.com:8080
      - HTTPS_PROXY=http://proxy.example.com:8080
      - NO_PROXY=localhost,127.0.0.1
```

**Docker Run:**
```bash
docker run -e HTTP_PROXY=http://proxy.example.com:8080 \
           -e HTTPS_PROXY=http://proxy.example.com:8080 \
           -e NO_PROXY=localhost,127.0.0.1 \
           your-image:tag
```

## Why This Change?

1. **Compatibility**: The `proxies` parameter in `httpx.Client` caused compatibility issues with certain versions of OpenAI SDK and langchain-openai
2. **Standards**: Using `HTTP_PROXY` and `HTTPS_PROXY` follows standard Unix/Linux conventions
3. **Reliability**: The `trust_env=True` parameter in httpx ensures proxy configuration works consistently across all httpx versions
4. **Flexibility**: Environment variables can be set at system level, container level, or application level

## Backward Compatibility

- The `proxy` parameter is still accepted but **deprecated** and will be **ignored**
- A warning message will be logged if you pass the `proxy` parameter
- The `OPENAI_PROXY` setting is still accepted but **deprecated** and will be **ignored**
- No breaking changes - existing code will continue to work, just without proxy support unless you migrate to environment variables

## Testing Your Configuration

Use the provided test script to verify your proxy configuration works:

```bash
# Set your proxy
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port

# Run the test
python test_httpx_client_fix.py
```

All tests should pass with your proxy configuration.

## Troubleshooting

### Proxy Not Working

1. Verify environment variables are set:
   ```bash
   echo $HTTP_PROXY
   echo $HTTPS_PROXY
   ```

2. Check proxy format (should include protocol):
   ```bash
   # Correct
   HTTP_PROXY=http://proxy.example.com:8080
   
   # Incorrect (missing http://)
   HTTP_PROXY=proxy.example.com:8080
   ```

3. Verify proxy server is accessible:
   ```bash
   curl -x $HTTP_PROXY https://api.openai.com/v1/models
   ```

### Still Getting TypeError

If you still see `TypeError: Client.__init__() got an unexpected keyword argument 'proxies'`, verify:

1. You're using the latest version of the code
2. You've restarted your application after updating
3. You're not passing the `proxy` parameter in your code

## Support

If you continue to experience issues after following this guide, please:

1. Check that you're using compatible versions:
   - `httpx==0.25.2` or compatible
   - `openai>=1.40.0,<2.0.0` for langchain-openai 0.1.23
   - `langchain-openai==0.1.23` or compatible

2. Run the test suite:
   ```bash
   python test_httpx_client_fix.py
   ```

3. Check the logs for warning messages about deprecated proxy parameters
