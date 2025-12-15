# Qwen Model Troubleshooting Guide

This guide helps you configure and troubleshoot Qwen (Alibaba Cloud DashScope) models with the AI Search Agents Platform.

## Quick Setup for Qwen

### 1. Get Your API Key
1. Visit [DashScope Console](https://dashscope.console.aliyun.com/)
2. Create or log in to your Alibaba Cloud account
3. Navigate to API-KEY management
4. Create a new API key or copy an existing one

### 2. Configure Environment Variables

Edit your `.env` file with these settings:

```bash
# Qwen API Configuration
OPENAI_API_KEY=sk-your-dashscope-api-key
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-turbo

# Recommended settings for Qwen
OPENAI_MAX_RETRIES=5
OPENAI_TIMEOUT=120.0
```

### 3. Available Qwen Models

| Model | Description | Context Window | Best For |
|-------|-------------|----------------|----------|
| `qwen-turbo` | Fast, economical | 8K tokens | Quick responses, high volume |
| `qwen-plus` | Balanced | 32K tokens | General purpose, good quality |
| `qwen-max` | Highest quality | 8K tokens | Complex tasks, best accuracy |
| `qwen-max-longcontext` | Extended context | 30K tokens | Long documents, large context |

## Common Issues and Solutions

### Error 502: Bad Gateway

**Symptom:**
```
openai.InternalServerError: Error code: 502
```

**Possible Causes and Solutions:**

#### 1. Incorrect API Base URL
- **Check:** Ensure your API base URL is correct
- **Solution:** 
  ```bash
  OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
  ```
  Note: Must include `/compatible-mode/v1` at the end

#### 2. Invalid API Key
- **Check:** Verify your API key is active and valid
- **Solution:**
  1. Log in to [DashScope Console](https://dashscope.console.aliyun.com/)
  2. Check API key status
  3. Generate a new key if needed
  4. Update `.env` with new key

#### 3. Quota Exceeded
- **Check:** Verify you have sufficient API quota
- **Solution:**
  1. Check your account balance in DashScope Console
  2. Top up your account if needed
  3. Check daily/monthly limits

#### 4. Model Not Available
- **Check:** Ensure the model name is correct and available in your region
- **Solution:**
  ```bash
  # Try a different model
  OPENAI_MODEL=qwen-turbo  # Most widely available
  ```

#### 5. Network/Timeout Issues
- **Check:** Request might be timing out
- **Solution:** Increase timeout settings
  ```bash
  OPENAI_TIMEOUT=180.0  # 3 minutes
  OPENAI_MAX_RETRIES=5
  ```

### Error 401: Unauthorized

**Symptom:**
```
Authentication failed
```

**Solution:**
1. Verify API key format (should start with `sk-`)
2. Check for extra spaces or newlines in `.env` file
3. Ensure API key is active in DashScope Console
4. Regenerate API key if needed

### Slow Response Times

**Symptom:**
Requests take very long or timeout

**Solutions:**

1. **Increase Timeout:**
   ```bash
   OPENAI_TIMEOUT=180.0
   ```

2. **Use Faster Model:**
   ```bash
   OPENAI_MODEL=qwen-turbo
   ```

3. **Reduce Temperature:**
   ```bash
   AGENT_TEMPERATURE=0.3  # Lower = faster, more deterministic
   ```

4. **Check Network:**
   - Test connectivity to DashScope endpoints
   - Consider using a proxy if in restricted network:
     ```bash
     OPENAI_PROXY=http://your-proxy:8080
     ```

### Connection Refused

**Symptom:**
```
Connection refused or cannot connect to API
```

**Solutions:**

1. **Check Firewall:**
   - Ensure outbound HTTPS (443) is allowed
   - Whitelist DashScope domains if needed

2. **Verify DNS:**
   ```bash
   # Test DNS resolution
   ping dashscope.aliyuncs.com
   ```

3. **Try Proxy:**
   ```bash
   OPENAI_PROXY=http://proxy.example.com:8080
   ```

## Advanced Configuration

### For Production Deployments

```bash
# Production-ready Qwen configuration
OPENAI_API_KEY=sk-your-production-key
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-plus

# Robust retry and timeout settings
OPENAI_MAX_RETRIES=5
OPENAI_TIMEOUT=120.0

# Agent settings
AGENT_TEMPERATURE=0.5
AGENT_MAX_TURNS=4

# Enable logging for debugging
LOG_LEVEL=INFO
ENABLE_DEBUG=false
```

### Using Multiple Models

You can create different agent instances with different models:

```python
import requests

BASE_URL = "http://localhost:8000"

# Create agent with qwen-turbo (fast)
resp1 = requests.post(f"{BASE_URL}/api/v1/agents", json={
    "agent_type": "bot_creator"
})
agent_fast = resp1.json()["agent_id"]

# Note: To use different models per agent, you would need to 
# modify the agent creation endpoint to accept model_name parameter
# Currently all agents use the model specified in .env
```

## Testing Your Configuration

### 1. Test API Connection

```python
import requests

# Test if API is reachable
url = "https://dashscope.aliyuncs.com/compatible-mode/v1/models"
headers = {
    "Authorization": f"Bearer {your_api_key}",
    "Content-Type": "application/json"
}

try:
    response = requests.get(url, headers=headers, timeout=10)
    print(f"Status: {response.status_code}")
    print(f"Models: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
```

### 2. Test Agent Creation

```bash
# Start the server
python src/main.py

# In another terminal, test agent creation
curl -X POST http://localhost:8000/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "bot_creator"}'
```

### 3. Test Bot Creation

```bash
# Create a bot (replace AGENT_ID with your agent's ID)
curl -X POST http://localhost:8000/api/v1/agents/AGENT_ID/bot-creator/create \
  -H "Content-Type: application/json" \
  -d '{
    "persona_prompt": "A friendly AI assistant",
    "bot_name": "TestBot"
  }'
```

## Getting Help

If you continue to experience issues:

1. **Check Logs:**
   ```bash
   tail -f logs/ai_search_agents.log
   ```

2. **Enable Debug Mode:**
   ```bash
   LOG_LEVEL=DEBUG
   ENABLE_DEBUG=true
   ```

3. **Verify Settings:**
   ```python
   # Check your configuration
   from src.config.settings import settings
   print(f"API Base: {settings.openai_api_base}")
   print(f"Model: {settings.openai_model}")
   print(f"Timeout: {settings.openai_timeout}")
   ```

4. **Report Issues:**
   - Include error messages from logs
   - Include your configuration (without API keys!)
   - Describe steps to reproduce

## Regional Considerations

### China Region
- Use DashScope China endpoint: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- Ensure your account is registered in the correct region
- Some models may have regional availability restrictions

### International Regions
- Check model availability in your region
- Some models may have different names or versions
- Contact Alibaba Cloud support for region-specific information

## Performance Optimization

### 1. Choose the Right Model
- Use `qwen-turbo` for simple tasks and high volume
- Use `qwen-plus` for balanced performance
- Use `qwen-max` only when highest quality is needed

### 2. Optimize Temperature
```bash
# For consistent, focused responses
AGENT_TEMPERATURE=0.3

# For creative, varied responses
AGENT_TEMPERATURE=0.8
```

### 3. Use Connection Pooling
The platform already handles connection pooling, but ensure:
- Don't create too many agent instances
- Reuse agents when possible
- Clean up unused agents with DELETE endpoint

### 4. Monitor Usage
- Track API calls in DashScope Console
- Monitor response times
- Set up alerts for quota limits

## Migration from OpenAI to Qwen

If you're migrating from OpenAI to Qwen:

1. **Update API Configuration:**
   ```bash
   # Old (OpenAI)
   OPENAI_API_KEY=sk-openai-key
   OPENAI_API_BASE=https://api.openai.com/v1
   OPENAI_MODEL=gpt-3.5-turbo
   
   # New (Qwen)
   OPENAI_API_KEY=sk-dashscope-key
   OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
   OPENAI_MODEL=qwen-plus
   ```

2. **Adjust Settings:**
   - Qwen may need longer timeouts
   - Increase max retries for reliability

3. **Test Thoroughly:**
   - Test all agent types
   - Verify response quality
   - Check for any compatibility issues

4. **Monitor Performance:**
   - Compare response times
   - Track API costs
   - Monitor error rates

## Security Best Practices

1. **Protect API Keys:**
   - Never commit `.env` file to version control
   - Use `.env.example` for templates
   - Rotate keys regularly

2. **Use Environment-Specific Keys:**
   - Different keys for development, staging, production
   - Limit permissions on API keys

3. **Monitor Access:**
   - Review API access logs regularly
   - Set up alerts for unusual activity
   - Track quota usage

4. **Network Security:**
   - Use HTTPS for all API calls
   - Consider VPN for sensitive deployments
   - Whitelist IP addresses if possible
