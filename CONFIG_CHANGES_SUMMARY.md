# Configuration Changes Summary

## Changes Implemented

This PR implements two configuration improvements:

### 1. Remove Proxy Parameter from API
- **What**: Removed `proxy` field from `ExtractAndCleanRequest` API model
- **Why**: Simplifies API by centralizing proxy configuration via settings
- **How**: Proxy now read from `settings.openai_proxy` or environment variables (`HTTP_PROXY`, `HTTPS_PROXY`)

### 2. Parse model_name from .env
- **What**: All agents now default to `settings.openai_model` instead of hardcoded `"gpt-3.5-turbo"`
- **Why**: Allows model configuration via `.env` file without code changes
- **How**: Changed `model_name: str = "gpt-3.5-turbo"` to `model_name: Optional[str] = None` with `model_name = model_name or settings.openai_model`

## Files Changed (9 files, 52 insertions, 48 deletions)

### API Changes:
- `src/api/main.py` - Removed proxy parameter, use settings instead
- `example_extract_and_clean.py` - Updated examples
- `test_web_opinion_api.py` - Updated tests
- `WEB_OPINION_API_DOCS.md` - Updated documentation
- `COMBINED_EXTRACT_CLEAN_API_SUMMARY.md` - Updated documentation

### Agent Changes:
- `src/agents/nudge_collapse/agent.py` - Use settings for model_name
- `src/agents/summarizer/agent.py` - Use settings for model_name
- `src/agents/bot_creator/agent.py` - Use settings for model_name
- `src/agents/web_opinion_extractor/agent.py` - Use settings for model_name (both classes)

## Migration Required

### For API Users:
Remove `proxy` parameter from API requests:
```python
# OLD
requests.post("/api/v1/web-opinion/extractandclean", 
    json={"url": "...", "proxy": "http://..."})  # ❌

# NEW - Configure in .env or environment variables
requests.post("/api/v1/web-opinion/extractandclean", 
    json={"url": "..."})  # ✓
```

### For Agent Users:
No changes required! But you can now configure model in `.env`:
```bash
# .env file
OPENAI_MODEL=gpt-4  # Will be used by all agents by default
```

## Backward Compatibility

✓ Agents can still override model_name if needed
✓ Proxy configuration via environment variables already supported
✗ API no longer accepts per-request proxy (use settings instead)

## Verification

All changes verified:
- ✓ Settings load correctly
- ✓ All agents import successfully
- ✓ Code review passed with no issues
- ✓ Backward compatibility maintained (except proxy in API)
