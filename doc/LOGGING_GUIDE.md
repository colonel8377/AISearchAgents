# Logging System Guide

## Overview

This document describes the improved logging system for the AI Search Agents Platform. The logging functionality has been centralized and enhanced to make error tracking and debugging much easier.

## Features

### 1. Centralized Logging Utility
- Location: `src/utils/logger.py`
- Provides consistent logging across the entire application
- Supports both console and file output
- Includes detailed context: timestamp, module name, log level, file name, and line number

### 2. Configurable Logging Levels
- DEBUG: Detailed information for debugging
- INFO: General informational messages
- WARNING: Warning messages for potentially problematic situations
- ERROR: Error messages for failures
- CRITICAL: Critical issues that prevent normal operation

### 3. File and Console Handlers
- Logs can be written to both console and file simultaneously
- Log files are automatically created with timestamps
- Log directory is excluded from version control

## Configuration

### Environment Variables

Add these settings to your `.env` file:

```bash
# Logging Settings
LOG_LEVEL=INFO  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE=logs/ai_search_agents.log
ENABLE_DEBUG=false
```

### Configuration Options

- `LOG_LEVEL`: Controls which log messages are displayed (default: INFO)
- `LOG_FILE`: Path to the log file (default: logs/ai_search_agents.log)
- `ENABLE_DEBUG`: When true, sets log level to DEBUG and enables verbose logging

## Usage Examples

### Basic Usage in a Module

```python
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Log different levels
logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred", exc_info=True)  # Include stack trace
```

### Configuring Application-Wide Logging

In your main application file:

```python
from src.utils.logger import configure_app_logging
from src.config.settings import settings

# Configure logging at application startup
configure_app_logging(
    log_level=settings.log_level,
    log_file=settings.log_file,
    enable_debug=settings.enable_debug
)
```

### Custom Logger Configuration

For more control:

```python
from src.utils.logger import setup_logger
import logging

# Create a custom logger
logger = setup_logger(
    name='my_module',
    level=logging.DEBUG,
    log_file='/path/to/custom.log',
    console_output=True
)
```

## Log Format

The default log format includes:

```
%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s
```

Example output:
```
2025-12-15 16:14:38 - src.agents.manager - INFO - manager.py:30 - AgentManager initialized
```

This format provides:
- **Timestamp**: When the log was created
- **Logger name**: Which module/component logged the message
- **Log level**: Severity of the message
- **File and line**: Exact location in the code
- **Message**: The log message content

## Enhanced Logging Throughout the Application

### API Endpoints (`src/api/main.py`)
- Logs all agent creation requests with details
- Logs errors with full context and stack traces
- Tracks API endpoint calls and results

### Agent Manager (`src/agents/manager.py`)
- Logs agent registration and deletion
- Tracks agent lifecycle events
- Helps debug agent management issues

### Agents
All agent classes now include comprehensive logging:

- **NudgeCollapseAgent**: Logs turn generation, LLM calls, and vector memory operations
- **SummarizerAgent**: Logs summarization requests, truncation, and results
- **BotCreatorAgent**: Logs bot creation and configuration

### Debate System (`src/debate/service.py`)
- Logs debate session creation
- Tracks vote rounds and stability calculations
- Logs KS statistic calculations for debugging

### Vector Store Factory (`src/memory/factory.py`)
- Logs vector store creation for different backends
- Helps debug connection issues with Redis, PostgreSQL, or Chroma

## Debugging Tips

### Enable Debug Mode

Set in your `.env`:
```bash
ENABLE_DEBUG=true
LOG_LEVEL=DEBUG
```

This will:
- Show all DEBUG level messages
- Provide maximum detail for troubleshooting
- Write detailed logs to file

### View Log Files

```bash
# View the latest logs
tail -f logs/ai_search_agents.log

# Search for errors
grep ERROR logs/ai_search_agents.log

# View logs for a specific module
grep "src.agents.manager" logs/ai_search_agents.log
```

### Common Issues

1. **Missing log file**: Check that the logs directory exists and is writable
2. **Not seeing debug messages**: Verify LOG_LEVEL is set to DEBUG
3. **Too many logs**: Increase LOG_LEVEL to WARNING or ERROR

## Best Practices

1. **Use appropriate log levels**:
   - DEBUG: Detailed diagnostic information
   - INFO: Confirmation that things are working as expected
   - WARNING: Something unexpected, but the application continues
   - ERROR: A serious problem that prevented a function from working
   - CRITICAL: A critical error that may prevent the application from running

2. **Include context in log messages**:
   ```python
   logger.info(f"Creating agent: type={agent_type}, id={agent_id}")
   ```

3. **Log exceptions with stack traces**:
   ```python
   try:
       # some code
   except Exception as e:
       logger.error(f"Operation failed: {e}", exc_info=True)
   ```

4. **Don't log sensitive information**:
   - Avoid logging API keys, passwords, or personal data
   - Redact sensitive information if necessary

## Migration from Print Statements

All `print()` statements have been replaced with appropriate logger calls:

- `print()` → `logger.info()`
- `print(f"Warning: ...")` → `logger.warning()`
- `print(f"Error: ...")` → `logger.error()`

## Benefits

✅ **Easy error location**: File name and line number in every log
✅ **Centralized configuration**: Change logging behavior application-wide
✅ **Flexible output**: Console and/or file logging
✅ **Production-ready**: Properly structured for deployment
✅ **Debugging friendly**: Debug mode for detailed troubleshooting
✅ **Consistent format**: Same log format across all modules

## Support

For issues or questions about the logging system:
1. Check the log file for error messages
2. Verify your `.env` configuration
3. Consult this guide for usage examples
4. Review the code in `src/utils/logger.py` for implementation details
