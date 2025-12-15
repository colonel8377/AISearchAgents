"""
Centralized logging configuration for AI Search Agents Platform.

This module provides a unified logging setup with both console and file handlers,
making it easier to track errors and debug issues across the application.
"""

import logging
import os
import sys
from typing import Optional
from pathlib import Path


# Default log format
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Global logger cache
_loggers = {}


def setup_logger(
    name: str = "ai_search_agents",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.
    
    Args:
        name: Logger name (usually module name or application name)
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Optional path to log file. If provided, logs will be written to this file
        log_format: Optional custom format string. Uses DEFAULT_FORMAT if not provided
        console_output: Whether to output logs to console (default: True)
        
    Returns:
        Configured logger instance
    """
    # Return cached logger if already configured
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Use default format if not provided
    format_str = log_format or DEFAULT_FORMAT
    formatter = logging.Formatter(format_str, datefmt=DEFAULT_DATE_FORMAT)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler (if log_file is specified)
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    # Cache the logger
    _loggers[name] = logger
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger instance.
    
    This function will return an existing logger if already configured via setup_logger(),
    or create a new logger with default settings if not.
    
    Args:
        name: Logger name (typically __name__ in the calling module)
        
    Returns:
        Logger instance
    """
    # If logger already exists in cache, return it
    if name in _loggers:
        return _loggers[name]
    
    # If root application logger exists, create child logger
    if "ai_search_agents" in _loggers:
        logger = logging.getLogger(f"ai_search_agents.{name}")
        return logger
    
    # Otherwise, create a new logger with default settings
    return setup_logger(name)


def configure_app_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_debug: bool = False
) -> logging.Logger:
    """
    Configure application-wide logging settings.
    
    This should be called once at application startup to set up the root logger.
    
    Args:
        log_level: Logging level as string ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        log_file: Optional path to log file
        enable_debug: Enable debug mode with more verbose logging
        
    Returns:
        Root application logger
    """
    # Convert string level to logging constant
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    level = level_map.get(log_level.upper(), logging.INFO)
    
    # Use DEBUG level if debug mode is enabled
    if enable_debug:
        level = logging.DEBUG
    
    # Determine default log file location
    if log_file is None and enable_debug:
        log_file = "logs/ai_search_agents.log"
    
    # Set up the root application logger
    logger = setup_logger(
        name="ai_search_agents",
        level=level,
        log_file=log_file,
        console_output=True
    )
    
    logger.info(f"Logging configured: level={log_level}, file={log_file}, debug={enable_debug}")
    
    return logger
