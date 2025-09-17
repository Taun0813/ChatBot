"""
Logging utilities for the AI Agent system
"""

import logging
import sys
from typing import Optional, Dict, Any
from datetime import datetime
import json
import traceback

class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, ensure_ascii=False)

class AILogger:
    """Custom logger for AI Agent system"""
    
    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup console and file handlers"""
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log(logging.CRITICAL, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method"""
        extra_fields = kwargs.pop('extra_fields', {})
        if kwargs:
            extra_fields.update(kwargs)
        
        if extra_fields:
            # Create a new record with extra fields
            record = self.logger.makeRecord(
                self.logger.name, level, "", 0, message, (), None
            )
            record.extra_fields = extra_fields
            self.logger.handle(record)
        else:
            self.logger.log(level, message)

def setup_logger(name: str, level: str = "INFO") -> AILogger:
    """Setup and return a logger instance"""
    return AILogger(name, level)

def get_logger(name: str) -> AILogger:
    """Get existing logger or create new one"""
    return AILogger(name)

# Performance logging decorator
def log_performance(logger: AILogger):
    """Decorator to log function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            logger.info(f"Starting {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"Completed {func.__name__} in {duration:.2f}s")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"Failed {func.__name__} after {duration:.2f}s: {str(e)}")
                raise
        
        return wrapper
    return decorator

# Async performance logging decorator
def log_async_performance(logger: AILogger):
    """Decorator to log async function performance"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = datetime.now()
            logger.info(f"Starting async {func.__name__}")
            
            try:
                result = await func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"Completed async {func.__name__} in {duration:.2f}s")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"Failed async {func.__name__} after {duration:.2f}s: {str(e)}")
                raise
        
        return wrapper
    return decorator

# Model-specific logging
class ModelLogger:
    """Logger specifically for model operations"""
    
    def __init__(self, model_name: str):
        self.logger = AILogger(f"model.{model_name}")
        self.model_name = model_name
    
    def log_generation(self, prompt: str, response: str, tokens_used: int, processing_time: float):
        """Log model generation"""
        self.logger.info(
            f"Generated response for {self.model_name}",
            extra_fields={
                "prompt_length": len(prompt),
                "response_length": len(response),
                "tokens_used": tokens_used,
                "processing_time": processing_time,
                "tokens_per_second": tokens_used / processing_time if processing_time > 0 else 0
            }
        )
    
    def log_embedding(self, text: str, embedding_dim: int, processing_time: float):
        """Log embedding generation"""
        self.logger.info(
            f"Generated embedding for {self.model_name}",
            extra_fields={
                "text_length": len(text),
                "embedding_dimension": embedding_dim,
                "processing_time": processing_time
            }
        )
    
    def log_search(self, query: str, results_count: int, processing_time: float):
        """Log vector search"""
        self.logger.info(
            f"Vector search completed for {self.model_name}",
            extra_fields={
                "query_length": len(query),
                "results_count": results_count,
                "processing_time": processing_time
            }
        )

# API request logging
class APILogger:
    """Logger for API requests and responses"""
    
    def __init__(self):
        self.logger = AILogger("api")
    
    def log_request(self, method: str, url: str, headers: Dict[str, str] = None, body: str = None):
        """Log API request"""
        self.logger.info(
            f"API Request: {method} {url}",
            extra_fields={
                "method": method,
                "url": url,
                "headers": headers,
                "body_length": len(body) if body else 0
            }
        )
    
    def log_response(self, status_code: int, response_time: float, response_size: int = None):
        """Log API response"""
        self.logger.info(
            f"API Response: {status_code}",
            extra_fields={
                "status_code": status_code,
                "response_time": response_time,
                "response_size": response_size
            }
        )
    
    def log_error(self, error: Exception, context: str = None):
        """Log API error"""
        self.logger.error(
            f"API Error: {str(error)}",
            extra_fields={
                "error_type": type(error).__name__,
                "context": context,
                "traceback": traceback.format_exc()
            }
        )

# Chat session logging
class ChatLogger:
    """Logger for chat sessions"""
    
    def __init__(self):
        self.logger = AILogger("chat")
    
    def log_message(self, user_id: str, session_id: str, message: str, intent: str = None):
        """Log user message"""
        self.logger.info(
            f"User message received",
            extra_fields={
                "user_id": user_id,
                "session_id": session_id,
                "message_length": len(message),
                "intent": intent
            }
        )
    
    def log_response(self, user_id: str, session_id: str, response: str, model_name: str, processing_time: float):
        """Log AI response"""
        self.logger.info(
            f"AI response generated",
            extra_fields={
                "user_id": user_id,
                "session_id": session_id,
                "response_length": len(response),
                "model_name": model_name,
                "processing_time": processing_time
            }
        )
    
    def log_session_start(self, user_id: str, session_id: str):
        """Log session start"""
        self.logger.info(
            f"Chat session started",
            extra_fields={
                "user_id": user_id,
                "session_id": session_id
            }
        )
    
    def log_session_end(self, user_id: str, session_id: str, duration: float, message_count: int):
        """Log session end"""
        self.logger.info(
            f"Chat session ended",
            extra_fields={
                "user_id": user_id,
                "session_id": session_id,
                "duration": duration,
                "message_count": message_count
            }
        )