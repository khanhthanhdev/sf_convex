"""
Enhanced logging configuration for the configuration system.

This module provides detailed logging configuration with structured logging,
performance monitoring, and audit trails for configuration operations.
"""

import logging
import logging.handlers
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import threading


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging with JSON output."""
    
    def __init__(self, include_extra: bool = True):
        """Initialize the structured formatter.
        
        Args:
            include_extra: Whether to include extra fields in log records
        """
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log message as JSON string
        """
        # Base log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
            'process': record.process
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info) if record.exc_info else None
            }
        
        # Add extra fields if enabled
        if self.include_extra:
            # Get all extra attributes (those not in standard LogRecord)
            standard_attrs = {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                'module', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                'thread', 'threadName', 'processName', 'process', 'getMessage',
                'exc_info', 'exc_text', 'stack_info'
            }
            
            extra_attrs = set(record.__dict__.keys()) - standard_attrs
            if extra_attrs:
                log_entry['extra'] = {
                    attr: getattr(record, attr) for attr in extra_attrs
                }
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)


class ConfigurationLogger:
    """Enhanced logger for configuration operations with multiple output formats."""
    
    def __init__(self, log_dir: Optional[str] = None, enable_structured_logging: bool = True):
        """Initialize the configuration logger.
        
        Args:
            log_dir: Directory for log files. Defaults to 'logs'
            enable_structured_logging: Whether to enable structured JSON logging
        """
        self.log_dir = Path(log_dir or 'logs')
        self.log_dir.mkdir(exist_ok=True)
        self.enable_structured_logging = enable_structured_logging
        
        # Setup loggers
        self._setup_main_logger()
        self._setup_audit_logger()
        self._setup_performance_logger()
        self._setup_security_logger()
        self._setup_debug_logger()
        
        # Thread-local storage for context
        self._local = threading.local()
        
        self.main_logger.info("ConfigurationLogger initialized")
    
    def _setup_main_logger(self):
        """Setup main configuration logger."""
        self.main_logger = logging.getLogger('config.main')
        self.main_logger.setLevel(logging.DEBUG)
        
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        if self.enable_structured_logging:
            console_formatter = StructuredFormatter(include_extra=False)
        else:
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        console_handler.setFormatter(console_formatter)
        self.main_logger.addHandler(console_handler)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'config_main.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        
        if self.enable_structured_logging:
            file_formatter = StructuredFormatter(include_extra=True)
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
        file_handler.setFormatter(file_formatter)
        self.main_logger.addHandler(file_handler)
    
    def _setup_audit_logger(self):
        """Setup audit logger for security and compliance."""
        self.audit_logger = logging.getLogger('config.audit')
        self.audit_logger.setLevel(logging.INFO)
        
        # Audit log should always be structured for compliance
        audit_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'config_audit.log',
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10
        )
        audit_handler.setLevel(logging.INFO)
        audit_formatter = StructuredFormatter(include_extra=True)
        audit_handler.setFormatter(audit_formatter)
        self.audit_logger.addHandler(audit_handler)
        
        # Prevent audit logs from propagating to other loggers
        self.audit_logger.propagate = False
    
    def _setup_performance_logger(self):
        """Setup performance logger for monitoring slow operations."""
        self.perf_logger = logging.getLogger('config.performance')
        self.perf_logger.setLevel(logging.INFO)
        
        perf_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'config_performance.log',
            maxBytes=20*1024*1024,  # 20MB
            backupCount=5
        )
        perf_handler.setLevel(logging.INFO)
        
        if self.enable_structured_logging:
            perf_formatter = StructuredFormatter(include_extra=True)
        else:
            perf_formatter = logging.Formatter(
                '%(asctime)s - PERFORMANCE - %(message)s'
            )
        perf_handler.setFormatter(perf_formatter)
        self.perf_logger.addHandler(perf_handler)
        self.perf_logger.propagate = False
    
    def _setup_security_logger(self):
        """Setup security logger for security-related events."""
        self.security_logger = logging.getLogger('config.security')
        self.security_logger.setLevel(logging.WARNING)
        
        security_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'config_security.log',
            maxBytes=20*1024*1024,  # 20MB
            backupCount=10
        )
        security_handler.setLevel(logging.WARNING)
        security_formatter = StructuredFormatter(include_extra=True)
        security_handler.setFormatter(security_formatter)
        self.security_logger.addHandler(security_handler)
        self.security_logger.propagate = False
    
    def _setup_debug_logger(self):
        """Setup debug logger for detailed troubleshooting."""
        self.debug_logger = logging.getLogger('config.debug')
        self.debug_logger.setLevel(logging.DEBUG)
        
        # Only log to file for debug to avoid console spam
        debug_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'config_debug.log',
            maxBytes=50*1024*1024,  # 50MB
            backupCount=3
        )
        debug_handler.setLevel(logging.DEBUG)
        
        if self.enable_structured_logging:
            debug_formatter = StructuredFormatter(include_extra=True)
        else:
            debug_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s\n'
                'Context: %(context)s\n'
                '---'
            )
        debug_handler.setFormatter(debug_formatter)
        self.debug_logger.addHandler(debug_handler)
        self.debug_logger.propagate = False
    
    def set_context(self, **context):
        """Set logging context for current thread.
        
        Args:
            **context: Context key-value pairs
        """
        if not hasattr(self._local, 'context'):
            self._local.context = {}
        self._local.context.update(context)
    
    def clear_context(self):
        """Clear logging context for current thread."""
        if hasattr(self._local, 'context'):
            self._local.context.clear()
    
    def get_context(self) -> Dict[str, Any]:
        """Get current logging context.
        
        Returns:
            Current context dictionary
        """
        if hasattr(self._local, 'context'):
            return self._local.context.copy()
        return {}
    
    def _add_context_to_extra(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add current context to extra fields.
        
        Args:
            extra: Existing extra fields
            
        Returns:
            Extra fields with context added
        """
        result = extra or {}
        context = self.get_context()
        if context:
            result['context'] = context
        return result
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message with context.
        
        Args:
            message: Log message
            extra: Extra fields to include
        """
        self.main_logger.info(message, extra=self._add_context_to_extra(extra))
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message with context.
        
        Args:
            message: Log message
            extra: Extra fields to include
        """
        self.main_logger.warning(message, extra=self._add_context_to_extra(extra))
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = False):
        """Log error message with context.
        
        Args:
            message: Log message
            extra: Extra fields to include
            exc_info: Whether to include exception information
        """
        self.main_logger.error(message, extra=self._add_context_to_extra(extra), exc_info=exc_info)
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = False):
        """Log critical message with context.
        
        Args:
            message: Log message
            extra: Extra fields to include
            exc_info: Whether to include exception information
        """
        self.main_logger.critical(message, extra=self._add_context_to_extra(extra), exc_info=exc_info)
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message with context.
        
        Args:
            message: Log message
            extra: Extra fields to include
        """
        self.debug_logger.debug(message, extra=self._add_context_to_extra(extra))
    
    def audit(self, event_type: str, details: Dict[str, Any], severity: str = 'info'):
        """Log audit event.
        
        Args:
            event_type: Type of audit event
            details: Event details
            severity: Event severity (info, warning, error, critical)
        """
        audit_entry = {
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            'user': os.getenv('USER', 'unknown'),
            'process_id': os.getpid(),
            'thread_id': threading.get_ident(),
            'severity': severity,
            'audit_details': details,  # Renamed to avoid conflict
            'context': self.get_context()
        }
        
        # Log with appropriate level
        if severity == 'critical':
            self.audit_logger.critical('AUDIT', extra=audit_entry)
        elif severity == 'error':
            self.audit_logger.error('AUDIT', extra=audit_entry)
        elif severity == 'warning':
            self.audit_logger.warning('AUDIT', extra=audit_entry)
        else:
            self.audit_logger.info('AUDIT', extra=audit_entry)
    
    def performance(self, operation: str, duration: float, details: Optional[Dict[str, Any]] = None):
        """Log performance metrics.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            details: Additional performance details
        """
        perf_entry = {
            'operation': operation,
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat(),
            'perf_details': details or {},  # Renamed to avoid conflict
            'context': self.get_context()
        }
        
        # Determine log level based on duration
        if duration > 10.0:  # Very slow
            level = 'critical'
            self.perf_logger.critical(f"VERY SLOW OPERATION: {operation} took {duration:.2f}s", extra=perf_entry)
        elif duration > 5.0:  # Slow
            level = 'warning'
            self.perf_logger.warning(f"SLOW OPERATION: {operation} took {duration:.2f}s", extra=perf_entry)
        else:  # Normal
            level = 'info'
            self.perf_logger.info(f"OPERATION: {operation} took {duration:.2f}s", extra=perf_entry)
    
    def security(self, event_type: str, message: str, details: Optional[Dict[str, Any]] = None, severity: str = 'warning'):
        """Log security event.
        
        Args:
            event_type: Type of security event
            message: Security message
            details: Additional security details
            severity: Event severity
        """
        security_entry = {
            'event_type': event_type,
            'security_message': message,  # Renamed to avoid conflict
            'timestamp': datetime.now().isoformat(),
            'user': os.getenv('USER', 'unknown'),
            'process_id': os.getpid(),
            'severity': severity,
            'details': details or {},
            'context': self.get_context()
        }
        
        # Log with appropriate level
        if severity == 'critical':
            self.security_logger.critical(f"SECURITY: {message}", extra=security_entry)
        elif severity == 'error':
            self.security_logger.error(f"SECURITY: {message}", extra=security_entry)
        else:
            self.security_logger.warning(f"SECURITY: {message}", extra=security_entry)
    
    def log_configuration_change(self, component: str, old_value: Any, new_value: Any, user: Optional[str] = None):
        """Log configuration changes for audit purposes.
        
        Args:
            component: Component being changed
            old_value: Previous value
            new_value: New value
            user: User making the change
        """
        self.audit(
            event_type='configuration_change',
            details={
                'component': component,
                'old_value': str(old_value) if old_value is not None else None,
                'new_value': str(new_value) if new_value is not None else None,
                'user': user or os.getenv('USER', 'unknown'),
                'change_type': 'update' if old_value is not None else 'create'
            }
        )
    
    def log_api_key_usage(self, provider: str, operation: str, success: bool, details: Optional[Dict[str, Any]] = None):
        """Log API key usage for security monitoring.
        
        Args:
            provider: Provider name
            operation: Operation performed
            success: Whether operation was successful
            details: Additional details
        """
        self.security(
            event_type='api_key_usage',
            message=f"API key used for {provider} - {operation}",
            details={
                'provider': provider,
                'operation': operation,
                'success': success,
                'details': details or {}
            },
            severity='info' if success else 'warning'
        )
    
    def log_fallback_usage(self, component: str, reason: str):
        """Log fallback configuration usage.
        
        Args:
            component: Component using fallback
            reason: Reason for fallback
        """
        self.audit(
            event_type='fallback_config_used',
            details={
                'component': component,
                'reason': reason,
                'fallback_type': 'configuration'
            },
            severity='warning'
        )
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics.
        
        Returns:
            Dictionary with logging statistics
        """
        stats = {
            'log_directory': str(self.log_dir),
            'structured_logging_enabled': self.enable_structured_logging,
            'log_files': [],
            'total_log_size_mb': 0
        }
        
        # Get information about log files
        for log_file in self.log_dir.glob('*.log*'):
            try:
                file_size = log_file.stat().st_size
                stats['log_files'].append({
                    'name': log_file.name,
                    'size_mb': file_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                })
                stats['total_log_size_mb'] += file_size / (1024 * 1024)
            except Exception:
                # Skip files we can't access
                continue
        
        return stats


# Global logger instance
_config_logger: Optional[ConfigurationLogger] = None
_logger_lock = threading.Lock()


def get_config_logger() -> ConfigurationLogger:
    """Get the global configuration logger instance."""
    global _config_logger
    
    if _config_logger is None:
        with _logger_lock:
            if _config_logger is None:
                _config_logger = ConfigurationLogger()
    
    return _config_logger


def setup_logging(log_dir: Optional[str] = None, enable_structured_logging: bool = True) -> ConfigurationLogger:
    """Setup configuration logging.
    
    Args:
        log_dir: Directory for log files
        enable_structured_logging: Whether to enable structured JSON logging
        
    Returns:
        ConfigurationLogger instance
    """
    global _config_logger
    
    with _logger_lock:
        _config_logger = ConfigurationLogger(
            log_dir=log_dir,
            enable_structured_logging=enable_structured_logging
        )
    
    return _config_logger