"""
Enhanced error handling and logging for the configuration system.

This module provides comprehensive error handling, detailed logging, fallback mechanisms,
and audit logging for configuration operations.
"""

import logging
import traceback
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from enum import Enum
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading

from .models import ValidationResult, SystemConfig


class ConfigErrorSeverity(Enum):
    """Severity levels for configuration errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConfigErrorCategory(Enum):
    """Categories of configuration errors."""
    LOADING = "loading"
    VALIDATION = "validation"
    CONNECTION = "connection"
    PERMISSION = "permission"
    RESOURCE = "resource"
    SECURITY = "security"
    COMPATIBILITY = "compatibility"
    MIGRATION = "migration"


@dataclass
class ConfigError:
    """Detailed configuration error information."""
    category: ConfigErrorCategory
    severity: ConfigErrorSeverity
    message: str
    component: str
    timestamp: datetime
    exception: Optional[Exception] = None
    context: Optional[Dict[str, Any]] = None
    suggested_fix: Optional[str] = None
    error_code: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging."""
        return {
            'category': self.category.value,
            'severity': self.severity.value,
            'message': self.message,
            'component': self.component,
            'timestamp': self.timestamp.isoformat(),
            'exception': str(self.exception) if self.exception else None,
            'exception_type': type(self.exception).__name__ if self.exception else None,
            'context': self.context or {},
            'suggested_fix': self.suggested_fix,
            'error_code': self.error_code,
            'traceback': traceback.format_exception(type(self.exception), self.exception, self.exception.__traceback__) if self.exception else None
        }


class ConfigurationErrorHandler:
    """Enhanced error handler for configuration operations."""
    
    def __init__(self, log_dir: Optional[str] = None):
        """Initialize the error handler.
        
        Args:
            log_dir: Directory for log files. Defaults to 'logs'
        """
        self.log_dir = Path(log_dir or 'logs')
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup loggers
        self._setup_loggers()
        
        # Error tracking
        self.error_history: List[ConfigError] = []
        self.error_counts: Dict[str, int] = {}
        self._lock = threading.RLock()
        
        # Fallback configurations
        self.fallback_configs: Dict[str, Any] = {}
        self.fallback_enabled = True
        
        # Audit logging
        self.audit_enabled = True
        self.audit_logger = self._setup_audit_logger()
        
        self.logger.info("ConfigurationErrorHandler initialized")
    
    def _setup_loggers(self):
        """Setup specialized loggers for different purposes."""
        # Main configuration logger
        self.logger = logging.getLogger('config.error_handler')
        
        # Error-specific logger with detailed formatting
        self.error_logger = logging.getLogger('config.errors')
        error_handler = logging.FileHandler(self.log_dir / 'config_errors.log')
        error_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
            'Context: %(context)s\n'
            'Suggested Fix: %(suggested_fix)s\n'
            '---'
        )
        error_handler.setFormatter(error_formatter)
        self.error_logger.addHandler(error_handler)
        self.error_logger.setLevel(logging.ERROR)
        
        # Performance logger for slow operations
        self.perf_logger = logging.getLogger('config.performance')
        perf_handler = logging.FileHandler(self.log_dir / 'config_performance.log')
        perf_formatter = logging.Formatter(
            '%(asctime)s - PERFORMANCE - %(message)s'
        )
        perf_handler.setFormatter(perf_formatter)
        self.perf_logger.addHandler(perf_handler)
        self.perf_logger.setLevel(logging.INFO)
    
    def _setup_audit_logger(self) -> logging.Logger:
        """Setup audit logger for security and debugging."""
        audit_logger = logging.getLogger('config.audit')
        audit_handler = logging.FileHandler(self.log_dir / 'config_audit.log')
        audit_formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(message)s'
        )
        audit_handler.setFormatter(audit_formatter)
        audit_logger.addHandler(audit_handler)
        audit_logger.setLevel(logging.INFO)
        return audit_logger
    
    def handle_error(
        self,
        category: ConfigErrorCategory,
        severity: ConfigErrorSeverity,
        message: str,
        component: str,
        exception: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
        suggested_fix: Optional[str] = None,
        error_code: Optional[str] = None
    ) -> ConfigError:
        """Handle a configuration error with comprehensive logging.
        
        Args:
            category: Error category
            severity: Error severity
            message: Error message
            component: Component that generated the error
            exception: Original exception if any
            context: Additional context information
            suggested_fix: Suggested fix for the error
            error_code: Unique error code
            
        Returns:
            ConfigError instance
        """
        error = ConfigError(
            category=category,
            severity=severity,
            message=message,
            component=component,
            timestamp=datetime.now(),
            exception=exception,
            context=context or {},
            suggested_fix=suggested_fix,
            error_code=error_code
        )
        
        with self._lock:
            # Track error
            self.error_history.append(error)
            error_key = f"{category.value}:{component}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
            
            # Log error with appropriate level
            self._log_error(error)
            
            # Audit log for security-related errors
            if category in [ConfigErrorCategory.SECURITY, ConfigErrorCategory.PERMISSION]:
                self._audit_log_error(error)
            
            # Check for error patterns that might indicate systemic issues
            self._check_error_patterns(error)
        
        return error
    
    def _log_error(self, error: ConfigError):
        """Log error with appropriate level and detail."""
        error_dict = error.to_dict()
        
        # Choose logging level based on severity
        if error.severity == ConfigErrorSeverity.CRITICAL:
            log_level = logging.CRITICAL
        elif error.severity == ConfigErrorSeverity.HIGH:
            log_level = logging.ERROR
        elif error.severity == ConfigErrorSeverity.MEDIUM:
            log_level = logging.WARNING
        else:
            log_level = logging.INFO
        
        # Log to main logger
        self.logger.log(
            log_level,
            f"[{error.category.value.upper()}] {error.component}: {error.message}",
            extra={
                'context': error.context,
                'suggested_fix': error.suggested_fix or 'No suggestion available'
            }
        )
        
        # Log detailed error information
        if error.severity in [ConfigErrorSeverity.HIGH, ConfigErrorSeverity.CRITICAL]:
            self.error_logger.error(
                f"DETAILED ERROR REPORT\n"
                f"Category: {error.category.value}\n"
                f"Severity: {error.severity.value}\n"
                f"Component: {error.component}\n"
                f"Message: {error.message}\n"
                f"Error Code: {error.error_code or 'N/A'}\n"
                f"Context: {json.dumps(error.context, indent=2) if error.context else 'None'}\n"
                f"Suggested Fix: {error.suggested_fix or 'No suggestion available'}\n"
                f"Exception: {error.exception}\n"
                f"Timestamp: {error.timestamp.isoformat()}",
                extra={
                    'context': error.context,
                    'suggested_fix': error.suggested_fix or 'No suggestion available'
                }
            )
    
    def _audit_log_error(self, error: ConfigError):
        """Log security-related errors to audit log."""
        if not self.audit_enabled:
            return
        
        audit_entry = {
            'event_type': 'configuration_error',
            'category': error.category.value,
            'severity': error.severity.value,
            'component': error.component,
            'message': error.message,
            'timestamp': error.timestamp.isoformat(),
            'context': error.context or {},
            'user': os.getenv('USER', 'unknown'),
            'process_id': os.getpid(),
            'error_code': error.error_code
        }
        
        self.audit_logger.info(json.dumps(audit_entry))
    
    def _check_error_patterns(self, error: ConfigError):
        """Check for error patterns that might indicate systemic issues."""
        error_key = f"{error.category.value}:{error.component}"
        count = self.error_counts.get(error_key, 0)
        
        # Alert on repeated errors
        if count > 5:
            self.logger.warning(
                f"Repeated error pattern detected: {error_key} occurred {count} times. "
                f"This might indicate a systemic issue."
            )
        
        # Alert on critical errors
        if error.severity == ConfigErrorSeverity.CRITICAL:
            self.logger.critical(
                f"CRITICAL CONFIGURATION ERROR: {error.message} in {error.component}. "
                f"System stability may be compromised."
            )
    
    @contextmanager
    def error_context(
        self,
        component: str,
        operation: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """Context manager for handling errors in configuration operations.
        
        Args:
            component: Component performing the operation
            operation: Description of the operation
            context: Additional context information
        """
        start_time = datetime.now()
        operation_context = context or {}
        operation_context.update({
            'operation': operation,
            'start_time': start_time.isoformat()
        })
        
        try:
            self.logger.debug(f"Starting {operation} in {component}")
            yield
            
            # Log successful completion
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.debug(f"Completed {operation} in {component} ({duration:.2f}s)")
            
            # Log performance warning for slow operations
            if duration > 5.0:  # 5 seconds threshold
                self.perf_logger.warning(
                    f"Slow operation: {operation} in {component} took {duration:.2f}s"
                )
                
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            operation_context.update({
                'duration_seconds': duration,
                'end_time': datetime.now().isoformat()
            })
            
            # Determine error severity based on exception type
            if isinstance(e, (PermissionError, OSError)):
                severity = ConfigErrorSeverity.HIGH
                category = ConfigErrorCategory.PERMISSION
            elif isinstance(e, (ConnectionError, TimeoutError)):
                severity = ConfigErrorSeverity.MEDIUM
                category = ConfigErrorCategory.CONNECTION
            elif isinstance(e, (ValueError, TypeError)):
                severity = ConfigErrorSeverity.MEDIUM
                category = ConfigErrorCategory.VALIDATION
            else:
                severity = ConfigErrorSeverity.HIGH
                category = ConfigErrorCategory.LOADING
            
            # Handle the error
            self.handle_error(
                category=category,
                severity=severity,
                message=f"Failed to {operation}: {str(e)}",
                component=component,
                exception=e,
                context=operation_context,
                suggested_fix=self._get_suggested_fix(e, operation)
            )
            
            # Re-raise the exception
            raise
    
    def _get_suggested_fix(self, exception: Exception, operation: str) -> str:
        """Get suggested fix based on exception type and operation."""
        if isinstance(exception, FileNotFoundError):
            return f"Ensure the required file exists and is accessible. Check file path and permissions."
        elif isinstance(exception, PermissionError):
            return f"Check file/directory permissions. Ensure the process has read/write access."
        elif isinstance(exception, ConnectionError):
            return f"Check network connectivity and service availability. Verify API endpoints and credentials."
        elif isinstance(exception, TimeoutError):
            return f"Increase timeout values or check service responsiveness. Consider retry mechanisms."
        elif isinstance(exception, ValueError):
            return f"Verify configuration values are in the correct format and within valid ranges."
        elif isinstance(exception, KeyError):
            return f"Ensure all required configuration keys are present. Check environment variables."
        else:
            return f"Review the error details and check the {operation} implementation."
    
    def register_fallback_config(self, component: str, config: Dict[str, Any]):
        """Register a fallback configuration for a component.
        
        Args:
            component: Component name
            config: Fallback configuration
        """
        self.fallback_configs[component] = config
        self.logger.info(f"Registered fallback configuration for {component}")
    
    def get_fallback_config(self, component: str) -> Optional[Dict[str, Any]]:
        """Get fallback configuration for a component.
        
        Args:
            component: Component name
            
        Returns:
            Fallback configuration or None
        """
        if not self.fallback_enabled:
            return None
        
        fallback = self.fallback_configs.get(component)
        if fallback:
            self.logger.warning(f"Using fallback configuration for {component}")
            self._audit_log_fallback_usage(component)
        
        return fallback
    
    def _audit_log_fallback_usage(self, component: str):
        """Log fallback configuration usage to audit log."""
        if not self.audit_enabled:
            return
        
        audit_entry = {
            'event_type': 'fallback_config_used',
            'component': component,
            'timestamp': datetime.now().isoformat(),
            'user': os.getenv('USER', 'unknown'),
            'process_id': os.getpid()
        }
        
        self.audit_logger.warning(json.dumps(audit_entry))
    
    def create_minimal_config(self) -> SystemConfig:
        """Create a minimal working configuration as last resort fallback.
        
        Returns:
            Minimal SystemConfig instance
        """
        self.logger.critical("Creating minimal configuration as last resort fallback")
        
        try:
            # Import here to avoid circular imports
            from .factory import ConfigurationFactory
            
            # Try to create basic configuration
            minimal_config = ConfigurationFactory.create_minimal_config()
            
            self._audit_log_minimal_config_creation()
            return minimal_config
            
        except Exception as e:
            self.handle_error(
                category=ConfigErrorCategory.LOADING,
                severity=ConfigErrorSeverity.CRITICAL,
                message=f"Failed to create minimal configuration: {str(e)}",
                component="error_handler",
                exception=e,
                suggested_fix="Check system dependencies and basic environment setup"
            )
            
            # Absolute last resort - create truly minimal config
            return self._create_absolute_minimal_config()
    
    def _create_absolute_minimal_config(self) -> SystemConfig:
        """Create absolute minimal configuration that should always work."""
        try:
            return SystemConfig(
                environment="development",
                debug=True,
                default_llm_provider="openai",
                llm_providers={},
                agent_configs={}
            )
        except Exception as e:
            # If even this fails, we have a serious problem
            self.logger.critical(f"Failed to create absolute minimal configuration: {e}")
            raise RuntimeError("Cannot create any configuration - system is in critical state")
    
    def _audit_log_minimal_config_creation(self):
        """Log minimal configuration creation to audit log."""
        if not self.audit_enabled:
            return
        
        audit_entry = {
            'event_type': 'minimal_config_created',
            'timestamp': datetime.now().isoformat(),
            'user': os.getenv('USER', 'unknown'),
            'process_id': os.getpid(),
            'reason': 'fallback_mechanism_triggered'
        }
        
        self.audit_logger.critical(json.dumps(audit_entry))
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors for monitoring and debugging.
        
        Returns:
            Error summary dictionary
        """
        with self._lock:
            recent_errors = [
                error for error in self.error_history
                if (datetime.now() - error.timestamp).total_seconds() < 3600  # Last hour
            ]
            
            return {
                'total_errors': len(self.error_history),
                'recent_errors': len(recent_errors),
                'error_counts_by_category': self._get_error_counts_by_category(),
                'error_counts_by_severity': self._get_error_counts_by_severity(),
                'error_counts_by_component': dict(self.error_counts),
                'critical_errors': len([e for e in recent_errors if e.severity == ConfigErrorSeverity.CRITICAL]),
                'fallback_enabled': self.fallback_enabled,
                'audit_enabled': self.audit_enabled,
                'registered_fallbacks': list(self.fallback_configs.keys())
            }
    
    def _get_error_counts_by_category(self) -> Dict[str, int]:
        """Get error counts grouped by category."""
        counts = {}
        for error in self.error_history:
            category = error.category.value
            counts[category] = counts.get(category, 0) + 1
        return counts
    
    def _get_error_counts_by_severity(self) -> Dict[str, int]:
        """Get error counts grouped by severity."""
        counts = {}
        for error in self.error_history:
            severity = error.severity.value
            counts[severity] = counts.get(severity, 0) + 1
        return counts
    
    def clear_error_history(self, older_than_hours: int = 24):
        """Clear old errors from history.
        
        Args:
            older_than_hours: Clear errors older than this many hours
        """
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        with self._lock:
            original_count = len(self.error_history)
            self.error_history = [
                error for error in self.error_history
                if error.timestamp > cutoff_time
            ]
            cleared_count = original_count - len(self.error_history)
            
            if cleared_count > 0:
                self.logger.info(f"Cleared {cleared_count} old errors from history")
    
    def enable_audit_logging(self, enabled: bool = True):
        """Enable or disable audit logging.
        
        Args:
            enabled: Whether to enable audit logging
        """
        self.audit_enabled = enabled
        self.logger.info(f"Audit logging {'enabled' if enabled else 'disabled'}")
    
    def enable_fallback_configs(self, enabled: bool = True):
        """Enable or disable fallback configurations.
        
        Args:
            enabled: Whether to enable fallback configurations
        """
        self.fallback_enabled = enabled
        self.logger.info(f"Fallback configurations {'enabled' if enabled else 'disabled'}")


# Global error handler instance
_error_handler: Optional[ConfigurationErrorHandler] = None
_handler_lock = threading.Lock()


def get_error_handler() -> ConfigurationErrorHandler:
    """Get the global configuration error handler instance."""
    global _error_handler
    
    if _error_handler is None:
        with _handler_lock:
            if _error_handler is None:
                _error_handler = ConfigurationErrorHandler()
    
    return _error_handler


def handle_config_error(
    category: ConfigErrorCategory,
    severity: ConfigErrorSeverity,
    message: str,
    component: str,
    exception: Optional[Exception] = None,
    context: Optional[Dict[str, Any]] = None,
    suggested_fix: Optional[str] = None,
    error_code: Optional[str] = None
) -> ConfigError:
    """Convenience function to handle configuration errors.
    
    Args:
        category: Error category
        severity: Error severity
        message: Error message
        component: Component that generated the error
        exception: Original exception if any
        context: Additional context information
        suggested_fix: Suggested fix for the error
        error_code: Unique error code
        
    Returns:
        ConfigError instance
    """
    return get_error_handler().handle_error(
        category=category,
        severity=severity,
        message=message,
        component=component,
        exception=exception,
        context=context,
        suggested_fix=suggested_fix,
        error_code=error_code
    )


# Convenience functions for common error types
def handle_loading_error(component: str, message: str, exception: Exception = None, **kwargs):
    """Handle configuration loading errors."""
    return handle_config_error(
        category=ConfigErrorCategory.LOADING,
        severity=ConfigErrorSeverity.HIGH,
        message=message,
        component=component,
        exception=exception,
        **kwargs
    )


def handle_validation_error(component: str, message: str, exception: Exception = None, **kwargs):
    """Handle configuration validation errors."""
    return handle_config_error(
        category=ConfigErrorCategory.VALIDATION,
        severity=ConfigErrorSeverity.MEDIUM,
        message=message,
        component=component,
        exception=exception,
        **kwargs
    )


def handle_connection_error(component: str, message: str, exception: Exception = None, **kwargs):
    """Handle connection-related errors."""
    return handle_config_error(
        category=ConfigErrorCategory.CONNECTION,
        severity=ConfigErrorSeverity.MEDIUM,
        message=message,
        component=component,
        exception=exception,
        **kwargs
    )


def handle_security_error(component: str, message: str, exception: Exception = None, **kwargs):
    """Handle security-related errors."""
    return handle_config_error(
        category=ConfigErrorCategory.SECURITY,
        severity=ConfigErrorSeverity.HIGH,
        message=message,
        component=component,
        exception=exception,
        **kwargs
    )