"""
Configuration audit logging system for security and debugging.

This module provides comprehensive audit logging for configuration operations,
including security events, configuration changes, and system access patterns.
"""

import logging
import json
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib


class AuditEventType(Enum):
    """Types of audit events."""
    CONFIG_LOADED = "config_loaded"
    CONFIG_SAVED = "config_saved"
    CONFIG_CHANGED = "config_changed"
    CONFIG_VALIDATED = "config_validated"
    CONFIG_FAILED = "config_failed"
    API_KEY_USED = "api_key_used"
    API_KEY_VALIDATED = "api_key_validated"
    FALLBACK_USED = "fallback_used"
    SECURITY_VIOLATION = "security_violation"
    ACCESS_DENIED = "access_denied"
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    HOT_RELOAD = "hot_reload"
    MIGRATION_APPLIED = "migration_applied"
    BACKUP_CREATED = "backup_created"
    EMERGENCY_CONFIG = "emergency_config"


class AuditSeverity(Enum):
    """Severity levels for audit events."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Represents an audit event."""
    event_type: AuditEventType
    severity: AuditSeverity
    timestamp: datetime
    user: str
    process_id: int
    thread_id: int
    component: str
    message: str
    details: Dict[str, Any]
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary."""
        return {
            'event_type': self.event_type.value,
            'severity': self.severity.value,
            'timestamp': self.timestamp.isoformat(),
            'user': self.user,
            'process_id': self.process_id,
            'thread_id': self.thread_id,
            'component': self.component,
            'message': self.message,
            'details': self.details,
            'session_id': self.session_id,
            'request_id': self.request_id
        }
    
    def to_json(self) -> str:
        """Convert audit event to JSON string."""
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False)


class ConfigurationAuditor:
    """Audit logger for configuration operations."""
    
    def __init__(self, audit_dir: Optional[str] = None, retention_days: int = 90):
        """Initialize the configuration auditor.
        
        Args:
            audit_dir: Directory for audit logs. Defaults to 'logs/audit'
            retention_days: Number of days to retain audit logs
        """
        self.audit_dir = Path(audit_dir or 'logs/audit')
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        
        # Setup audit logger
        self.logger = self._setup_audit_logger()
        
        # Thread-local storage for context
        self._local = threading.local()
        
        # Event counters for monitoring
        self.event_counters: Dict[str, int] = {}
        self._counter_lock = threading.RLock()
        
        # Security monitoring
        self.security_events: List[AuditEvent] = []
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self._security_lock = threading.RLock()
        
        self.logger.info("ConfigurationAuditor initialized")
    
    def _setup_audit_logger(self) -> logging.Logger:
        """Setup the audit logger with proper formatting and rotation."""
        audit_logger = logging.getLogger('config.audit')
        audit_logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in audit_logger.handlers[:]:
            audit_logger.removeHandler(handler)
        
        # Create rotating file handler
        from logging.handlers import RotatingFileHandler
        
        audit_file = self.audit_dir / 'config_audit.log'
        handler = RotatingFileHandler(
            audit_file,
            maxBytes=100*1024*1024,  # 100MB
            backupCount=10
        )
        
        # Use structured JSON formatter
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        audit_logger.addHandler(handler)
        
        # Prevent propagation to avoid duplicate logs
        audit_logger.propagate = False
        
        return audit_logger
    
    def set_context(self, session_id: Optional[str] = None, request_id: Optional[str] = None):
        """Set audit context for current thread.
        
        Args:
            session_id: Session identifier
            request_id: Request identifier
        """
        self._local.session_id = session_id
        self._local.request_id = request_id
    
    def clear_context(self):
        """Clear audit context for current thread."""
        self._local.session_id = None
        self._local.request_id = None
    
    def log_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        component: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        user: Optional[str] = None
    ):
        """Log an audit event.
        
        Args:
            event_type: Type of audit event
            severity: Event severity
            component: Component generating the event
            message: Event message
            details: Additional event details
            user: User associated with the event
        """
        try:
            # Create audit event
            event = AuditEvent(
                event_type=event_type,
                severity=severity,
                timestamp=datetime.now(),
                user=user or os.getenv('USER', 'unknown'),
                process_id=os.getpid(),
                thread_id=threading.get_ident(),
                component=component,
                message=message,
                details=details or {},
                session_id=getattr(self._local, 'session_id', None),
                request_id=getattr(self._local, 'request_id', None)
            )
            
            # Log the event
            self.logger.info(event.to_json())
            
            # Update counters
            with self._counter_lock:
                counter_key = f"{event_type.value}:{severity.value}"
                self.event_counters[counter_key] = self.event_counters.get(counter_key, 0) + 1
            
            # Handle security events
            if event_type in [AuditEventType.SECURITY_VIOLATION, AuditEventType.ACCESS_DENIED]:
                self._handle_security_event(event)
            
        except Exception as e:
            # Fallback logging if audit system fails
            logging.getLogger(__name__).error(f"Audit logging failed: {e}")
    
    def _handle_security_event(self, event: AuditEvent):
        """Handle security-related events with special processing.
        
        Args:
            event: Security audit event
        """
        with self._security_lock:
            # Store security event
            self.security_events.append(event)
            
            # Track failed attempts by user
            user = event.user
            if user not in self.failed_attempts:
                self.failed_attempts[user] = []
            
            self.failed_attempts[user].append(event.timestamp)
            
            # Clean old attempts (keep only last hour)
            cutoff_time = datetime.now() - timedelta(hours=1)
            self.failed_attempts[user] = [
                timestamp for timestamp in self.failed_attempts[user]
                if timestamp > cutoff_time
            ]
            
            # Check for suspicious patterns
            if len(self.failed_attempts[user]) > 5:  # More than 5 failures in an hour
                self.log_event(
                    event_type=AuditEventType.SECURITY_VIOLATION,
                    severity=AuditSeverity.CRITICAL,
                    component='audit_system',
                    message=f"Suspicious activity detected for user {user}",
                    details={
                        'failed_attempts_count': len(self.failed_attempts[user]),
                        'time_window_hours': 1,
                        'original_event': event.to_dict()
                    },
                    user='system'
                )
    
    def log_config_loaded(self, config_source: str, config_size: int, validation_errors: int = 0):
        """Log configuration loading event.
        
        Args:
            config_source: Source of configuration (file, environment, etc.)
            config_size: Size of configuration in bytes
            validation_errors: Number of validation errors
        """
        severity = AuditSeverity.ERROR if validation_errors > 0 else AuditSeverity.INFO
        
        self.log_event(
            event_type=AuditEventType.CONFIG_LOADED,
            severity=severity,
            component='configuration_manager',
            message=f"Configuration loaded from {config_source}",
            details={
                'config_source': config_source,
                'config_size_bytes': config_size,
                'validation_errors': validation_errors,
                'load_timestamp': datetime.now().isoformat()
            }
        )
    
    def log_config_saved(self, config_path: str, config_size: int, backup_created: bool = False):
        """Log configuration saving event.
        
        Args:
            config_path: Path where configuration was saved
            config_size: Size of saved configuration
            backup_created: Whether a backup was created
        """
        self.log_event(
            event_type=AuditEventType.CONFIG_SAVED,
            severity=AuditSeverity.INFO,
            component='configuration_manager',
            message=f"Configuration saved to {config_path}",
            details={
                'config_path': config_path,
                'config_size_bytes': config_size,
                'backup_created': backup_created,
                'save_timestamp': datetime.now().isoformat()
            }
        )
    
    def log_config_change(self, component: str, field: str, old_value: Any, new_value: Any):
        """Log configuration change event.
        
        Args:
            component: Component being changed
            field: Field being changed
            old_value: Previous value
            new_value: New value
        """
        # Hash sensitive values for security
        old_hash = self._hash_sensitive_value(old_value) if self._is_sensitive_field(field) else str(old_value)
        new_hash = self._hash_sensitive_value(new_value) if self._is_sensitive_field(field) else str(new_value)
        
        self.log_event(
            event_type=AuditEventType.CONFIG_CHANGED,
            severity=AuditSeverity.WARNING,
            component=component,
            message=f"Configuration changed: {field}",
            details={
                'field': field,
                'old_value': old_hash,
                'new_value': new_hash,
                'is_sensitive': self._is_sensitive_field(field),
                'change_timestamp': datetime.now().isoformat()
            }
        )
    
    def log_api_key_usage(self, provider: str, operation: str, success: bool, response_time: Optional[float] = None):
        """Log API key usage event.
        
        Args:
            provider: API provider name
            operation: Operation performed
            success: Whether operation was successful
            response_time: Response time in seconds
        """
        severity = AuditSeverity.INFO if success else AuditSeverity.WARNING
        
        self.log_event(
            event_type=AuditEventType.API_KEY_USED,
            severity=severity,
            component=f'provider_{provider}',
            message=f"API key used for {provider} - {operation}",
            details={
                'provider': provider,
                'operation': operation,
                'success': success,
                'response_time_seconds': response_time,
                'usage_timestamp': datetime.now().isoformat()
            }
        )
    
    def log_fallback_usage(self, component: str, reason: str, fallback_type: str):
        """Log fallback configuration usage.
        
        Args:
            component: Component using fallback
            reason: Reason for fallback
            fallback_type: Type of fallback used
        """
        self.log_event(
            event_type=AuditEventType.FALLBACK_USED,
            severity=AuditSeverity.WARNING,
            component=component,
            message=f"Fallback configuration used for {component}",
            details={
                'component': component,
                'reason': reason,
                'fallback_type': fallback_type,
                'fallback_timestamp': datetime.now().isoformat()
            }
        )
    
    def log_system_startup(self, config_valid: bool, startup_time: float):
        """Log system startup event.
        
        Args:
            config_valid: Whether configuration is valid
            startup_time: Startup time in seconds
        """
        severity = AuditSeverity.INFO if config_valid else AuditSeverity.WARNING
        
        self.log_event(
            event_type=AuditEventType.SYSTEM_STARTUP,
            severity=severity,
            component='system',
            message="System startup completed",
            details={
                'config_valid': config_valid,
                'startup_time_seconds': startup_time,
                'startup_timestamp': datetime.now().isoformat(),
                'environment': os.getenv('ENVIRONMENT', 'unknown')
            }
        )
    
    def log_emergency_config(self, reason: str):
        """Log emergency configuration creation.
        
        Args:
            reason: Reason for emergency configuration
        """
        self.log_event(
            event_type=AuditEventType.EMERGENCY_CONFIG,
            severity=AuditSeverity.CRITICAL,
            component='fallback_system',
            message="Emergency configuration created",
            details={
                'reason': reason,
                'emergency_timestamp': datetime.now().isoformat(),
                'system_state': 'critical'
            }
        )
    
    def _is_sensitive_field(self, field: str) -> bool:
        """Check if a field contains sensitive information.
        
        Args:
            field: Field name to check
            
        Returns:
            True if field is sensitive
        """
        sensitive_keywords = [
            'api_key', 'secret', 'password', 'token', 'credential',
            'private_key', 'auth', 'bearer'
        ]
        
        field_lower = field.lower()
        return any(keyword in field_lower for keyword in sensitive_keywords)
    
    def _hash_sensitive_value(self, value: Any) -> str:
        """Hash sensitive values for audit logging.
        
        Args:
            value: Value to hash
            
        Returns:
            Hashed value
        """
        if value is None:
            return 'null'
        
        value_str = str(value)
        if not value_str:
            return 'empty'
        
        # Create hash with first and last 4 characters visible
        hash_obj = hashlib.sha256(value_str.encode()).hexdigest()[:8]
        
        if len(value_str) > 8:
            return f"{value_str[:4]}...{value_str[-4:]} (hash: {hash_obj})"
        else:
            return f"****** (hash: {hash_obj})"
    
    def get_audit_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get audit summary for the specified time period.
        
        Args:
            hours: Number of hours to include in summary
            
        Returns:
            Audit summary dictionary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._counter_lock:
            recent_security_events = [
                event for event in self.security_events
                if event.timestamp > cutoff_time
            ]
            
            return {
                'time_period_hours': hours,
                'total_events': sum(self.event_counters.values()),
                'event_counts_by_type': dict(self.event_counters),
                'security_events_count': len(recent_security_events),
                'failed_attempts_by_user': {
                    user: len(attempts) for user, attempts in self.failed_attempts.items()
                    if attempts and attempts[-1] > cutoff_time
                },
                'audit_log_path': str(self.audit_dir),
                'retention_days': self.retention_days
            }
    
    def cleanup_old_logs(self):
        """Clean up old audit logs based on retention policy."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            # Clean up log files
            for log_file in self.audit_dir.glob('*.log*'):
                try:
                    file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_mtime < cutoff_date:
                        log_file.unlink()
                        self.logger.info(f"Deleted old audit log: {log_file}")
                except Exception as e:
                    self.logger.error(f"Failed to delete old log {log_file}: {e}")
            
            # Clean up in-memory security events
            with self._security_lock:
                self.security_events = [
                    event for event in self.security_events
                    if event.timestamp > cutoff_date
                ]
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old audit logs: {e}")


# Global auditor instance
_auditor: Optional[ConfigurationAuditor] = None
_auditor_lock = threading.Lock()


def get_auditor() -> ConfigurationAuditor:
    """Get the global configuration auditor instance."""
    global _auditor
    
    if _auditor is None:
        with _auditor_lock:
            if _auditor is None:
                _auditor = ConfigurationAuditor()
    
    return _auditor


def audit_config_event(
    event_type: AuditEventType,
    severity: AuditSeverity,
    component: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    user: Optional[str] = None
):
    """Convenience function to log audit events.
    
    Args:
        event_type: Type of audit event
        severity: Event severity
        component: Component generating the event
        message: Event message
        details: Additional event details
        user: User associated with the event
    """
    get_auditor().log_event(
        event_type=event_type,
        severity=severity,
        component=component,
        message=message,
        details=details,
        user=user
    )