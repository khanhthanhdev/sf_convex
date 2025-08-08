"""
Centralized configuration system for the video generation application.

This module provides a unified configuration system that replaces the existing
config.py and integrates with the langgraph multi-agent system.
"""

from .models import (
    SystemConfig,
    AgentConfig,
    LLMProviderConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    RAGConfig,
    MonitoringConfig,
    LangfuseConfig,
    DoclingConfig,
    MCPServerConfig,
    Context7Config,
    HumanLoopConfig,
    WorkflowConfig,
    ValidationResult
)

from .factory import ConfigurationFactory

from .manager import (
    ConfigurationManager,
    config_manager,
    get_config,
    get_model_config,
    get_agent_config,
    Config  # Legacy compatibility
)

from .service import ConfigurationService

from .validation import (
    ConfigValidationService,
    ProviderCompatibility,
    ConnectionTestResult
)

from .health_check import (
    ConfigurationHealthChecker,
    HealthCheckResult,
    ComponentHealth
)

from .migration import (
    ConfigurationMigrator,
    MigrationResult,
    MigrationStep
)

from .startup_validation import StartupValidationService

from .error_handling import (
    ConfigurationErrorHandler,
    ConfigErrorCategory,
    ConfigErrorSeverity,
    handle_config_error,
    get_error_handler
)

from .logging_config import (
    ConfigurationLogger,
    get_config_logger,
    setup_logging
)

from .fallback import (
    FallbackConfigurationProvider,
    get_fallback_provider
)

from .audit import (
    ConfigurationAuditor,
    AuditEventType,
    AuditSeverity,
    get_auditor,
    audit_config_event
)

# For backward compatibility, expose the legacy Config class
__all__ = [
    # Core models
    'SystemConfig',
    'AgentConfig',
    'LLMProviderConfig',
    'EmbeddingConfig',
    'VectorStoreConfig',
    'RAGConfig',
    'MonitoringConfig',
    'LangfuseConfig',
    'DoclingConfig',
    'MCPServerConfig',
    'Context7Config',
    'HumanLoopConfig',
    'WorkflowConfig',
    'ValidationResult',
    
    # Factory
    'ConfigurationFactory',
    
    # Manager and utilities
    'ConfigurationManager',
    'config_manager',
    'get_config',
    'get_model_config',
    'get_agent_config',
    
    # Service
    'ConfigurationService',
    
    # Validation
    'ConfigValidationService',
    'ProviderCompatibility',
    'ConnectionTestResult',
    
    # Health checking
    'ConfigurationHealthChecker',
    'HealthCheckResult',
    'ComponentHealth',
    
    # Migration
    'ConfigurationMigrator',
    'MigrationResult',
    'MigrationStep',
    
    # Startup validation
    'StartupValidationService',
    
    # Error handling and logging
    'ConfigurationErrorHandler',
    'ConfigErrorCategory',
    'ConfigErrorSeverity',
    'handle_config_error',
    'get_error_handler',
    'ConfigurationLogger',
    'get_config_logger',
    'setup_logging',
    'FallbackConfigurationProvider',
    'get_fallback_provider',
    'ConfigurationAuditor',
    'AuditEventType',
    'AuditSeverity',
    'get_auditor',
    'audit_config_event',
    
    # Legacy compatibility
    'Config'
]