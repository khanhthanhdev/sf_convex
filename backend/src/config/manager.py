"""
Unified configuration manager that replaces existing config systems.

This module provides a centralized configuration system that integrates
the original config.py, langgraph config, and the new Pydantic-based models.
"""

import os
import json
import logging
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from dataclasses import asdict

from .models import SystemConfig, AgentConfig, ValidationResult, LLMProviderConfig, RAGConfig, MonitoringConfig
from .factory import ConfigurationFactory
from .service import ConfigurationService
from .error_handling import (
    ConfigurationErrorHandler, ConfigErrorCategory, ConfigErrorSeverity,
    handle_config_error, get_error_handler
)
from .logging_config import get_config_logger
from .fallback import get_fallback_provider


logger = logging.getLogger(__name__)


class ConfigurationManager:
    """Unified configuration manager for the entire system.
    
    This class provides a single interface for all configuration needs,
    replacing the existing config.py and langgraph config systems.
    
    Features:
    - Singleton pattern for centralized access
    - Configuration caching for performance
    - Hot-reloading for development
    - Component-specific configuration access
    - Validation and error handling
    """
    
    _instance: Optional['ConfigurationManager'] = None
    _config: Optional[SystemConfig] = None
    _config_cache: Dict[str, Any] = {}
    _cache_timestamps: Dict[str, float] = {}
    _cache_ttl: int = 300  # 5 minutes cache TTL
    _lock = threading.RLock()
    
    def __new__(cls) -> 'ConfigurationManager':
        """Singleton pattern to ensure single configuration instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize configuration manager."""
        if not hasattr(self, '_initialized'):
            # Initialize error handling and logging first
            self.error_handler = get_error_handler()
            self.config_logger = get_config_logger()
            self.fallback_provider = get_fallback_provider()
            
            # Set logging context
            self.config_logger.set_context(component='configuration_manager')
            
            with self.error_handler.error_context('configuration_manager', 'initialization'):
                self.config_dir = Path("config")
                self.config_dir.mkdir(exist_ok=True)
                
                # Configuration file paths
                self.system_config_path = self.config_dir / "system_config.json"
                self.backup_config_path = self.config_dir / "system_config_backup.json"
                
                # Initialize configuration service
                self.config_service = ConfigurationService()
                
                # Development mode settings
                self.development_mode = os.getenv('ENVIRONMENT', 'development') == 'development'
                self.hot_reload_enabled = self.development_mode and os.getenv('ENABLE_HOT_RELOAD', 'true').lower() == 'true'
                
                # Cache settings
                self._cache_ttl = int(os.getenv('CONFIG_CACHE_TTL', '300'))
                
                # Configuration change notification system
                self._change_callbacks: List[Callable[[SystemConfig, SystemConfig], None]] = []
                self._notification_lock = threading.RLock()
                
                # Register fallback configurations
                self._register_fallback_configs()
                
                # Start file watching if enabled
                if self.hot_reload_enabled:
                    self._start_file_watching()
                
                self._initialized = True
                
                # Log successful initialization
                self.config_logger.info(
                    f"ConfigurationManager initialized successfully",
                    extra={
                        'development_mode': self.development_mode,
                        'hot_reload': self.hot_reload_enabled,
                        'cache_ttl': self._cache_ttl
                    }
                )
                
                # Audit log initialization
                self.config_logger.audit(
                    event_type='configuration_manager_initialized',
                    details={
                        'development_mode': self.development_mode,
                        'hot_reload_enabled': self.hot_reload_enabled,
                        'cache_ttl_seconds': self._cache_ttl
                    }
                )
    
    def _register_fallback_configs(self):
        """Register fallback configurations with the error handler."""
        try:
            # Register basic fallback configurations
            self.error_handler.register_fallback_config('llm_providers', {
                'openai': {
                    'provider': 'openai',
                    'models': ['gpt-4o-mini'],
                    'default_model': 'gpt-4o-mini',
                    'enabled': bool(os.getenv('OPENAI_API_KEY'))
                }
            })
            
            self.error_handler.register_fallback_config('agent_configs', {
                'planner_agent': {
                    'name': 'planner_agent',
                    'enabled': False,
                    'timeout_seconds': 300,
                    'max_retries': 3
                }
            })
            
            self.config_logger.debug("Registered fallback configurations")
            
        except Exception as e:
            self.config_logger.warning(f"Failed to register fallback configurations: {e}")
    
    @property
    def config(self) -> SystemConfig:
        """Get the current system configuration with caching."""
        with self._lock:
            cache_key = 'system_config'
            
            # Check if we have a cached config and it's still valid
            if (self._config is not None and 
                cache_key in self._cache_timestamps and 
                time.time() - self._cache_timestamps[cache_key] < self._cache_ttl):
                return self._config
            
            # Load fresh configuration
            self._config = self.load_system_config()
            self._cache_timestamps[cache_key] = time.time()
            
            return self._config
    
    def load_system_config(self) -> SystemConfig:
        """Load system configuration from environment and files."""
        with self.error_handler.error_context('configuration_manager', 'load_system_config'):
            try:
                self.config_logger.info("Loading system configuration")
                start_time = time.time()
                
                # First try to load from environment variables (primary source)
                config = ConfigurationFactory.build_system_config()
                
                # If a saved config file exists, merge with environment config
                if self.system_config_path.exists():
                    saved_config = self._load_config_from_file()
                    if saved_config:
                        config = self._merge_configs(config, saved_config)
                
                # Validate the configuration
                validation_result = self.validate_configuration(config)
                if not validation_result.valid:
                    self.config_logger.warning(
                        f"Configuration validation failed with {len(validation_result.errors)} errors",
                        extra={'errors': validation_result.errors}
                    )
                    
                    # Log each error with appropriate handling
                    for error in validation_result.errors:
                        handle_config_error(
                            category=ConfigErrorCategory.VALIDATION,
                            severity=ConfigErrorSeverity.MEDIUM,
                            message=error,
                            component='configuration_manager',
                            suggested_fix="Review configuration values and fix validation errors"
                        )
                
                # Log warnings
                for warning in validation_result.warnings:
                    self.config_logger.warning(f"Configuration warning: {warning}")
                
                self._config = config
                
                # Log performance metrics
                duration = time.time() - start_time
                self.config_logger.performance(
                    operation='load_system_config',
                    duration=duration,
                    details={
                        'validation_errors': len(validation_result.errors),
                        'validation_warnings': len(validation_result.warnings),
                        'config_valid': validation_result.valid
                    }
                )
                
                self.config_logger.info("System configuration loaded successfully")
                return config
                
            except Exception as e:
                # Handle the error through error handler
                handle_config_error(
                    category=ConfigErrorCategory.LOADING,
                    severity=ConfigErrorSeverity.HIGH,
                    message=f"Failed to load system configuration: {str(e)}",
                    component='configuration_manager',
                    exception=e,
                    suggested_fix="Check environment variables and configuration files"
                )
                
                # Try to get fallback configuration
                fallback_config = self.fallback_provider.get_fallback_config('system_config')
                if fallback_config:
                    self.config_logger.warning("Using fallback system configuration")
                    return fallback_config
                
                # Last resort - create minimal config
                return self._create_minimal_config()
    
    def _load_config_from_file(self) -> Optional[SystemConfig]:
        """Load configuration from saved file."""
        try:
            with open(self.system_config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Convert agent configs
            agents = {}
            for name, agent_dict in config_dict.get('agent_configs', {}).items():
                agents[name] = AgentConfig(**agent_dict)
            
            # Build SystemConfig with saved data
            # Note: This is a simplified version - in practice you'd need to reconstruct all nested objects
            return None  # For now, rely on environment variables only
            
        except Exception as e:
            logger.error(f"Failed to load config from file: {e}")
            return None
    
    def _merge_configs(self, env_config: SystemConfig, file_config: SystemConfig) -> SystemConfig:
        """Merge environment config with file config (env takes precedence)."""
        # For now, just return env_config
        # In a full implementation, you'd merge the configurations intelligently
        return env_config
    
    def _create_minimal_config(self) -> SystemConfig:
        """Create a minimal working configuration."""
        try:
            self.config_logger.warning("Creating minimal configuration as fallback")
            
            # Try to use fallback provider first
            fallback_config = self.fallback_provider.get_fallback_config('system_config')
            if fallback_config:
                return fallback_config
            
            # Try to build a basic configuration with defaults
            from .factory import ConfigurationFactory
            return ConfigurationFactory.build_system_config()
            
        except Exception as e:
            # Handle the error
            handle_config_error(
                category=ConfigErrorCategory.LOADING,
                severity=ConfigErrorSeverity.CRITICAL,
                message=f"Failed to create minimal configuration: {str(e)}",
                component='configuration_manager',
                exception=e,
                suggested_fix="Check system dependencies and basic environment setup"
            )
            
            # Use error handler's emergency config
            return self.error_handler.create_minimal_config()
    
    def save_system_config(self, config: Optional[SystemConfig] = None) -> bool:
        """Save system configuration to file."""
        with self.error_handler.error_context('configuration_manager', 'save_system_config'):
            try:
                if config is None:
                    config = self.config
                
                start_time = time.time()
                
                # Create backup of existing config
                if self.system_config_path.exists():
                    import shutil
                    shutil.copy2(self.system_config_path, self.backup_config_path)
                    self.config_logger.debug("Created backup of existing configuration")
                
                # Convert to serializable format
                config_dict = self._config_to_dict(config)
                
                # Write configuration file
                with open(self.system_config_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
                
                # Log performance metrics
                duration = time.time() - start_time
                self.config_logger.performance(
                    operation='save_system_config',
                    duration=duration,
                    details={'config_size_bytes': len(json.dumps(config_dict))}
                )
                
                # Audit log configuration save
                self.config_logger.audit(
                    event_type='configuration_saved',
                    details={
                        'config_file': str(self.system_config_path),
                        'backup_created': self.backup_config_path.exists(),
                        'config_size_bytes': len(json.dumps(config_dict))
                    }
                )
                
                self.config_logger.info("System configuration saved successfully")
                return True
                
            except Exception as e:
                handle_config_error(
                    category=ConfigErrorCategory.LOADING,
                    severity=ConfigErrorSeverity.HIGH,
                    message=f"Failed to save system configuration: {str(e)}",
                    component='configuration_manager',
                    exception=e,
                    context={'config_path': str(self.system_config_path)},
                    suggested_fix="Check file permissions and disk space"
                )
                return False
    
    def _config_to_dict(self, config: SystemConfig) -> Dict[str, Any]:
        """Convert SystemConfig to serializable dictionary."""
        # This is a simplified version - you'd need to handle all nested objects properly
        return {
            'environment': config.environment,
            'debug': config.debug,
            'default_llm_provider': config.default_llm_provider,
            # Add other fields as needed
        }
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        config = self.config
        
        # Determine provider from model name
        if model_name.startswith('openai/'):
            provider_name = 'openai'
        elif model_name.startswith('bedrock/'):
            provider_name = 'bedrock'
        elif model_name.startswith('openrouter/'):
            provider_name = 'openrouter'
        elif model_name.startswith('gemini/'):
            provider_name = 'gemini'
        else:
            provider_name = config.default_llm_provider
        
        provider_config = config.llm_providers.get(provider_name)
        if not provider_config:
            logger.warning(f"No configuration found for provider: {provider_name}")
            return {'model_name': model_name}
        
        return {
            'model_name': model_name,
            'provider': provider_name,
            'api_key': provider_config.api_key,
            'base_url': provider_config.base_url,
            'timeout': provider_config.timeout,
            'max_retries': provider_config.max_retries
        }
    
    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """Get configuration for a specific agent with caching."""
        with self._lock:
            cache_key = f'agent_config_{agent_name}'
            
            # Check cache first
            if (cache_key in self._config_cache and 
                cache_key in self._cache_timestamps and 
                time.time() - self._cache_timestamps[cache_key] < self._cache_ttl):
                return self._config_cache[cache_key]
            
            # Get from main config
            agent_config = self.config.agent_configs.get(agent_name)
            
            # Cache the result
            self._config_cache[cache_key] = agent_config
            self._cache_timestamps[cache_key] = time.time()
            
            return agent_config
    
    def get_llm_config(self) -> Dict[str, LLMProviderConfig]:
        """Get LLM provider configurations with caching."""
        with self._lock:
            cache_key = 'llm_config'
            
            # Check cache first
            if (cache_key in self._config_cache and 
                cache_key in self._cache_timestamps and 
                time.time() - self._cache_timestamps[cache_key] < self._cache_ttl):
                return self._config_cache[cache_key]
            
            # Get from main config
            llm_config = self.config.llm_providers
            
            # Cache the result
            self._config_cache[cache_key] = llm_config
            self._cache_timestamps[cache_key] = time.time()
            
            return llm_config
    
    def get_rag_config(self) -> Optional[RAGConfig]:
        """Get RAG configuration with caching."""
        with self._lock:
            cache_key = 'rag_config'
            
            # Check cache first
            if (cache_key in self._config_cache and 
                cache_key in self._cache_timestamps and 
                time.time() - self._cache_timestamps[cache_key] < self._cache_ttl):
                return self._config_cache[cache_key]
            
            # Get from main config
            rag_config = self.config.rag_config
            
            # Cache the result
            self._config_cache[cache_key] = rag_config
            self._cache_timestamps[cache_key] = time.time()
            
            return rag_config
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration with caching."""
        with self._lock:
            cache_key = 'monitoring_config'
            
            # Check cache first
            if (cache_key in self._config_cache and 
                cache_key in self._cache_timestamps and 
                time.time() - self._cache_timestamps[cache_key] < self._cache_ttl):
                return self._config_cache[cache_key]
            
            # Get from main config
            monitoring_config = self.config.monitoring_config
            
            # Cache the result
            self._config_cache[cache_key] = monitoring_config
            self._cache_timestamps[cache_key] = time.time()
            
            return monitoring_config
    
    def get_provider_config(self, provider_name: str) -> Optional[LLMProviderConfig]:
        """Get configuration for a specific LLM provider with caching."""
        with self._lock:
            cache_key = f'provider_config_{provider_name}'
            
            # Check cache first
            if (cache_key in self._config_cache and 
                cache_key in self._cache_timestamps and 
                time.time() - self._cache_timestamps[cache_key] < self._cache_ttl):
                return self._config_cache[cache_key]
            
            # Get from main config
            provider_config = self.config.llm_providers.get(provider_name)
            
            # Cache the result
            self._config_cache[cache_key] = provider_config
            self._cache_timestamps[cache_key] = time.time()
            
            return provider_config
    
    def update_agent_config(self, agent_name: str, updates: Dict[str, Any]) -> bool:
        """Update configuration for a specific agent."""
        try:
            with self._lock:
                config = self.config
                
                if agent_name not in config.agent_configs:
                    logger.error(f"Agent {agent_name} not found in configuration")
                    return False
                
                # Update agent configuration
                agent_config = config.agent_configs[agent_name]
                for key, value in updates.items():
                    if hasattr(agent_config, key):
                        setattr(agent_config, key, value)
                    else:
                        logger.warning(f"Unknown configuration key: {key}")
                
                # Clear cache for this agent
                cache_key = f'agent_config_{agent_name}'
                if cache_key in self._config_cache:
                    del self._config_cache[cache_key]
                if cache_key in self._cache_timestamps:
                    del self._cache_timestamps[cache_key]
                
                # Save updated configuration
                return self.save_system_config(config)
                
        except Exception as e:
            logger.error(f"Failed to update agent configuration: {e}")
            return False
    
    def get_all_agent_names(self) -> List[str]:
        """Get list of all configured agent names."""
        return list(self.config.agent_configs.keys())
    
    def get_all_provider_names(self) -> List[str]:
        """Get list of all configured LLM provider names."""
        return list(self.config.llm_providers.keys())
    
    def get_default_provider(self) -> str:
        """Get the default LLM provider name."""
        return self.config.default_llm_provider
    
    def is_provider_enabled(self, provider_name: str) -> bool:
        """Check if a specific provider is enabled."""
        provider_config = self.get_provider_config(provider_name)
        return provider_config is not None and provider_config.enabled
    
    def is_agent_enabled(self, agent_name: str) -> bool:
        """Check if a specific agent is enabled."""
        agent_config = self.get_agent_config(agent_name)
        return agent_config is not None and agent_config.enabled
    
    def is_rag_enabled(self) -> bool:
        """Check if RAG system is enabled."""
        rag_config = self.get_rag_config()
        return rag_config is not None and rag_config.enabled
    
    def is_monitoring_enabled(self) -> bool:
        """Check if monitoring is enabled."""
        monitoring_config = self.get_monitoring_config()
        return monitoring_config.enabled
    
    def get_environment(self) -> str:
        """Get current environment (development, staging, production)."""
        return self.config.environment
    
    def is_development_mode(self) -> bool:
        """Check if running in development mode."""
        return self.development_mode
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration for debugging."""
        config = self.config
        
        return {
            'environment': config.environment,
            'debug': config.debug,
            'default_llm_provider': config.default_llm_provider,
            'llm_providers': list(config.llm_providers.keys()),
            'enabled_providers': [name for name, provider in config.llm_providers.items() if provider.enabled],
            'agents': list(config.agent_configs.keys()),
            'enabled_agents': [name for name, agent in config.agent_configs.items() if agent.enabled],
            'rag_enabled': config.rag_config.enabled if config.rag_config else False,
            'monitoring_enabled': config.monitoring_config.enabled,
            'hot_reload_enabled': self.hot_reload_enabled,
            'cache_enabled': self.is_cache_enabled(),
            'cache_ttl': self._cache_ttl,
            'cached_items': len(self._config_cache)
        }
    
    def validate_configuration(self, config: Optional[SystemConfig] = None) -> ValidationResult:
        """Validate system configuration."""
        with self.error_handler.error_context('configuration_manager', 'validate_configuration'):
            if config is None:
                config = self.config
            
            result = ValidationResult(valid=True)
            start_time = time.time()
            
            try:
                self.config_logger.debug("Starting configuration validation")
                
                # Validate required agents for langgraph system
                required_agents = ['planner_agent', 'code_generator_agent', 'renderer_agent']
                for agent_name in required_agents:
                    if agent_name not in config.agent_configs:
                        result.add_error(f"Required agent missing: {agent_name}")
                        handle_config_error(
                            category=ConfigErrorCategory.VALIDATION,
                            severity=ConfigErrorSeverity.HIGH,
                            message=f"Required agent missing: {agent_name}",
                            component='configuration_manager',
                            suggested_fix=f"Add configuration for {agent_name} in agent_configs"
                        )
                
                # Validate agent configurations
                for agent_name, agent_config in config.agent_configs.items():
                    if not agent_config.name:
                        result.add_error(f"Agent {agent_name} missing name")
                    
                    if agent_config.timeout_seconds <= 0:
                        result.add_error(f"Agent {agent_name} has invalid timeout_seconds")
                    
                    if agent_config.max_retries < 0:
                        result.add_error(f"Agent {agent_name} has invalid max_retries")
                
                # Validate LLM provider configurations
                if not config.llm_providers:
                    result.add_warning("No LLM providers configured")
                    handle_config_error(
                        category=ConfigErrorCategory.VALIDATION,
                        severity=ConfigErrorSeverity.MEDIUM,
                        message="No LLM providers configured",
                        component='configuration_manager',
                        suggested_fix="Configure at least one LLM provider with API key"
                    )
                else:
                    for provider_name, provider_config in config.llm_providers.items():
                        if not provider_config.api_key and provider_name != 'local':
                            result.add_warning(f"No API key configured for provider: {provider_name}")
                
                # Validate RAG configuration
                if config.rag_config and config.rag_config.enabled:
                    if not config.rag_config.embedding_config.api_key and config.rag_config.embedding_config.provider != 'local':
                        result.add_warning(f"No API key for embedding provider: {config.rag_config.embedding_config.provider}")
                
                # Validate workflow settings
                if config.workflow_config.max_workflow_retries < 0:
                    result.add_error("Invalid max_workflow_retries")
                
                if config.workflow_config.workflow_timeout_seconds <= 0:
                    result.add_error("Invalid workflow_timeout_seconds")
                
                # Log validation results
                duration = time.time() - start_time
                self.config_logger.performance(
                    operation='validate_configuration',
                    duration=duration,
                    details={
                        'validation_errors': len(result.errors),
                        'validation_warnings': len(result.warnings),
                        'config_valid': result.valid
                    }
                )
                
                if result.valid:
                    self.config_logger.debug("Configuration validation passed")
                else:
                    self.config_logger.warning(
                        f"Configuration validation failed with {len(result.errors)} errors",
                        extra={'errors': result.errors}
                    )
                
            except Exception as e:
                result.add_error(f"Validation error: {str(e)}")
                handle_config_error(
                    category=ConfigErrorCategory.VALIDATION,
                    severity=ConfigErrorSeverity.HIGH,
                    message=f"Configuration validation failed: {str(e)}",
                    component='configuration_manager',
                    exception=e,
                    suggested_fix="Check configuration structure and values"
                )
            
            return result
    
    def get_compatible_initialization_params(self) -> Dict[str, Any]:
        """Get initialization parameters compatible with existing system."""
        config = self.config
        
        return {
            'output_dir': config.workflow_config.output_dir,
            'print_response': False,  # Default value
            'use_rag': config.rag_config.enabled if config.rag_config else True,
            'use_context_learning': True,  # Default value
            'context_learning_path': 'data/context_learning',
            'chroma_db_path': config.rag_config.vector_store_config.connection_params.get('db_path', 'data/rag/chroma_db') if config.rag_config else 'data/rag/chroma_db',
            'manim_docs_path': 'data/rag/manim_docs',
            'embedding_model': config.rag_config.embedding_config.model_name if config.rag_config else 'hf:ibm-granite/granite-embedding-30m-english',
            'use_visual_fix_code': False,  # Default value
            'use_langfuse': config.monitoring_config.langfuse_config.enabled if config.monitoring_config.langfuse_config else True,
            'max_scene_concurrency': config.workflow_config.max_scene_concurrency,
            'max_topic_concurrency': config.workflow_config.max_topic_concurrency,
            'max_retries': 5,  # Default value
            'enable_caching': config.rag_config.enable_caching if config.rag_config else True,
            'default_quality': config.workflow_config.default_quality,
            'use_gpu_acceleration': config.workflow_config.use_gpu_acceleration,
            'preview_mode': config.workflow_config.preview_mode,
            'max_concurrent_renders': config.workflow_config.max_concurrent_renders
        }
    
    def get_workflow_config(self):
        """Get workflow configuration with caching."""
        with self._lock:
            cache_key = 'workflow_config'
            
            # Check cache first
            if (cache_key in self._config_cache and 
                cache_key in self._cache_timestamps and 
                time.time() - self._cache_timestamps[cache_key] < self._cache_ttl):
                return self._config_cache[cache_key]
            
            # Get from main config
            workflow_config = self.config.workflow_config
            
            # Cache the result
            self._config_cache[cache_key] = workflow_config
            self._cache_timestamps[cache_key] = time.time()
            
            return workflow_config
    
    def get_human_loop_config(self):
        """Get human loop configuration with caching."""
        with self._lock:
            cache_key = 'human_loop_config'
            
            # Check cache first
            if (cache_key in self._config_cache and 
                cache_key in self._cache_timestamps and 
                time.time() - self._cache_timestamps[cache_key] < self._cache_ttl):
                return self._config_cache[cache_key]
            
            # Get from main config
            human_loop_config = self.config.human_loop_config
            
            # Cache the result
            self._config_cache[cache_key] = human_loop_config
            self._cache_timestamps[cache_key] = time.time()
            
            return human_loop_config
    
    def get_mcp_servers_config(self):
        """Get MCP servers configuration with caching."""
        with self._lock:
            cache_key = 'mcp_servers_config'
            
            # Check cache first
            if (cache_key in self._config_cache and 
                cache_key in self._cache_timestamps and 
                time.time() - self._cache_timestamps[cache_key] < self._cache_ttl):
                return self._config_cache[cache_key]
            
            # Get from main config
            mcp_config = self.config.mcp_servers
            
            # Cache the result
            self._config_cache[cache_key] = mcp_config
            self._cache_timestamps[cache_key] = time.time()
            
            return mcp_config
    
    def get_context7_config(self):
        """Get Context7 configuration with caching."""
        with self._lock:
            cache_key = 'context7_config'
            
            # Check cache first
            if (cache_key in self._config_cache and 
                cache_key in self._cache_timestamps and 
                time.time() - self._cache_timestamps[cache_key] < self._cache_ttl):
                return self._config_cache[cache_key]
            
            # Get from main config
            context7_config = self.config.context7_config
            
            # Cache the result
            self._config_cache[cache_key] = context7_config
            self._cache_timestamps[cache_key] = time.time()
            
            return context7_config
    
    def get_docling_config(self):
        """Get Docling configuration with caching."""
        with self._lock:
            cache_key = 'docling_config'
            
            # Check cache first
            if (cache_key in self._config_cache and 
                cache_key in self._cache_timestamps and 
                time.time() - self._cache_timestamps[cache_key] < self._cache_ttl):
                return self._config_cache[cache_key]
            
            # Get from main config
            docling_config = self.config.docling_config
            
            # Cache the result
            self._config_cache[cache_key] = docling_config
            self._cache_timestamps[cache_key] = time.time()
            
            return docling_config
    
    def reload_config(self, force: bool = False) -> SystemConfig:
        """Reload configuration from environment variables.
        
        Args:
            force: Force reload even if not in development mode
            
        Returns:
            Reloaded SystemConfig instance
        """
        if not self.development_mode and not force:
            logger.warning("Configuration reload is disabled in production mode")
            return self.config
        
        # Use safe reload if in development mode, otherwise use legacy behavior
        if self.development_mode and self.hot_reload_enabled:
            success = self.safe_reload_config()
            if success:
                return self.config
            else:
                logger.warning("Safe reload failed, returning current configuration")
                return self._config or self._create_minimal_config()
        
        # Legacy reload behavior for forced reloads or when hot-reload is disabled
        with self._lock:
            try:
                logger.info("Reloading configuration...")
                
                # Clear all caches
                self.clear_cache()
                
                # Reload environment variables if in development
                if self.development_mode:
                    self.config_service.reload_env_config()
                
                # Force reload of system config
                self._config = None
                new_config = self.config
                
                # Validate the new configuration
                validation_result = self.validate_configuration(new_config)
                if not validation_result.valid:
                    logger.error(f"Configuration reload validation failed: {validation_result.errors}")
                    for error in validation_result.errors:
                        logger.error(f"Config error: {error}")
                else:
                    logger.info("Configuration reloaded successfully")
                
                return new_config
                
            except Exception as e:
                logger.error(f"Failed to reload configuration: {e}")
                # Return current config if reload fails
                return self._config or self._create_minimal_config()
    
    def clear_cache(self):
        """Clear all configuration caches."""
        with self._lock:
            self._config_cache.clear()
            self._cache_timestamps.clear()
            logger.debug("Configuration cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        with self._lock:
            current_time = time.time()
            cache_stats = {
                'cache_size': len(self._config_cache),
                'cache_ttl': self._cache_ttl,
                'cached_items': []
            }
            
            for key, timestamp in self._cache_timestamps.items():
                age = current_time - timestamp
                is_expired = age > self._cache_ttl
                cache_stats['cached_items'].append({
                    'key': key,
                    'age_seconds': age,
                    'expired': is_expired
                })
            
            return cache_stats
    
    def set_cache_ttl(self, ttl_seconds: int):
        """Set cache TTL for configuration items.
        
        Args:
            ttl_seconds: Time to live in seconds
        """
        if ttl_seconds < 0:
            raise ValueError("Cache TTL must be non-negative")
        
        with self._lock:
            self._cache_ttl = ttl_seconds
            logger.info(f"Configuration cache TTL set to {ttl_seconds} seconds")
    
    def enable_hot_reload(self, enabled: bool = True):
        """Enable or disable hot-reloading for development.
        
        Args:
            enabled: Whether to enable hot-reloading
        """
        if not self.development_mode and enabled:
            logger.warning("Hot-reload can only be enabled in development mode")
            return
        
        old_enabled = self.hot_reload_enabled
        self.hot_reload_enabled = enabled
        
        if enabled and not old_enabled:
            self._start_file_watching()
        elif not enabled and old_enabled:
            self._stop_file_watching()
        
        logger.info(f"Hot-reload {'enabled' if enabled else 'disabled'}")
    
    def is_cache_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self._cache_ttl > 0
    
    def add_config_change_callback(self, callback: Callable[[SystemConfig, SystemConfig], None]):
        """Add a callback to be notified when configuration changes.
        
        Args:
            callback: Function that takes (old_config, new_config) as arguments
        """
        with self._notification_lock:
            if callback not in self._change_callbacks:
                self._change_callbacks.append(callback)
                logger.debug(f"Added configuration change callback: {callback.__name__}")
    
    def remove_config_change_callback(self, callback: Callable[[SystemConfig, SystemConfig], None]):
        """Remove a configuration change callback.
        
        Args:
            callback: Function to remove
        """
        with self._notification_lock:
            if callback in self._change_callbacks:
                self._change_callbacks.remove(callback)
                logger.debug(f"Removed configuration change callback: {callback.__name__}")
    
    def _notify_config_change(self, old_config: SystemConfig, new_config: SystemConfig):
        """Notify all registered callbacks about configuration changes.
        
        Args:
            old_config: Previous configuration
            new_config: New configuration
        """
        with self._notification_lock:
            for callback in self._change_callbacks.copy():  # Copy to avoid modification during iteration
                try:
                    callback(old_config, new_config)
                except Exception as e:
                    logger.error(f"Error in configuration change callback {callback.__name__}: {e}")
    
    def _start_file_watching(self):
        """Start watching configuration files for changes."""
        if not self.hot_reload_enabled:
            return
        
        try:
            success = self.config_service.watch_config_files(self._on_config_file_changed)
            if success:
                logger.info("Configuration file watching started")
            else:
                logger.warning("Failed to start configuration file watching")
        except Exception as e:
            logger.error(f"Error starting file watching: {e}")
    
    def _stop_file_watching(self):
        """Stop watching configuration files."""
        try:
            self.config_service.stop_watching_config_files()
            logger.info("Configuration file watching stopped")
        except Exception as e:
            logger.error(f"Error stopping file watching: {e}")
    
    def _on_config_file_changed(self, file_path: str):
        """Handle configuration file changes with safe reloading.
        
        Args:
            file_path: Path to the changed file
        """
        if not self.hot_reload_enabled:
            return
        
        logger.info(f"Configuration file changed: {file_path}")
        
        try:
            # Perform safe configuration reload
            self.safe_reload_config()
        except Exception as e:
            logger.error(f"Failed to reload configuration after file change: {e}")
    
    def safe_reload_config(self) -> bool:
        """Safely reload configuration with validation and rollback capability.
        
        Returns:
            True if reload was successful, False if rolled back to previous config
        """
        if not self.development_mode:
            logger.warning("Safe configuration reload is only available in development mode")
            return False
        
        with self._lock:
            old_config = self._config
            
            try:
                logger.info("Starting safe configuration reload...")
                
                # Clear caches to force fresh load
                self.clear_cache()
                
                # Reload environment variables
                self.config_service.reload_env_config()
                
                # Load new configuration
                self._config = None
                new_config = self.config
                
                # Validate the new configuration
                validation_result = self.validate_configuration(new_config)
                
                if not validation_result.valid:
                    logger.error(f"New configuration is invalid: {validation_result.errors}")
                    
                    # Rollback to previous configuration
                    self._config = old_config
                    logger.warning("Rolled back to previous configuration due to validation errors")
                    
                    return False
                
                # Configuration is valid, notify callbacks
                if old_config:
                    self._notify_config_change(old_config, new_config)
                
                logger.info("Configuration reloaded successfully")
                return True
                
            except Exception as e:
                logger.error(f"Error during configuration reload: {e}")
                
                # Rollback to previous configuration
                if old_config:
                    self._config = old_config
                    logger.warning("Rolled back to previous configuration due to reload error")
                
                return False
    
    def get_file_watcher_status(self) -> dict:
        """Get status of the configuration file watcher.
        
        Returns:
            Dictionary with watcher status information
        """
        return self.config_service.get_file_watcher_status()
    
    def __del__(self):
        """Cleanup when the configuration manager is destroyed."""
        try:
            if hasattr(self, 'hot_reload_enabled') and self.hot_reload_enabled:
                self._stop_file_watching()
        except Exception:
            pass  # Ignore errors during cleanup


# Global configuration manager instance
config_manager = ConfigurationManager()


# Compatibility functions for existing code
def get_config() -> SystemConfig:
    """Get the current system configuration."""
    return config_manager.config


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model."""
    return config_manager.get_model_config(model_name)


def get_agent_config(agent_name: str) -> Optional[AgentConfig]:
    """Get configuration for a specific agent."""
    return config_manager.get_agent_config(agent_name)


# Legacy Config class for backward compatibility
class Config:
    """Legacy Config class for backward compatibility."""
    
    @property
    def OUTPUT_DIR(self) -> str:
        return config_manager.config.workflow_config.output_dir
    
    @property
    def THEOREMS_PATH(self) -> str:
        return os.path.join("data", "easy_20.json")
    
    @property
    def CONTEXT_LEARNING_PATH(self) -> str:
        return "data/context_learning"
    
    @property
    def CHROMA_DB_PATH(self) -> str:
        config = config_manager.config
        if config.rag_config and config.rag_config.vector_store_config.provider == 'chroma':
            return config.rag_config.vector_store_config.connection_params.get('db_path', 'data/rag/chroma_db')
        return 'data/rag/chroma_db'
    
    @property
    def MANIM_DOCS_PATH(self) -> str:
        return "data/rag/manim_docs"
    
    @property
    def EMBEDDING_MODEL(self) -> str:
        config = config_manager.config
        if config.rag_config:
            return config.rag_config.embedding_config.model_name
        return "hf:ibm-granite/granite-embedding-30m-english"
    
    @property
    def KOKORO_MODEL_PATH(self) -> Optional[str]:
        return config_manager.config.kokoro_model_path
    
    @property
    def KOKORO_VOICES_PATH(self) -> Optional[str]:
        return config_manager.config.kokoro_voices_path
    
    @property
    def KOKORO_DEFAULT_VOICE(self) -> str:
        return config_manager.config.kokoro_default_voice
    
    @property
    def KOKORO_DEFAULT_SPEED(self) -> float:
        return config_manager.config.kokoro_default_speed
    
    @property
    def KOKORO_DEFAULT_LANG(self) -> str:
        return config_manager.config.kokoro_default_lang