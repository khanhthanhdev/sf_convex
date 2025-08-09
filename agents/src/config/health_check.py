"""
Configuration Health Check Service for monitoring system configuration status.

This module provides health check endpoints and utilities for monitoring
the configuration system's health and status.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from .models import SystemConfig, ValidationResult
from .validation import ConfigValidationService
from .manager import ConfigurationManager


logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Result of a configuration health check."""
    
    status: str  # "healthy", "warning", "critical"
    timestamp: datetime
    response_time_ms: float
    checks: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'status': self.status,
            'timestamp': self.timestamp.isoformat(),
            'response_time_ms': self.response_time_ms,
            'checks': self.checks,
            'errors': self.errors,
            'warnings': self.warnings,
            'metadata': self.metadata
        }


@dataclass
class ComponentHealth:
    """Health status of a system component."""
    
    name: str
    status: str  # "healthy", "warning", "critical", "unknown"
    message: str
    response_time_ms: Optional[float] = None
    last_check: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.last_check is None:
            self.last_check = datetime.now()


class ConfigurationHealthChecker:
    """Service for checking configuration health and status.
    
    This service provides comprehensive health checking for the configuration
    system, including validation status, connectivity, and performance metrics.
    """
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """Initialize the health checker.
        
        Args:
            config_manager: Configuration manager instance (optional)
        """
        self.config_manager = config_manager or ConfigurationManager()
        self.validator = ConfigValidationService()
        
        # Health check cache
        self._health_cache: Dict[str, ComponentHealth] = {}
        self._cache_ttl = 60  # 1 minute cache TTL
        self._last_full_check: Optional[datetime] = None
        
        logger.info("ConfigurationHealthChecker initialized")
    
    async def perform_health_check(self, include_connectivity: bool = True) -> HealthCheckResult:
        """Perform comprehensive configuration health check.
        
        Args:
            include_connectivity: Whether to include connectivity tests
            
        Returns:
            HealthCheckResult with comprehensive health status
        """
        start_time = time.time()
        
        try:
            logger.info("Starting configuration health check...")
            
            # Initialize result
            result = HealthCheckResult(
                status="healthy",
                timestamp=datetime.now(),
                response_time_ms=0.0,
                checks={},
                errors=[],
                warnings=[],
                metadata={}
            )
            
            # Get current configuration
            config = self.config_manager.config
            
            # 1. Basic configuration validation
            await self._check_basic_configuration(config, result)
            
            # 2. Provider status checks
            await self._check_provider_status(config, result, include_connectivity)
            
            # 3. RAG system health
            await self._check_rag_system_health(config, result, include_connectivity)
            
            # 4. Agent configuration health
            await self._check_agent_health(config, result)
            
            # 5. External services health
            await self._check_external_services_health(config, result, include_connectivity)
            
            # 6. Resource availability
            await self._check_resource_availability(config, result)
            
            # 7. Performance metrics
            await self._check_performance_metrics(config, result)
            
            # Calculate final status
            result.response_time_ms = (time.time() - start_time) * 1000
            result.status = self._calculate_overall_status(result)
            
            # Update metadata
            result.metadata.update({
                'config_environment': config.environment,
                'total_providers': len(config.llm_providers),
                'enabled_providers': len([p for p in config.llm_providers.values() if p.enabled]),
                'total_agents': len(config.agent_configs),
                'enabled_agents': len([a for a in config.agent_configs.values() if a.enabled]),
                'rag_enabled': config.rag_config.enabled if config.rag_config else False,
                'monitoring_enabled': config.monitoring_config.enabled,
                'cache_stats': self.config_manager.get_cache_stats()
            })
            
            self._last_full_check = datetime.now()
            
            logger.info(f"Configuration health check completed: {result.status} ({result.response_time_ms:.1f}ms)")
            
            return result
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"Health check failed: {e}")
            
            return HealthCheckResult(
                status="critical",
                timestamp=datetime.now(),
                response_time_ms=response_time,
                checks={},
                errors=[f"Health check failed: {str(e)}"],
                warnings=[],
                metadata={'error': str(e)}
            )
    
    async def _check_basic_configuration(self, config: SystemConfig, result: HealthCheckResult):
        """Check basic configuration validity."""
        check_start = time.time()
        
        try:
            # Validate configuration
            validation_result = self.validator.validate_system_config(config)
            
            check_time = (time.time() - check_start) * 1000
            
            result.checks['basic_configuration'] = {
                'status': 'healthy' if validation_result.valid else 'critical',
                'response_time_ms': check_time,
                'valid': validation_result.valid,
                'error_count': len(validation_result.errors),
                'warning_count': len(validation_result.warnings),
                'errors': validation_result.errors[:5],  # Limit to first 5 errors
                'warnings': validation_result.warnings[:5]  # Limit to first 5 warnings
            }
            
            # Add errors and warnings to main result
            result.errors.extend(validation_result.errors)
            result.warnings.extend(validation_result.warnings)
            
        except Exception as e:
            check_time = (time.time() - check_start) * 1000
            error_msg = f"Basic configuration check failed: {str(e)}"
            
            result.checks['basic_configuration'] = {
                'status': 'critical',
                'response_time_ms': check_time,
                'error': error_msg
            }
            result.errors.append(error_msg)
    
    async def _check_provider_status(self, config: SystemConfig, result: HealthCheckResult, include_connectivity: bool):
        """Check LLM provider status and connectivity."""
        check_start = time.time()
        provider_checks = {}
        
        try:
            for provider_name, provider_config in config.llm_providers.items():
                provider_start = time.time()
                provider_status = {
                    'enabled': provider_config.enabled,
                    'has_api_key': bool(provider_config.api_key),
                    'model_count': len(provider_config.models),
                    'default_model': provider_config.default_model
                }
                
                if provider_config.enabled and include_connectivity:
                    # Test connectivity
                    try:
                        connection_result = await self.validator.test_provider_connection(provider_config)
                        provider_status.update({
                            'connectivity': 'healthy' if connection_result.success else 'critical',
                            'connection_time_ms': connection_result.response_time_ms,
                            'connection_error': connection_result.error_message
                        })
                        
                        if not connection_result.success:
                            result.errors.append(f"Provider {provider_name} connection failed: {connection_result.error_message}")
                    
                    except Exception as e:
                        provider_status['connectivity'] = 'critical'
                        provider_status['connection_error'] = str(e)
                        result.errors.append(f"Provider {provider_name} connectivity check failed: {str(e)}")
                
                provider_status['check_time_ms'] = (time.time() - provider_start) * 1000
                provider_checks[provider_name] = provider_status
            
            total_check_time = (time.time() - check_start) * 1000
            
            # Calculate overall provider status
            enabled_providers = [p for p in config.llm_providers.values() if p.enabled]
            healthy_providers = 0
            
            for provider_name, status in provider_checks.items():
                if status['enabled']:
                    if include_connectivity:
                        if status.get('connectivity') == 'healthy':
                            healthy_providers += 1
                    else:
                        if status['has_api_key']:
                            healthy_providers += 1
            
            overall_status = 'healthy'
            if len(enabled_providers) == 0:
                overall_status = 'critical'
            elif healthy_providers == 0:
                overall_status = 'critical'
            elif healthy_providers < len(enabled_providers):
                overall_status = 'warning'
            
            result.checks['providers'] = {
                'status': overall_status,
                'response_time_ms': total_check_time,
                'total_providers': len(config.llm_providers),
                'enabled_providers': len(enabled_providers),
                'healthy_providers': healthy_providers,
                'providers': provider_checks
            }
            
        except Exception as e:
            check_time = (time.time() - check_start) * 1000
            error_msg = f"Provider status check failed: {str(e)}"
            
            result.checks['providers'] = {
                'status': 'critical',
                'response_time_ms': check_time,
                'error': error_msg
            }
            result.errors.append(error_msg)
    
    async def _check_rag_system_health(self, config: SystemConfig, result: HealthCheckResult, include_connectivity: bool):
        """Check RAG system health."""
        check_start = time.time()
        
        try:
            if not config.rag_config or not config.rag_config.enabled:
                result.checks['rag_system'] = {
                    'status': 'healthy',
                    'response_time_ms': (time.time() - check_start) * 1000,
                    'enabled': False,
                    'message': 'RAG system is disabled'
                }
                return
            
            rag_config = config.rag_config
            rag_status = {
                'enabled': True,
                'embedding_provider': rag_config.embedding_config.provider,
                'embedding_model': rag_config.embedding_config.model_name,
                'vector_store_provider': rag_config.vector_store_config.provider,
                'chunk_size': rag_config.chunk_size,
                'caching_enabled': rag_config.enable_caching
            }
            
            # Check embedding-vector store compatibility
            compatibility_result = self.validator.validate_embedding_vector_store_compatibility(
                rag_config.embedding_config, rag_config.vector_store_config
            )
            
            rag_status['compatibility'] = 'healthy' if compatibility_result.valid else 'critical'
            if not compatibility_result.valid:
                rag_status['compatibility_errors'] = compatibility_result.errors
                result.errors.extend([f"RAG compatibility: {error}" for error in compatibility_result.errors])
            
            # Test vector store connectivity if requested
            if include_connectivity:
                try:
                    connection_result = await self.validator.test_vector_store_connection(rag_config.vector_store_config)
                    rag_status['vector_store_connectivity'] = 'healthy' if connection_result.success else 'critical'
                    rag_status['vector_store_connection_time_ms'] = connection_result.response_time_ms
                    
                    if not connection_result.success:
                        rag_status['vector_store_error'] = connection_result.error_message
                        result.errors.append(f"Vector store connection failed: {connection_result.error_message}")
                
                except Exception as e:
                    rag_status['vector_store_connectivity'] = 'critical'
                    rag_status['vector_store_error'] = str(e)
                    result.errors.append(f"Vector store connectivity check failed: {str(e)}")
            
            # Determine overall RAG status
            overall_status = 'healthy'
            if rag_status.get('compatibility') == 'critical':
                overall_status = 'critical'
            elif include_connectivity and rag_status.get('vector_store_connectivity') == 'critical':
                overall_status = 'critical'
            
            result.checks['rag_system'] = {
                'status': overall_status,
                'response_time_ms': (time.time() - check_start) * 1000,
                **rag_status
            }
            
        except Exception as e:
            check_time = (time.time() - check_start) * 1000
            error_msg = f"RAG system health check failed: {str(e)}"
            
            result.checks['rag_system'] = {
                'status': 'critical',
                'response_time_ms': check_time,
                'error': error_msg
            }
            result.errors.append(error_msg)
    
    async def _check_agent_health(self, config: SystemConfig, result: HealthCheckResult):
        """Check agent configuration health."""
        check_start = time.time()
        
        try:
            agent_checks = {}
            required_agents = {'planner_agent', 'code_generator_agent', 'renderer_agent'}
            
            for agent_name, agent_config in config.agent_configs.items():
                agent_status = {
                    'enabled': agent_config.enabled,
                    'timeout_seconds': agent_config.timeout_seconds,
                    'max_retries': agent_config.max_retries,
                    'has_llm_config': bool(agent_config.llm_config),
                    'tool_count': len(agent_config.tools)
                }
                
                # Check model configurations
                model_fields = ['planner_model', 'scene_model', 'helper_model']
                model_issues = []
                
                for field in model_fields:
                    model_name = getattr(agent_config, field, None)
                    if model_name and '/' in model_name:
                        provider, model = model_name.split('/', 1)
                        if provider not in config.llm_providers:
                            model_issues.append(f"{field} references unknown provider: {provider}")
                        elif not config.llm_providers[provider].enabled:
                            model_issues.append(f"{field} references disabled provider: {provider}")
                
                agent_status['model_issues'] = model_issues
                agent_status['status'] = 'critical' if model_issues else 'healthy'
                
                agent_checks[agent_name] = agent_status
                
                # Add model issues to main result
                for issue in model_issues:
                    result.errors.append(f"Agent {agent_name}: {issue}")
            
            # Check for missing required agents
            missing_agents = required_agents - set(config.agent_configs.keys())
            if missing_agents:
                result.errors.extend([f"Required agent missing: {agent}" for agent in missing_agents])
            
            # Calculate overall agent status
            enabled_agents = [a for a in config.agent_configs.values() if a.enabled]
            healthy_agents = len([a for name, a in agent_checks.items() if a['status'] == 'healthy' and config.agent_configs[name].enabled])
            
            overall_status = 'healthy'
            if missing_agents:
                overall_status = 'critical'
            elif len(enabled_agents) == 0:
                overall_status = 'warning'
            elif healthy_agents < len(enabled_agents):
                overall_status = 'warning'
            
            result.checks['agents'] = {
                'status': overall_status,
                'response_time_ms': (time.time() - check_start) * 1000,
                'total_agents': len(config.agent_configs),
                'enabled_agents': len(enabled_agents),
                'healthy_agents': healthy_agents,
                'missing_required': list(missing_agents),
                'agents': agent_checks
            }
            
        except Exception as e:
            check_time = (time.time() - check_start) * 1000
            error_msg = f"Agent health check failed: {str(e)}"
            
            result.checks['agents'] = {
                'status': 'critical',
                'response_time_ms': check_time,
                'error': error_msg
            }
            result.errors.append(error_msg)
    
    async def _check_external_services_health(self, config: SystemConfig, result: HealthCheckResult, include_connectivity: bool):
        """Check external services health."""
        check_start = time.time()
        services_status = {}
        
        try:
            # Check Langfuse
            if config.monitoring_config.langfuse_config and config.monitoring_config.langfuse_config.enabled:
                langfuse_config = config.monitoring_config.langfuse_config
                langfuse_status = {
                    'enabled': True,
                    'has_secret_key': bool(langfuse_config.secret_key),
                    'has_public_key': bool(langfuse_config.public_key),
                    'host': langfuse_config.host
                }
                
                if not langfuse_config.secret_key or not langfuse_config.public_key:
                    langfuse_status['status'] = 'critical'
                    result.errors.append("Langfuse enabled but missing required keys")
                else:
                    langfuse_status['status'] = 'healthy'
                
                services_status['langfuse'] = langfuse_status
            
            # Check MCP servers
            mcp_status = {
                'total_servers': len(config.mcp_servers),
                'enabled_servers': len([s for s in config.mcp_servers.values() if not s.disabled]),
                'servers': {}
            }
            
            for server_name, server_config in config.mcp_servers.items():
                server_status = {
                    'enabled': not server_config.disabled,
                    'has_command': bool(server_config.command),
                    'arg_count': len(server_config.args),
                    'env_var_count': len(server_config.env),
                    'auto_approve_count': len(server_config.auto_approve)
                }
                
                if not server_config.disabled and not server_config.command:
                    server_status['status'] = 'critical'
                    result.errors.append(f"MCP server {server_name} enabled but missing command")
                else:
                    server_status['status'] = 'healthy'
                
                mcp_status['servers'][server_name] = server_status
            
            services_status['mcp'] = mcp_status
            
            # Check Context7
            if config.context7_config.enabled:
                context7_status = {
                    'enabled': True,
                    'default_tokens': config.context7_config.default_tokens,
                    'timeout_seconds': config.context7_config.timeout_seconds,
                    'cache_responses': config.context7_config.cache_responses
                }
                
                if config.context7_config.default_tokens <= 0:
                    context7_status['status'] = 'critical'
                    result.errors.append("Context7 enabled but invalid default_tokens")
                else:
                    context7_status['status'] = 'healthy'
                
                services_status['context7'] = context7_status
            
            # Check Docling
            if config.docling_config.enabled:
                docling_status = {
                    'enabled': True,
                    'max_file_size_mb': config.docling_config.max_file_size_mb,
                    'timeout_seconds': config.docling_config.timeout_seconds,
                    'supported_formats': config.docling_config.supported_formats
                }
                
                if config.docling_config.max_file_size_mb <= 0:
                    docling_status['status'] = 'critical'
                    result.errors.append("Docling enabled but invalid max_file_size_mb")
                else:
                    docling_status['status'] = 'healthy'
                
                services_status['docling'] = docling_status
            
            # Calculate overall external services status
            service_statuses = [s.get('status', 'healthy') for s in services_status.values() if isinstance(s, dict) and 'status' in s]
            overall_status = 'healthy'
            if 'critical' in service_statuses:
                overall_status = 'critical'
            elif 'warning' in service_statuses:
                overall_status = 'warning'
            
            result.checks['external_services'] = {
                'status': overall_status,
                'response_time_ms': (time.time() - check_start) * 1000,
                'services': services_status
            }
            
        except Exception as e:
            check_time = (time.time() - check_start) * 1000
            error_msg = f"External services health check failed: {str(e)}"
            
            result.checks['external_services'] = {
                'status': 'critical',
                'response_time_ms': check_time,
                'error': error_msg
            }
            result.errors.append(error_msg)
    
    async def _check_resource_availability(self, config: SystemConfig, result: HealthCheckResult):
        """Check resource availability."""
        check_start = time.time()
        
        try:
            resource_status = {}
            
            # Check output directory
            output_dir = Path(config.workflow_config.output_dir)
            output_status = {
                'path': str(output_dir),
                'exists': output_dir.exists(),
                'writable': False,
                'free_space_gb': 0
            }
            
            if output_dir.exists():
                try:
                    # Test write access
                    test_file = output_dir / '.health_check_test'
                    test_file.write_text('test')
                    test_file.unlink()
                    output_status['writable'] = True
                except Exception:
                    output_status['writable'] = False
                    result.warnings.append(f"Output directory not writable: {output_dir}")
                
                # Check disk space
                try:
                    import shutil
                    total, used, free = shutil.disk_usage(output_dir)
                    free_gb = free // (1024**3)
                    output_status['free_space_gb'] = free_gb
                    
                    if free_gb < 1:
                        result.warnings.append(f"Low disk space in output directory: {free_gb}GB free")
                except Exception:
                    pass
            else:
                result.warnings.append(f"Output directory does not exist: {output_dir}")
            
            resource_status['output_directory'] = output_status
            
            # Check TTS model files
            tts_status = {}
            if config.kokoro_model_path:
                model_path = Path(config.kokoro_model_path)
                tts_status['model_file'] = {
                    'path': str(model_path),
                    'exists': model_path.exists()
                }
                if not model_path.exists():
                    result.errors.append(f"TTS model file not found: {config.kokoro_model_path}")
            
            if config.kokoro_voices_path:
                voices_path = Path(config.kokoro_voices_path)
                tts_status['voices_file'] = {
                    'path': str(voices_path),
                    'exists': voices_path.exists()
                }
                if not voices_path.exists():
                    result.errors.append(f"TTS voices file not found: {config.kokoro_voices_path}")
            
            if tts_status:
                resource_status['tts'] = tts_status
            
            # Check vector store persistence
            if config.rag_config and config.rag_config.enabled:
                vector_config = config.rag_config.vector_store_config
                if vector_config.provider == 'chroma':
                    persist_dir = vector_config.connection_params.get('persist_directory', './chroma_db')
                    persist_path = Path(persist_dir)
                    
                    vector_storage_status = {
                        'provider': 'chroma',
                        'persist_directory': str(persist_path),
                        'exists': persist_path.exists(),
                        'writable': False
                    }
                    
                    if persist_path.exists():
                        try:
                            test_file = persist_path / '.health_check_test'
                            test_file.write_text('test')
                            test_file.unlink()
                            vector_storage_status['writable'] = True
                        except Exception:
                            result.warnings.append(f"Vector store directory not writable: {persist_path}")
                    
                    resource_status['vector_storage'] = vector_storage_status
            
            # Calculate overall resource status
            overall_status = 'healthy'
            if not output_status['exists'] or not output_status['writable']:
                overall_status = 'warning'
            
            result.checks['resources'] = {
                'status': overall_status,
                'response_time_ms': (time.time() - check_start) * 1000,
                **resource_status
            }
            
        except Exception as e:
            check_time = (time.time() - check_start) * 1000
            error_msg = f"Resource availability check failed: {str(e)}"
            
            result.checks['resources'] = {
                'status': 'critical',
                'response_time_ms': check_time,
                'error': error_msg
            }
            result.errors.append(error_msg)
    
    async def _check_performance_metrics(self, config: SystemConfig, result: HealthCheckResult):
        """Check performance-related metrics."""
        check_start = time.time()
        
        try:
            perf_metrics = {}
            
            # Configuration cache metrics
            cache_stats = self.config_manager.get_cache_stats()
            perf_metrics['configuration_cache'] = cache_stats
            
            # Concurrency settings analysis
            workflow_config = config.workflow_config
            total_concurrency = (workflow_config.max_scene_concurrency + 
                               workflow_config.max_topic_concurrency + 
                               workflow_config.max_concurrent_renders)
            
            perf_metrics['concurrency'] = {
                'max_scene_concurrency': workflow_config.max_scene_concurrency,
                'max_topic_concurrency': workflow_config.max_topic_concurrency,
                'max_concurrent_renders': workflow_config.max_concurrent_renders,
                'total_concurrency': total_concurrency
            }
            
            if total_concurrency > 20:
                result.warnings.append(f"High total concurrency ({total_concurrency}) may impact performance")
            
            # Timeout analysis
            timeout_metrics = {
                'workflow_timeout_seconds': workflow_config.workflow_timeout_seconds,
                'checkpoint_interval': workflow_config.checkpoint_interval,
                'agent_timeouts': {}
            }
            
            for agent_name, agent_config in config.agent_configs.items():
                timeout_metrics['agent_timeouts'][agent_name] = agent_config.timeout_seconds
                if agent_config.timeout_seconds > 600:  # 10 minutes
                    result.warnings.append(f"Agent {agent_name} has long timeout ({agent_config.timeout_seconds}s)")
            
            perf_metrics['timeouts'] = timeout_metrics
            
            # RAG performance settings
            if config.rag_config and config.rag_config.enabled:
                rag_perf = {
                    'chunk_size': config.rag_config.chunk_size,
                    'chunk_overlap': config.rag_config.chunk_overlap,
                    'embedding_batch_size': config.rag_config.embedding_config.batch_size,
                    'max_cache_size': config.rag_config.max_cache_size,
                    'caching_enabled': config.rag_config.enable_caching
                }
                
                if config.rag_config.embedding_config.batch_size > 1000:
                    result.warnings.append(f"Large embedding batch size ({config.rag_config.embedding_config.batch_size}) may cause memory issues")
                
                if config.rag_config.max_cache_size > 10000:
                    result.warnings.append(f"Large RAG cache size ({config.rag_config.max_cache_size}) may cause memory issues")
                
                perf_metrics['rag_performance'] = rag_perf
            
            result.checks['performance'] = {
                'status': 'healthy',  # Performance checks generate warnings, not errors
                'response_time_ms': (time.time() - check_start) * 1000,
                'metrics': perf_metrics
            }
            
        except Exception as e:
            check_time = (time.time() - check_start) * 1000
            error_msg = f"Performance metrics check failed: {str(e)}"
            
            result.checks['performance'] = {
                'status': 'warning',
                'response_time_ms': check_time,
                'error': error_msg
            }
            result.warnings.append(error_msg)
    
    def _calculate_overall_status(self, result: HealthCheckResult) -> str:
        """Calculate overall health status based on individual checks."""
        if result.errors:
            return "critical"
        
        # Check individual component statuses
        critical_checks = [check for check in result.checks.values() 
                          if isinstance(check, dict) and check.get('status') == 'critical']
        
        if critical_checks:
            return "critical"
        
        warning_checks = [check for check in result.checks.values() 
                         if isinstance(check, dict) and check.get('status') == 'warning']
        
        if warning_checks or result.warnings:
            return "warning"
        
        return "healthy"
    
    def get_quick_status(self) -> Dict[str, Any]:
        """Get quick configuration status without full health check.
        
        Returns:
            Dictionary with basic status information
        """
        try:
            config = self.config_manager.config
            
            return {
                'status': 'healthy',  # Quick check assumes healthy unless obvious issues
                'timestamp': datetime.now().isoformat(),
                'environment': config.environment,
                'providers': {
                    'total': len(config.llm_providers),
                    'enabled': len([p for p in config.llm_providers.values() if p.enabled])
                },
                'agents': {
                    'total': len(config.agent_configs),
                    'enabled': len([a for a in config.agent_configs.values() if a.enabled])
                },
                'rag_enabled': config.rag_config.enabled if config.rag_config else False,
                'monitoring_enabled': config.monitoring_config.enabled,
                'last_full_check': self._last_full_check.isoformat() if self._last_full_check else None
            }
            
        except Exception as e:
            return {
                'status': 'critical',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def get_component_health(self, component_name: str) -> Optional[ComponentHealth]:
        """Get health status for a specific component.
        
        Args:
            component_name: Name of the component to check
            
        Returns:
            ComponentHealth object or None if not found
        """
        return self._health_cache.get(component_name)
    
    def set_cache_ttl(self, ttl_seconds: int):
        """Set cache TTL for health checks.
        
        Args:
            ttl_seconds: Time to live in seconds
        """
        if ttl_seconds < 0:
            raise ValueError("Cache TTL must be non-negative")
        
        self._cache_ttl = ttl_seconds
        logger.info(f"Health check cache TTL set to {ttl_seconds} seconds")