"""
Startup Configuration Validation Service.

This module provides comprehensive configuration validation that runs at system startup,
ensuring all components are properly configured and ready for operation.
"""

import asyncio
import logging
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from .models import SystemConfig, ValidationResult
from .manager import ConfigurationManager
from .validation import ConfigValidationService
from .health_check import ConfigurationHealthChecker
from .migration import ConfigurationMigrator


logger = logging.getLogger(__name__)


class StartupValidationService:
    """Service for comprehensive configuration validation at system startup.
    
    This service orchestrates all validation components to ensure the system
    is properly configured before allowing it to start.
    """
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """Initialize the startup validation service.
        
        Args:
            config_manager: Configuration manager instance (optional)
        """
        self.config_manager = config_manager or ConfigurationManager()
        self.validator = ConfigValidationService()
        self.health_checker = ConfigurationHealthChecker(self.config_manager)
        self.migrator = ConfigurationMigrator()
        
        # Validation settings
        self.strict_mode = False  # If True, warnings become errors
        self.fail_on_connectivity_issues = False  # If True, connectivity failures stop startup
        
        logger.info("StartupValidationService initialized")
    
    async def validate_startup_configuration(self, strict_mode: bool = False) -> ValidationResult:
        """Perform comprehensive startup configuration validation.
        
        Args:
            strict_mode: If True, treat warnings as errors
            
        Returns:
            ValidationResult with comprehensive validation status
        """
        self.strict_mode = strict_mode
        
        logger.info("=" * 60)
        logger.info("STARTING SYSTEM CONFIGURATION VALIDATION")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        overall_result = ValidationResult(valid=True)
        
        try:
            # 1. Load and validate basic configuration
            logger.info("Step 1: Loading system configuration...")
            config_result = await self._validate_configuration_loading()
            self._merge_results(overall_result, config_result)
            
            if not config_result.valid:
                logger.error("❌ Configuration loading failed - cannot continue")
                return overall_result
            
            config = self.config_manager.config
            
            # 2. Check for configuration migrations
            logger.info("Step 2: Checking for configuration migrations...")
            migration_result = await self._check_configuration_migrations(config)
            self._merge_results(overall_result, migration_result)
            
            # 3. Comprehensive configuration validation
            logger.info("Step 3: Performing comprehensive configuration validation...")
            validation_result = await self.validator.validate_startup_configuration(config)
            self._merge_results(overall_result, validation_result)
            
            # 4. Component compatibility validation
            logger.info("Step 4: Validating component compatibility...")
            compatibility_result = await self._validate_component_compatibility(config)
            self._merge_results(overall_result, compatibility_result)
            
            # 5. External service connectivity (if not in strict mode)
            if not self.fail_on_connectivity_issues:
                logger.info("Step 5: Testing external service connectivity...")
                connectivity_result = await self._test_external_connectivity(config)
                self._merge_results(overall_result, connectivity_result, treat_errors_as_warnings=True)
            else:
                logger.info("Step 5: Testing external service connectivity (strict mode)...")
                connectivity_result = await self._test_external_connectivity(config)
                self._merge_results(overall_result, connectivity_result)
            
            # 6. Resource availability validation
            logger.info("Step 6: Validating resource availability...")
            resource_result = await self._validate_resource_availability(config)
            self._merge_results(overall_result, resource_result)
            
            # 7. Security configuration validation
            logger.info("Step 7: Validating security configuration...")
            security_result = await self._validate_security_configuration(config)
            self._merge_results(overall_result, security_result)
            
            # 8. Performance configuration validation
            logger.info("Step 8: Validating performance configuration...")
            performance_result = await self._validate_performance_configuration(config)
            self._merge_results(overall_result, performance_result)
            
            # Apply strict mode rules
            if self.strict_mode and overall_result.warnings:
                logger.warning("Strict mode enabled - converting warnings to errors")
                overall_result.errors.extend([f"Warning (strict mode): {w}" for w in overall_result.warnings])
                overall_result.warnings.clear()
                overall_result.valid = False
            
            # Final status
            duration = (datetime.now() - start_time).total_seconds()
            
            if overall_result.valid:
                logger.info("=" * 60)
                logger.info("✅ CONFIGURATION VALIDATION COMPLETED SUCCESSFULLY")
                logger.info(f"   Duration: {duration:.2f} seconds")
                logger.info(f"   Warnings: {len(overall_result.warnings)}")
                logger.info("=" * 60)
            else:
                logger.error("=" * 60)
                logger.error("❌ CONFIGURATION VALIDATION FAILED")
                logger.error(f"   Duration: {duration:.2f} seconds")
                logger.error(f"   Errors: {len(overall_result.errors)}")
                logger.error(f"   Warnings: {len(overall_result.warnings)}")
                logger.error("=" * 60)
                
                # Log first few errors for immediate visibility
                for i, error in enumerate(overall_result.errors[:5]):
                    logger.error(f"   Error {i+1}: {error}")
                
                if len(overall_result.errors) > 5:
                    logger.error(f"   ... and {len(overall_result.errors) - 5} more errors")
            
        except Exception as e:
            overall_result.add_error(f"Startup validation failed with exception: {str(e)}")
            logger.error(f"Startup validation exception: {e}")
        
        return overall_result
    
    async def _validate_configuration_loading(self) -> ValidationResult:
        """Validate that configuration can be loaded properly."""
        result = ValidationResult(valid=True)
        
        try:
            # Test configuration loading
            config = self.config_manager.config
            
            if not config:
                result.add_error("Failed to load system configuration")
                return result
            
            # Basic structure validation
            if not config.llm_providers:
                result.add_error("No LLM providers configured")
            
            if not config.agent_configs:
                result.add_error("No agents configured")
            
            # Environment validation
            if config.environment not in ['development', 'staging', 'production']:
                result.add_error(f"Invalid environment: {config.environment}")
            
            # Default provider validation
            if config.default_llm_provider not in config.llm_providers:
                result.add_error(f"Default LLM provider '{config.default_llm_provider}' not found in configured providers")
            
            logger.info(f"✅ Configuration loaded successfully (environment: {config.environment})")
            
        except Exception as e:
            result.add_error(f"Configuration loading failed: {str(e)}")
        
        return result
    
    async def _check_configuration_migrations(self, config: SystemConfig) -> ValidationResult:
        """Check if configuration needs migration and handle it."""
        result = ValidationResult(valid=True)
        
        try:
            # Convert config to dict for migration check
            config_dict = config.model_dump()
            
            if self.migrator.needs_migration(config_dict):
                current_version = self.migrator.detect_config_version(config_dict)
                target_version = self.migrator.CURRENT_VERSION
                
                logger.warning(f"Configuration migration needed: {current_version} -> {target_version}")
                
                # Perform migration
                migration_result = self.migrator.migrate_configuration(config_dict, target_version)
                
                if migration_result.success:
                    logger.info(f"✅ Configuration migrated successfully")
                    for change in migration_result.changes_made:
                        result.add_warning(f"Migration: {change}")
                    
                    # Save migration record
                    self.migrator.save_migration_record(migration_result)
                else:
                    result.add_error(f"Configuration migration failed: {'; '.join(migration_result.errors)}")
                    for error in migration_result.errors:
                        logger.error(f"Migration error: {error}")
            else:
                logger.info("✅ Configuration is up to date")
            
        except Exception as e:
            result.add_error(f"Configuration migration check failed: {str(e)}")
        
        return result
    
    async def _validate_component_compatibility(self, config: SystemConfig) -> ValidationResult:
        """Validate compatibility between system components."""
        result = ValidationResult(valid=True)
        
        try:
            # Agent-provider compatibility
            for agent_name, agent_config in config.agent_configs.items():
                if not agent_config.enabled:
                    continue
                
                # Check model references
                model_fields = ['planner_model', 'scene_model', 'helper_model']
                for field in model_fields:
                    model_name = getattr(agent_config, field, None)
                    if model_name and '/' in model_name:
                        provider, model = model_name.split('/', 1)
                        
                        if provider not in config.llm_providers:
                            result.add_error(f"Agent '{agent_name}' {field} references unknown provider: {provider}")
                        else:
                            provider_config = config.llm_providers[provider]
                            if not provider_config.enabled:
                                result.add_error(f"Agent '{agent_name}' {field} references disabled provider: {provider}")
                            elif model not in provider_config.models:
                                result.add_warning(f"Agent '{agent_name}' {field} model '{model}' not in provider's model list")
            
            # RAG system compatibility
            if config.rag_config and config.rag_config.enabled:
                embedding_config = config.rag_config.embedding_config
                vector_config = config.rag_config.vector_store_config
                
                compatibility_result = self.validator.validate_embedding_vector_store_compatibility(
                    embedding_config, vector_config
                )
                
                if not compatibility_result.valid:
                    for error in compatibility_result.errors:
                        result.add_error(f"RAG compatibility: {error}")
                
                for warning in compatibility_result.warnings:
                    result.add_warning(f"RAG compatibility: {warning}")
            
            # Monitoring-provider compatibility
            if config.monitoring_config.enabled and config.monitoring_config.langfuse_config:
                langfuse_config = config.monitoring_config.langfuse_config
                if langfuse_config.enabled:
                    supported_providers = ['openai', 'anthropic', 'gemini']
                    enabled_supported = [p for p in supported_providers 
                                       if p in config.llm_providers and config.llm_providers[p].enabled]
                    if not enabled_supported:
                        result.add_warning("Langfuse monitoring enabled but no supported providers are enabled")
            
            if result.valid and not result.warnings:
                logger.info("✅ Component compatibility validation passed")
            
        except Exception as e:
            result.add_error(f"Component compatibility validation failed: {str(e)}")
        
        return result
    
    async def _test_external_connectivity(self, config: SystemConfig) -> ValidationResult:
        """Test connectivity to external services."""
        result = ValidationResult(valid=True)
        
        try:
            connectivity_tasks = []
            
            # Test LLM provider connections
            for provider_name, provider_config in config.llm_providers.items():
                if provider_config.enabled and provider_config.api_key:
                    connectivity_tasks.append(
                        self._test_single_provider_connectivity(provider_name, provider_config, result)
                    )
            
            # Test vector store connection
            if config.rag_config and config.rag_config.enabled:
                connectivity_tasks.append(
                    self._test_vector_store_connectivity(config.rag_config.vector_store_config, result)
                )
            
            # Run all connectivity tests
            if connectivity_tasks:
                await asyncio.gather(*connectivity_tasks, return_exceptions=True)
            
            # Count successful connections
            successful_providers = len([p for p in config.llm_providers.values() 
                                      if p.enabled and p.api_key])
            
            if successful_providers == 0:
                result.add_warning("No LLM providers with API keys are enabled")
            
        except Exception as e:
            result.add_error(f"External connectivity testing failed: {str(e)}")
        
        return result
    
    async def _test_single_provider_connectivity(self, provider_name: str, provider_config, result: ValidationResult):
        """Test connectivity to a single provider."""
        try:
            connection_result = await self.validator.test_provider_connection(provider_config)
            
            if connection_result.success:
                logger.info(f"✅ {provider_name} connectivity test passed ({connection_result.response_time_ms:.1f}ms)")
            else:
                error_msg = f"{provider_name} connectivity test failed: {connection_result.error_message}"
                result.add_error(error_msg)
                logger.error(f"❌ {error_msg}")
                
        except Exception as e:
            error_msg = f"{provider_name} connectivity test error: {str(e)}"
            result.add_error(error_msg)
            logger.error(f"❌ {error_msg}")
    
    async def _test_vector_store_connectivity(self, vector_config, result: ValidationResult):
        """Test connectivity to vector store."""
        try:
            connection_result = await self.validator.test_vector_store_connection(vector_config)
            
            if connection_result.success:
                logger.info(f"✅ Vector store ({vector_config.provider}) connectivity test passed ({connection_result.response_time_ms:.1f}ms)")
            else:
                error_msg = f"Vector store ({vector_config.provider}) connectivity test failed: {connection_result.error_message}"
                result.add_error(error_msg)
                logger.error(f"❌ {error_msg}")
                
        except Exception as e:
            error_msg = f"Vector store connectivity test error: {str(e)}"
            result.add_error(error_msg)
            logger.error(f"❌ {error_msg}")
    
    async def _validate_resource_availability(self, config: SystemConfig) -> ValidationResult:
        """Validate that required resources are available."""
        result = ValidationResult(valid=True)
        
        try:
            # Check output directory
            output_dir = Path(config.workflow_config.output_dir)
            if not output_dir.exists():
                try:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    result.add_warning(f"Created output directory: {output_dir}")
                    logger.info(f"✅ Created output directory: {output_dir}")
                except Exception as e:
                    result.add_error(f"Cannot create output directory '{config.workflow_config.output_dir}': {e}")
            else:
                logger.info(f"✅ Output directory exists: {output_dir}")
            
            # Check disk space
            try:
                import shutil
                total, used, free = shutil.disk_usage(output_dir)
                free_gb = free // (1024**3)
                
                if free_gb < 1:
                    result.add_warning(f"Low disk space in output directory: {free_gb}GB free")
                else:
                    logger.info(f"✅ Disk space available: {free_gb}GB free")
            except Exception as e:
                result.add_warning(f"Could not check disk space: {e}")
            
            # Check TTS model files if configured
            if config.kokoro_model_path:
                model_path = Path(config.kokoro_model_path)
                if model_path.exists():
                    logger.info(f"✅ TTS model file found: {config.kokoro_model_path}")
                else:
                    result.add_error(f"TTS model file not found: {config.kokoro_model_path}")
            
            if config.kokoro_voices_path:
                voices_path = Path(config.kokoro_voices_path)
                if voices_path.exists():
                    logger.info(f"✅ TTS voices file found: {config.kokoro_voices_path}")
                else:
                    result.add_error(f"TTS voices file not found: {config.kokoro_voices_path}")
            
            # Check vector store persistence directory
            if config.rag_config and config.rag_config.enabled:
                vector_config = config.rag_config.vector_store_config
                if vector_config.provider == 'chroma':
                    persist_dir = vector_config.connection_params.get('persist_directory', './chroma_db')
                    persist_path = Path(persist_dir)
                    
                    if not persist_path.exists():
                        try:
                            persist_path.mkdir(parents=True, exist_ok=True)
                            result.add_warning(f"Created ChromaDB persist directory: {persist_path}")
                            logger.info(f"✅ Created ChromaDB persist directory: {persist_path}")
                        except Exception as e:
                            result.add_error(f"Cannot create ChromaDB persist directory '{persist_dir}': {e}")
                    else:
                        logger.info(f"✅ ChromaDB persist directory exists: {persist_path}")
            
        except Exception as e:
            result.add_error(f"Resource availability validation failed: {str(e)}")
        
        return result
    
    async def _validate_security_configuration(self, config: SystemConfig) -> ValidationResult:
        """Validate security-related configuration."""
        result = ValidationResult(valid=True)
        
        try:
            security_issues = []
            
            # Check for hardcoded API keys
            for provider_name, provider_config in config.llm_providers.items():
                if provider_config.api_key and not provider_config.api_key.startswith('$'):
                    if len(provider_config.api_key) > 10:  # Likely a real key
                        security_issues.append(f"API key for {provider_name} appears to be hardcoded")
            
            # Production environment checks
            if config.environment == 'production':
                if config.debug:
                    result.add_error("Debug mode should be disabled in production")
                
                if config.monitoring_config.log_level == 'DEBUG':
                    result.add_warning("Debug logging enabled in production")
            
            # Check for secure URLs
            for provider_name, provider_config in config.llm_providers.items():
                if provider_config.base_url and not provider_config.base_url.startswith('https://'):
                    security_issues.append(f"Provider {provider_name} using non-HTTPS URL")
            
            # Langfuse security
            if config.monitoring_config.langfuse_config and config.monitoring_config.langfuse_config.enabled:
                langfuse_config = config.monitoring_config.langfuse_config
                if langfuse_config.host and not langfuse_config.host.startswith('https://'):
                    security_issues.append("Langfuse host using non-HTTPS URL")
            
            # Report security issues as warnings (not errors unless critical)
            for issue in security_issues:
                result.add_warning(f"Security: {issue}")
            
            if not security_issues:
                logger.info("✅ Security configuration validation passed")
            else:
                logger.warning(f"⚠️  Found {len(security_issues)} security considerations")
            
        except Exception as e:
            result.add_error(f"Security configuration validation failed: {str(e)}")
        
        return result
    
    async def _validate_performance_configuration(self, config: SystemConfig) -> ValidationResult:
        """Validate performance-related configuration."""
        result = ValidationResult(valid=True)
        
        try:
            performance_warnings = []
            
            # Check concurrency settings
            workflow_config = config.workflow_config
            total_concurrency = (workflow_config.max_scene_concurrency + 
                               workflow_config.max_topic_concurrency + 
                               workflow_config.max_concurrent_renders)
            
            if total_concurrency > 20:
                performance_warnings.append(f"High total concurrency ({total_concurrency}) may impact performance")
            
            # Check timeout settings
            if workflow_config.workflow_timeout_seconds > 7200:  # 2 hours
                performance_warnings.append(f"Very long workflow timeout ({workflow_config.workflow_timeout_seconds}s)")
            
            # Check agent timeouts
            for agent_name, agent_config in config.agent_configs.items():
                if agent_config.timeout_seconds > 600:  # 10 minutes
                    performance_warnings.append(f"Agent '{agent_name}' has long timeout ({agent_config.timeout_seconds}s)")
            
            # Check RAG performance settings
            if config.rag_config and config.rag_config.enabled:
                rag_config = config.rag_config
                
                if rag_config.embedding_config.batch_size > 1000:
                    performance_warnings.append(f"Large embedding batch size ({rag_config.embedding_config.batch_size})")
                
                if rag_config.max_cache_size > 10000:
                    performance_warnings.append(f"Large RAG cache size ({rag_config.max_cache_size})")
            
            # Report performance warnings
            for warning in performance_warnings:
                result.add_warning(f"Performance: {warning}")
            
            if not performance_warnings:
                logger.info("✅ Performance configuration validation passed")
            else:
                logger.warning(f"⚠️  Found {len(performance_warnings)} performance considerations")
            
        except Exception as e:
            result.add_error(f"Performance configuration validation failed: {str(e)}")
        
        return result
    
    def _merge_results(self, main_result: ValidationResult, sub_result: ValidationResult, treat_errors_as_warnings: bool = False):
        """Merge a sub-result into the main result.
        
        Args:
            main_result: Main validation result to merge into
            sub_result: Sub-result to merge
            treat_errors_as_warnings: If True, treat sub-result errors as warnings
        """
        if treat_errors_as_warnings:
            # Convert errors to warnings
            main_result.warnings.extend([f"Connectivity: {error}" for error in sub_result.errors])
            main_result.warnings.extend(sub_result.warnings)
        else:
            # Normal merge
            main_result.errors.extend(sub_result.errors)
            main_result.warnings.extend(sub_result.warnings)
            
            if not sub_result.valid:
                main_result.valid = False
    
    def set_strict_mode(self, enabled: bool):
        """Enable or disable strict validation mode.
        
        Args:
            enabled: Whether to enable strict mode
        """
        self.strict_mode = enabled
        logger.info(f"Strict validation mode {'enabled' if enabled else 'disabled'}")
    
    def set_fail_on_connectivity_issues(self, enabled: bool):
        """Set whether to fail startup on connectivity issues.
        
        Args:
            enabled: Whether to fail on connectivity issues
        """
        self.fail_on_connectivity_issues = enabled
        logger.info(f"Fail on connectivity issues {'enabled' if enabled else 'disabled'}")
    
    async def quick_health_check(self) -> Dict[str, Any]:
        """Perform a quick health check without full validation.
        
        Returns:
            Dictionary with basic health status
        """
        try:
            return self.health_checker.get_quick_status()
        except Exception as e:
            return {
                'status': 'critical',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def validate_and_exit_on_failure(self, strict_mode: bool = False):
        """Validate configuration and exit if validation fails.
        
        This is a convenience method for startup scripts that should
        exit if configuration validation fails.
        
        Args:
            strict_mode: Whether to use strict validation mode
        """
        async def _validate():
            result = await self.validate_startup_configuration(strict_mode)
            
            if not result.valid:
                logger.error("Configuration validation failed - exiting")
                sys.exit(1)
            
            return result
        
        return asyncio.run(_validate())