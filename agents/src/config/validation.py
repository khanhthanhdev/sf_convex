"""
Configuration Validation Service for validating configuration values and provider compatibility.

This module provides the ConfigValidationService class that validates:
- API keys for different LLM providers
- Provider-model compatibility checks
- Embedding-vector store compatibility validation
- Connection testing for external services
- System configuration validation
"""

import asyncio
import aiohttp
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
import json

from .models import (
    SystemConfig, LLMProviderConfig, EmbeddingConfig, VectorStoreConfig, 
    RAGConfig, ValidationResult, AgentConfig
)


logger = logging.getLogger(__name__)


@dataclass
class ProviderCompatibility:
    """Represents provider-model compatibility information."""
    provider: str
    model: str
    compatible: bool
    reason: Optional[str] = None


@dataclass
class ConnectionTestResult:
    """Result of connection testing for external services."""
    service: str
    success: bool
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None


class ConfigValidationService:
    """Service for validating configuration values and provider compatibility.
    
    This service provides comprehensive validation including:
    - API key validation for LLM providers
    - Provider-model compatibility checks
    - Embedding-vector store compatibility validation
    - Connection testing for external services
    - System configuration validation
    """
    
    def __init__(self, timeout: float = 10.0):
        """Initialize the configuration validation service.
        
        Args:
            timeout: Timeout for network requests in seconds
        """
        self.timeout = timeout
        self.session_timeout = aiohttp.ClientTimeout(total=timeout)
        
        # Provider validation endpoints
        self.validation_endpoints = {
            'openai': {
                'url': 'https://api.openai.com/v1/models',
                'headers_fn': lambda key: {'Authorization': f'Bearer {key}'},
                'method': 'GET'
            },
            'anthropic': {
                'url': 'https://api.anthropic.com/v1/messages',
                'headers_fn': lambda key: {
                    'x-api-key': key,
                    'anthropic-version': '2023-06-01',
                    'content-type': 'application/json'
                },
                'method': 'POST',
                'data': {
                    'model': 'claude-3-haiku-20240307',
                    'max_tokens': 1,
                    'messages': [{'role': 'user', 'content': 'test'}]
                }
            },
            'gemini': {
                'url': 'https://generativelanguage.googleapis.com/v1beta/models',
                'headers_fn': lambda key: {},
                'params_fn': lambda key: {'key': key},
                'method': 'GET'
            },
            'openrouter': {
                'url': 'https://openrouter.ai/api/v1/models',
                'headers_fn': lambda key: {'Authorization': f'Bearer {key}'},
                'method': 'GET'
            },
            'jina': {
                'url': 'https://api.jina.ai/v1/embeddings',
                'headers_fn': lambda key: {
                    'Authorization': f'Bearer {key}',
                    'Content-Type': 'application/json'
                },
                'method': 'POST',
                'data': {
                    'model': 'jina-embeddings-v3',
                    'input': ['test'],
                    'encoding_format': 'float'
                }
            }
        }
        
        # Known provider-model compatibility
        self.provider_models = {
            'openai': {
                'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo',
                'text-embedding-3-large', 'text-embedding-3-small', 'text-embedding-ada-002'
            },
            'anthropic': {
                'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022', 
                'claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'
            },
            'gemini': {
                'gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-1.0-pro',
                'text-embedding-004', 'embedding-001'
            },
            'openrouter': {
                # OpenRouter supports many models, we'll validate dynamically
            },
            'jina': {
                'jina-embeddings-v3', 'jina-embeddings-v2-base-en', 'jina-embeddings-v2-small-en'
            }
        }
        
        # Embedding dimensions for compatibility checking
        self.embedding_dimensions = {
            'jina-embeddings-v3': 1024,
            'text-embedding-3-large': 3072,
            'text-embedding-3-small': 1536,
            'text-embedding-ada-002': 1536,
            'text-embedding-004': 768,
            'embedding-001': 768
        }
        
        logger.info("ConfigValidationService initialized")
    
    async def validate_api_key(self, provider: str, api_key: str) -> ValidationResult:
        """Validate an API key for a specific provider.
        
        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')
            api_key: API key to validate
            
        Returns:
            ValidationResult with validation status and details
        """
        result = ValidationResult(valid=True)
        
        if not api_key or not api_key.strip():
            result.add_error(f"API key for {provider} cannot be empty")
            return result
        
        if provider not in self.validation_endpoints:
            result.add_warning(f"API key validation not supported for provider: {provider}")
            return result
        
        try:
            endpoint_config = self.validation_endpoints[provider]
            url = endpoint_config['url']
            headers = endpoint_config['headers_fn'](api_key)
            method = endpoint_config.get('method', 'GET')
            
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                kwargs = {'headers': headers}
                
                # Add query parameters if needed
                if 'params_fn' in endpoint_config:
                    kwargs['params'] = endpoint_config['params_fn'](api_key)
                
                # Add data for POST requests
                if method == 'POST' and 'data' in endpoint_config:
                    kwargs['json'] = endpoint_config['data']
                
                start_time = time.time()
                async with session.request(method, url, **kwargs) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status < 300:
                        result.add_warning(f"API key validated successfully for {provider} (response time: {response_time:.1f}ms)")
                        logger.info(f"API key validation successful for {provider}")
                    elif response.status in [401, 403]:
                        result.add_error(f"Invalid API key for {provider} - please check your credentials")
                        logger.warning(f"Invalid API key for {provider}: {response.status}")
                    else:
                        result.add_warning(f"Could not verify API key for {provider} (service returned {response.status})")
                        logger.warning(f"Unexpected response for {provider}: {response.status}")
                        
        except asyncio.TimeoutError:
            result.add_warning(f"API key validation timed out for {provider} - key might be valid")
            logger.warning(f"Timeout validating API key for {provider}")
        except Exception as e:
            result.add_error(f"API key validation error for {provider}: {str(e)}")
            logger.error(f"Error validating API key for {provider}: {str(e)}")
        
        return result
    
    def validate_provider_model_compatibility(self, provider: str, model: str) -> ProviderCompatibility:
        """Validate that a model is compatible with a provider.
        
        Args:
            provider: Provider name
            model: Model name
            
        Returns:
            ProviderCompatibility with compatibility information
        """
        if provider not in self.provider_models:
            return ProviderCompatibility(
                provider=provider,
                model=model,
                compatible=False,
                reason=f"Unknown provider: {provider}"
            )
        
        known_models = self.provider_models[provider]
        
        # For OpenRouter, we can't validate all models statically
        if provider == 'openrouter':
            return ProviderCompatibility(
                provider=provider,
                model=model,
                compatible=True,
                reason="OpenRouter supports many models - validation skipped"
            )
        
        if model in known_models:
            return ProviderCompatibility(
                provider=provider,
                model=model,
                compatible=True
            )
        else:
            return ProviderCompatibility(
                provider=provider,
                model=model,
                compatible=False,
                reason=f"Model '{model}' not found in known models for {provider}. Known models: {', '.join(sorted(known_models))}"
            )
    
    def validate_embedding_vector_store_compatibility(
        self, 
        embedding_config: EmbeddingConfig, 
        vector_config: VectorStoreConfig
    ) -> ValidationResult:
        """Validate compatibility between embedding and vector store configurations.
        
        Args:
            embedding_config: Embedding configuration
            vector_config: Vector store configuration
            
        Returns:
            ValidationResult with compatibility status
        """
        result = ValidationResult(valid=True)
        
        # Check embedding dimensions compatibility
        model_name = embedding_config.model_name
        expected_dimensions = self.embedding_dimensions.get(model_name)
        
        if expected_dimensions and expected_dimensions != embedding_config.dimensions:
            result.add_error(
                f"Embedding dimensions mismatch: {model_name} produces {expected_dimensions} dimensions, "
                f"but configuration specifies {embedding_config.dimensions}"
            )
        
        # Vector store specific validations
        if vector_config.provider == 'astradb':
            # AstraDB requires specific connection parameters
            required_params = ['api_endpoint', 'application_token']
            missing_params = [param for param in required_params if param not in vector_config.connection_params]
            
            if missing_params:
                result.add_error(f"AstraDB missing required connection parameters: {', '.join(missing_params)}")
            
            # AstraDB works with any embedding dimensions
            
        elif vector_config.provider == 'chroma':
            # ChromaDB is flexible with dimensions
            if 'persist_directory' not in vector_config.connection_params:
                result.add_warning("ChromaDB persist_directory not specified - data will not be persisted")
            
        elif vector_config.provider == 'pinecone':
            # Pinecone requires specific parameters
            required_params = ['api_key', 'environment']
            missing_params = [param for param in required_params if param not in vector_config.connection_params]
            
            if missing_params:
                result.add_error(f"Pinecone missing required connection parameters: {', '.join(missing_params)}")
        
        # Validate distance metric compatibility
        valid_metrics = {
            'astradb': ['cosine', 'euclidean', 'dot_product'],
            'chroma': ['cosine', 'l2', 'ip'],
            'pinecone': ['cosine', 'euclidean', 'dotproduct']
        }
        
        provider_metrics = valid_metrics.get(vector_config.provider, [])
        if provider_metrics and vector_config.distance_metric not in provider_metrics:
            result.add_error(
                f"Distance metric '{vector_config.distance_metric}' not supported by {vector_config.provider}. "
                f"Supported metrics: {', '.join(provider_metrics)}"
            )
        
        return result
    
    async def test_provider_connection(self, provider_config: LLMProviderConfig) -> ConnectionTestResult:
        """Test connection to an LLM provider.
        
        Args:
            provider_config: Provider configuration to test
            
        Returns:
            ConnectionTestResult with connection status
        """
        if not provider_config.enabled:
            return ConnectionTestResult(
                service=provider_config.provider,
                success=False,
                error_message="Provider is disabled"
            )
        
        if not provider_config.api_key:
            return ConnectionTestResult(
                service=provider_config.provider,
                success=False,
                error_message="No API key configured"
            )
        
        # Use the API key validation as connection test
        validation_result = await self.validate_api_key(provider_config.provider, provider_config.api_key)
        
        return ConnectionTestResult(
            service=provider_config.provider,
            success=validation_result.valid,
            error_message='; '.join(validation_result.errors) if validation_result.errors else None
        )
    
    async def test_vector_store_connection(self, vector_config: VectorStoreConfig) -> ConnectionTestResult:
        """Test connection to a vector store.
        
        Args:
            vector_config: Vector store configuration to test
            
        Returns:
            ConnectionTestResult with connection status
        """
        start_time = time.time()
        
        try:
            if vector_config.provider == 'astradb':
                return await self._test_astradb_connection(vector_config, start_time)
            elif vector_config.provider == 'chroma':
                return await self._test_chroma_connection(vector_config, start_time)
            elif vector_config.provider == 'pinecone':
                return await self._test_pinecone_connection(vector_config, start_time)
            else:
                return ConnectionTestResult(
                    service=f"vector_store_{vector_config.provider}",
                    success=False,
                    error_message=f"Unknown vector store provider: {vector_config.provider}"
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return ConnectionTestResult(
                service=f"vector_store_{vector_config.provider}",
                success=False,
                response_time_ms=response_time,
                error_message=f"Connection test failed: {str(e)}"
            )
    
    async def _test_astradb_connection(self, vector_config: VectorStoreConfig, start_time: float) -> ConnectionTestResult:
        """Test AstraDB connection."""
        connection_params = vector_config.connection_params
        api_endpoint = connection_params.get('api_endpoint')
        application_token = connection_params.get('application_token')
        
        if not api_endpoint or not application_token:
            return ConnectionTestResult(
                service="vector_store_astradb",
                success=False,
                error_message="Missing api_endpoint or application_token"
            )
        
        # Test connection by making a simple API call
        headers = {
            'X-Cassandra-Token': application_token,
            'Content-Type': 'application/json'
        }
        
        # Use the collections endpoint to test connectivity
        url = f"{api_endpoint}/api/rest/v2/namespaces/default_keyspace/collections"
        
        async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
            async with session.get(url, headers=headers) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status < 300:
                    return ConnectionTestResult(
                        service="vector_store_astradb",
                        success=True,
                        response_time_ms=response_time
                    )
                else:
                    error_text = await response.text()
                    return ConnectionTestResult(
                        service="vector_store_astradb",
                        success=False,
                        response_time_ms=response_time,
                        error_message=f"HTTP {response.status}: {error_text[:200]}"
                    )
    
    async def _test_chroma_connection(self, vector_config: VectorStoreConfig, start_time: float) -> ConnectionTestResult:
        """Test ChromaDB connection."""
        # For ChromaDB, we'll test if the persist directory is accessible
        connection_params = vector_config.connection_params
        persist_directory = connection_params.get('persist_directory', './chroma_db')
        
        try:
            persist_path = Path(persist_directory)
            
            # Try to create the directory if it doesn't exist
            persist_path.mkdir(parents=True, exist_ok=True)
            
            # Test write access by creating a temporary file
            test_file = persist_path / '.connection_test'
            test_file.write_text('test')
            test_file.unlink()  # Clean up
            
            response_time = (time.time() - start_time) * 1000
            return ConnectionTestResult(
                service="vector_store_chroma",
                success=True,
                response_time_ms=response_time
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return ConnectionTestResult(
                service="vector_store_chroma",
                success=False,
                response_time_ms=response_time,
                error_message=f"Directory access failed: {str(e)}"
            )
    
    async def _test_pinecone_connection(self, vector_config: VectorStoreConfig, start_time: float) -> ConnectionTestResult:
        """Test Pinecone connection."""
        connection_params = vector_config.connection_params
        api_key = connection_params.get('api_key')
        environment = connection_params.get('environment')
        
        if not api_key or not environment:
            return ConnectionTestResult(
                service="vector_store_pinecone",
                success=False,
                error_message="Missing api_key or environment"
            )
        
        # Test connection by listing indexes
        url = f"https://controller.{environment}.pinecone.io/databases"
        headers = {
            'Api-Key': api_key,
            'Content-Type': 'application/json'
        }
        
        async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
            async with session.get(url, headers=headers) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status < 300:
                    return ConnectionTestResult(
                        service="vector_store_pinecone",
                        success=True,
                        response_time_ms=response_time
                    )
                else:
                    error_text = await response.text()
                    return ConnectionTestResult(
                        service="vector_store_pinecone",
                        success=False,
                        response_time_ms=response_time,
                        error_message=f"HTTP {response.status}: {error_text[:200]}"
                    )
    
    def validate_system_config(self, config: SystemConfig) -> ValidationResult:
        """Validate complete system configuration.
        
        Args:
            config: System configuration to validate
            
        Returns:
            ValidationResult with comprehensive validation status
        """
        result = ValidationResult(valid=True)
        
        try:
            # Validate basic configuration
            self._validate_basic_config(config, result)
            
            # Validate LLM providers
            self._validate_llm_providers_config(config, result)
            
            # Validate RAG configuration
            if config.rag_config:
                self._validate_rag_config(config.rag_config, result)
            
            # Validate agent configurations
            self._validate_agents_config(config, result)
            
            # Validate workflow configuration
            self._validate_workflow_config(config, result)
            
            # Validate external services
            self._validate_external_services_config(config, result)
            
            logger.info(f"System configuration validation completed: {'Valid' if result.valid else 'Invalid'}")
            
        except Exception as e:
            result.add_error(f"Configuration validation failed: {str(e)}")
            logger.error(f"Configuration validation error: {e}")
        
        return result
    
    def _validate_basic_config(self, config: SystemConfig, result: ValidationResult):
        """Validate basic system configuration."""
        # Validate environment
        valid_environments = ['development', 'staging', 'production']
        if config.environment not in valid_environments:
            result.add_error(f"Invalid environment '{config.environment}'. Must be one of: {', '.join(valid_environments)}")
        
        # Validate default provider exists
        if config.default_llm_provider not in config.llm_providers:
            result.add_error(f"Default LLM provider '{config.default_llm_provider}' not found in configured providers")
    
    def _validate_llm_providers_config(self, config: SystemConfig, result: ValidationResult):
        """Validate LLM provider configurations."""
        if not config.llm_providers:
            result.add_error("No LLM providers configured")
            return
        
        for provider_name, provider_config in config.llm_providers.items():
            # Validate provider has required fields
            if not provider_config.models:
                result.add_error(f"Provider '{provider_name}' has no models configured")
            
            if provider_config.default_model not in provider_config.models:
                result.add_error(f"Provider '{provider_name}' default model '{provider_config.default_model}' not in models list")
            
            # Validate model compatibility
            for model in provider_config.models:
                compatibility = self.validate_provider_model_compatibility(provider_name, model)
                if not compatibility.compatible:
                    result.add_warning(f"Provider '{provider_name}' model compatibility issue: {compatibility.reason}")
            
            # Check for API key if provider is enabled
            if provider_config.enabled and not provider_config.api_key and provider_name != 'local':
                result.add_warning(f"Provider '{provider_name}' is enabled but has no API key configured")
    
    def _validate_rag_config(self, rag_config: RAGConfig, result: ValidationResult):
        """Validate RAG configuration."""
        if not rag_config.enabled:
            return
        
        # Validate embedding configuration
        embedding_config = rag_config.embedding_config
        if embedding_config.provider != 'local' and not embedding_config.api_key:
            result.add_warning(f"Embedding provider '{embedding_config.provider}' has no API key configured")
        
        # Validate vector store configuration
        vector_config = rag_config.vector_store_config
        compatibility_result = self.validate_embedding_vector_store_compatibility(embedding_config, vector_config)
        
        if not compatibility_result.valid:
            for error in compatibility_result.errors:
                result.add_error(f"RAG compatibility issue: {error}")
        
        for warning in compatibility_result.warnings:
            result.add_warning(f"RAG compatibility warning: {warning}")
        
        # Validate chunk settings
        if rag_config.chunk_overlap >= rag_config.chunk_size:
            result.add_error("RAG chunk_overlap must be less than chunk_size")
        
        if rag_config.min_chunk_size > rag_config.chunk_size:
            result.add_error("RAG min_chunk_size must be less than or equal to chunk_size")
    
    def _validate_agents_config(self, config: SystemConfig, result: ValidationResult):
        """Validate agent configurations."""
        if not config.agent_configs:
            result.add_warning("No agents configured")
            return
        
        # Check for required agents
        required_agents = {'planner_agent', 'code_generator_agent', 'renderer_agent'}
        configured_agents = set(config.agent_configs.keys())
        missing_agents = required_agents - configured_agents
        
        if missing_agents:
            result.add_error(f"Required agents missing: {', '.join(missing_agents)}")
        
        # Validate individual agent configurations
        for agent_name, agent_config in config.agent_configs.items():
            if not agent_config.enabled:
                continue
            
            # Validate agent models reference valid providers
            model_fields = ['planner_model', 'scene_model', 'helper_model']
            for field in model_fields:
                model_name = getattr(agent_config, field, None)
                if model_name:
                    if '/' in model_name:
                        provider, model = model_name.split('/', 1)
                        if provider not in config.llm_providers:
                            result.add_error(f"Agent '{agent_name}' {field} references unknown provider: {provider}")
                        else:
                            provider_config = config.llm_providers[provider]
                            if model not in provider_config.models:
                                result.add_warning(f"Agent '{agent_name}' {field} model '{model}' not in provider's model list")
            
            # Validate numeric settings
            if agent_config.timeout_seconds <= 0:
                result.add_error(f"Agent '{agent_name}' timeout_seconds must be positive")
            
            if agent_config.max_retries < 0:
                result.add_error(f"Agent '{agent_name}' max_retries must be non-negative")
    
    def _validate_workflow_config(self, config: SystemConfig, result: ValidationResult):
        """Validate workflow configuration."""
        workflow_config = config.workflow_config
        
        if workflow_config.max_workflow_retries < 0:
            result.add_error("max_workflow_retries must be non-negative")
        
        if workflow_config.workflow_timeout_seconds <= 0:
            result.add_error("workflow_timeout_seconds must be positive")
        
        if workflow_config.checkpoint_interval <= 0:
            result.add_error("checkpoint_interval must be positive")
        
        if workflow_config.checkpoint_interval > workflow_config.workflow_timeout_seconds:
            result.add_warning("checkpoint_interval is longer than workflow timeout")
        
        # Validate concurrency settings
        if workflow_config.max_scene_concurrency <= 0:
            result.add_error("max_scene_concurrency must be positive")
        
        if workflow_config.max_topic_concurrency <= 0:
            result.add_error("max_topic_concurrency must be positive")
        
        if workflow_config.max_concurrent_renders <= 0:
            result.add_error("max_concurrent_renders must be positive")
        
        # Validate output directory
        output_path = Path(workflow_config.output_dir)
        if not output_path.exists():
            try:
                output_path.mkdir(parents=True, exist_ok=True)
                result.add_warning(f"Created output directory: {output_path}")
            except Exception as e:
                result.add_error(f"Cannot create output directory '{workflow_config.output_dir}': {e}")
    
    def _validate_external_services_config(self, config: SystemConfig, result: ValidationResult):
        """Validate external services configuration."""
        # Validate Langfuse configuration
        if config.monitoring_config.enabled and config.monitoring_config.langfuse_config:
            langfuse_config = config.monitoring_config.langfuse_config
            
            if langfuse_config.enabled:
                if not langfuse_config.secret_key:
                    result.add_error("Langfuse secret key is required when Langfuse is enabled")
                if not langfuse_config.public_key:
                    result.add_error("Langfuse public key is required when Langfuse is enabled")
                if not langfuse_config.host:
                    result.add_error("Langfuse host is required when Langfuse is enabled")
        
        # Validate MCP servers configuration
        for server_name, server_config in config.mcp_servers.items():
            if not server_config.disabled:
                if not server_config.command:
                    result.add_error(f"MCP server '{server_name}' missing command")
        
        # Validate Context7 configuration
        if config.context7_config.enabled:
            if config.context7_config.default_tokens <= 0:
                result.add_error("Context7 default_tokens must be positive")
            if config.context7_config.timeout_seconds <= 0:
                result.add_error("Context7 timeout_seconds must be positive")
        
        # Validate Docling configuration
        if config.docling_config.enabled:
            if config.docling_config.max_file_size_mb <= 0:
                result.add_error("Docling max_file_size_mb must be positive")
            if config.docling_config.timeout_seconds <= 0:
                result.add_error("Docling timeout_seconds must be positive")
    
    async def validate_startup_configuration(self, config: SystemConfig) -> ValidationResult:
        """Comprehensive startup configuration validation with detailed error messages.
        
        This method performs thorough validation of the entire system configuration
        at startup, including connectivity tests and compatibility checks.
        
        Args:
            config: System configuration to validate
            
        Returns:
            ValidationResult with detailed validation status
        """
        result = ValidationResult(valid=True)
        
        logger.info("Starting comprehensive configuration validation...")
        
        try:
            # 1. Basic configuration validation
            logger.debug("Validating basic configuration...")
            basic_result = self.validate_system_config(config)
            result.errors.extend(basic_result.errors)
            result.warnings.extend(basic_result.warnings)
            if not basic_result.valid:
                result.valid = False
            
            # 2. API key validation for enabled providers
            logger.debug("Validating API keys...")
            await self._validate_api_keys_startup(config, result)
            
            # 3. Connection testing for external services
            logger.debug("Testing external service connections...")
            await self._test_external_connections_startup(config, result)
            
            # 4. Component compatibility validation
            logger.debug("Validating component compatibility...")
            self._validate_component_compatibility_startup(config, result)
            
            # 5. Resource availability checks
            logger.debug("Checking resource availability...")
            self._validate_resource_availability(config, result)
            
            # 6. Security validation
            logger.debug("Performing security validation...")
            self._validate_security_configuration(config, result)
            
            # 7. Performance configuration validation
            logger.debug("Validating performance configuration...")
            self._validate_performance_configuration(config, result)
            
            logger.info(f"Configuration validation completed: {'✅ Valid' if result.valid else '❌ Invalid'}")
            if result.errors:
                logger.error(f"Found {len(result.errors)} configuration errors")
            if result.warnings:
                logger.warning(f"Found {len(result.warnings)} configuration warnings")
                
        except Exception as e:
            result.add_error(f"Configuration validation failed with exception: {str(e)}")
            logger.error(f"Configuration validation exception: {e}")
        
        return result
    
    async def _validate_api_keys_startup(self, config: SystemConfig, result: ValidationResult):
        """Validate API keys for all enabled providers during startup."""
        validation_tasks = []
        
        # Validate LLM provider API keys
        for provider_name, provider_config in config.llm_providers.items():
            if provider_config.enabled and provider_config.api_key:
                validation_tasks.append(
                    self._validate_single_api_key(provider_name, provider_config.api_key, result)
                )
        
        # Validate embedding provider API key
        if config.rag_config and config.rag_config.enabled:
            embedding_config = config.rag_config.embedding_config
            if embedding_config.provider != 'local' and embedding_config.api_key:
                validation_tasks.append(
                    self._validate_single_api_key(embedding_config.provider, embedding_config.api_key, result)
                )
        
        # Run all validations concurrently
        if validation_tasks:
            await asyncio.gather(*validation_tasks, return_exceptions=True)
    
    async def _validate_single_api_key(self, provider: str, api_key: str, result: ValidationResult):
        """Validate a single API key and add results to the main result."""
        try:
            validation_result = await self.validate_api_key(provider, api_key)
            if not validation_result.valid:
                for error in validation_result.errors:
                    result.add_error(f"API key validation failed for {provider}: {error}")
            else:
                for warning in validation_result.warnings:
                    result.add_warning(f"API key validation for {provider}: {warning}")
        except Exception as e:
            result.add_error(f"API key validation error for {provider}: {str(e)}")
    
    async def _test_external_connections_startup(self, config: SystemConfig, result: ValidationResult):
        """Test connections to external services during startup."""
        connection_tasks = []
        
        # Test LLM provider connections
        for provider_name, provider_config in config.llm_providers.items():
            if provider_config.enabled:
                connection_tasks.append(
                    self._test_single_provider_connection(provider_config, result)
                )
        
        # Test vector store connection
        if config.rag_config and config.rag_config.enabled:
            connection_tasks.append(
                self._test_single_vector_store_connection(config.rag_config.vector_store_config, result)
            )
        
        # Run all connection tests concurrently
        if connection_tasks:
            await asyncio.gather(*connection_tasks, return_exceptions=True)
    
    async def _test_single_provider_connection(self, provider_config: LLMProviderConfig, result: ValidationResult):
        """Test connection to a single provider and add results to the main result."""
        try:
            connection_result = await self.test_provider_connection(provider_config)
            if not connection_result.success:
                result.add_error(f"Connection test failed for {provider_config.provider}: {connection_result.error_message}")
            else:
                result.add_warning(f"Connection test successful for {provider_config.provider} (response time: {connection_result.response_time_ms:.1f}ms)")
        except Exception as e:
            result.add_error(f"Connection test error for {provider_config.provider}: {str(e)}")
    
    async def _test_single_vector_store_connection(self, vector_config: VectorStoreConfig, result: ValidationResult):
        """Test connection to vector store and add results to the main result."""
        try:
            connection_result = await self.test_vector_store_connection(vector_config)
            if not connection_result.success:
                result.add_error(f"Vector store connection test failed for {vector_config.provider}: {connection_result.error_message}")
            else:
                result.add_warning(f"Vector store connection test successful for {vector_config.provider} (response time: {connection_result.response_time_ms:.1f}ms)")
        except Exception as e:
            result.add_error(f"Vector store connection test error for {vector_config.provider}: {str(e)}")
    
    def _validate_component_compatibility_startup(self, config: SystemConfig, result: ValidationResult):
        """Validate compatibility between different system components."""
        # Validate agent-provider compatibility
        for agent_name, agent_config in config.agent_configs.items():
            if not agent_config.enabled:
                continue
            
            # Check if agent's models are compatible with configured providers
            model_fields = ['planner_model', 'scene_model', 'helper_model']
            for field in model_fields:
                model_name = getattr(agent_config, field, None)
                if model_name and '/' in model_name:
                    provider, model = model_name.split('/', 1)
                    if provider in config.llm_providers:
                        provider_config = config.llm_providers[provider]
                        if not provider_config.enabled:
                            result.add_error(f"Agent '{agent_name}' {field} references disabled provider: {provider}")
                        elif model not in provider_config.models:
                            result.add_warning(f"Agent '{agent_name}' {field} model '{model}' not in provider's model list")
        
        # Validate RAG-embedding compatibility
        if config.rag_config and config.rag_config.enabled:
            embedding_config = config.rag_config.embedding_config
            vector_config = config.rag_config.vector_store_config
            
            compatibility_result = self.validate_embedding_vector_store_compatibility(embedding_config, vector_config)
            if not compatibility_result.valid:
                for error in compatibility_result.errors:
                    result.add_error(f"RAG component compatibility issue: {error}")
            for warning in compatibility_result.warnings:
                result.add_warning(f"RAG component compatibility: {warning}")
        
        # Validate monitoring-provider compatibility
        if config.monitoring_config.enabled and config.monitoring_config.langfuse_config:
            langfuse_config = config.monitoring_config.langfuse_config
            if langfuse_config.enabled:
                # Check if any providers support Langfuse integration
                supported_providers = ['openai', 'anthropic', 'gemini']
                enabled_supported = [p for p in supported_providers if p in config.llm_providers and config.llm_providers[p].enabled]
                if not enabled_supported:
                    result.add_warning("Langfuse monitoring enabled but no supported providers are enabled")
    
    def _validate_resource_availability(self, config: SystemConfig, result: ValidationResult):
        """Validate that required resources are available."""
        # Check output directory
        output_dir = Path(config.workflow_config.output_dir)
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                result.add_warning(f"Created output directory: {output_dir}")
            except Exception as e:
                result.add_error(f"Cannot create output directory '{config.workflow_config.output_dir}': {e}")
        
        # Check disk space for output directory
        try:
            import shutil
            total, used, free = shutil.disk_usage(output_dir)
            free_gb = free // (1024**3)
            if free_gb < 1:  # Less than 1GB free
                result.add_warning(f"Low disk space in output directory: {free_gb}GB free")
        except Exception as e:
            result.add_warning(f"Could not check disk space: {e}")
        
        # Check TTS model files if configured
        if config.kokoro_model_path:
            model_path = Path(config.kokoro_model_path)
            if not model_path.exists():
                result.add_error(f"Kokoro TTS model file not found: {config.kokoro_model_path}")
        
        if config.kokoro_voices_path:
            voices_path = Path(config.kokoro_voices_path)
            if not voices_path.exists():
                result.add_error(f"Kokoro TTS voices file not found: {config.kokoro_voices_path}")
        
        # Check vector store persistence directory
        if config.rag_config and config.rag_config.enabled:
            vector_config = config.rag_config.vector_store_config
            if vector_config.provider == 'chroma':
                persist_dir = vector_config.connection_params.get('persist_directory', './chroma_db')
                persist_path = Path(persist_dir)
                try:
                    persist_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    result.add_error(f"Cannot create ChromaDB persist directory '{persist_dir}': {e}")
    
    def _validate_security_configuration(self, config: SystemConfig, result: ValidationResult):
        """Validate security-related configuration."""
        # Check for API keys in environment variables vs hardcoded
        for provider_name, provider_config in config.llm_providers.items():
            if provider_config.api_key and not provider_config.api_key.startswith('$'):
                # API key appears to be hardcoded (not an env var reference)
                if len(provider_config.api_key) > 10:  # Likely a real key
                    result.add_warning(f"API key for {provider_name} appears to be hardcoded - consider using environment variables")
        
        # Check production environment security
        if config.environment == 'production':
            if config.debug:
                result.add_error("Debug mode should be disabled in production")
            
            if config.monitoring_config.log_level == 'DEBUG':
                result.add_warning("Debug logging enabled in production - consider using INFO or WARNING")
        
        # Check for secure URLs
        for provider_name, provider_config in config.llm_providers.items():
            if provider_config.base_url and not provider_config.base_url.startswith('https://'):
                result.add_warning(f"Provider {provider_name} using non-HTTPS URL: {provider_config.base_url}")
        
        # Check Langfuse configuration security
        if config.monitoring_config.langfuse_config and config.monitoring_config.langfuse_config.enabled:
            langfuse_config = config.monitoring_config.langfuse_config
            if langfuse_config.host and not langfuse_config.host.startswith('https://'):
                result.add_warning(f"Langfuse host using non-HTTPS URL: {langfuse_config.host}")
    
    def _validate_performance_configuration(self, config: SystemConfig, result: ValidationResult):
        """Validate performance-related configuration."""
        workflow_config = config.workflow_config
        
        # Check concurrency settings
        total_concurrency = (workflow_config.max_scene_concurrency + 
                           workflow_config.max_topic_concurrency + 
                           workflow_config.max_concurrent_renders)
        
        if total_concurrency > 20:  # Arbitrary threshold
            result.add_warning(f"High total concurrency ({total_concurrency}) may impact system performance")
        
        # Check timeout settings
        if workflow_config.workflow_timeout_seconds > 7200:  # 2 hours
            result.add_warning(f"Very long workflow timeout ({workflow_config.workflow_timeout_seconds}s) may cause resource issues")
        
        # Check agent timeout settings
        for agent_name, agent_config in config.agent_configs.items():
            if agent_config.timeout_seconds > 600:  # 10 minutes
                result.add_warning(f"Agent '{agent_name}' has long timeout ({agent_config.timeout_seconds}s)")
        
        # Check RAG performance settings
        if config.rag_config and config.rag_config.enabled:
            rag_config = config.rag_config
            if rag_config.embedding_config.batch_size > 1000:
                result.add_warning(f"Large embedding batch size ({rag_config.embedding_config.batch_size}) may cause memory issues")
            
            if rag_config.max_cache_size > 10000:
                result.add_warning(f"Large RAG cache size ({rag_config.max_cache_size}) may cause memory issues")
    
    def get_supported_providers(self) -> List[str]:
        """Get list of supported providers for validation."""
        return list(self.validation_endpoints.keys())
    
    def get_provider_models(self, provider: str) -> Set[str]:
        """Get known models for a provider."""
        return self.provider_models.get(provider, set())
    
    def get_embedding_dimensions(self, model_name: str) -> Optional[int]:
        """Get embedding dimensions for a model."""
        return self.embedding_dimensions.get(model_name)
    
    def validate_api_key_sync(self, provider: str, api_key: str) -> ValidationResult:
        """Synchronous wrapper for API key validation."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, create a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(self.validate_api_key(provider, api_key))
                    )
                    return future.result(timeout=self.timeout + 5)
            else:
                return loop.run_until_complete(self.validate_api_key(provider, api_key))
        except Exception as e:
            logger.error(f"Error in sync validation: {str(e)}")
            result = ValidationResult(valid=False)
            result.add_error(f"Validation failed: {str(e)}")
            return result
    
    def test_provider_connection_sync(self, provider_config: LLMProviderConfig) -> ConnectionTestResult:
        """Synchronous wrapper for provider connection testing."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, create a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(self.test_provider_connection(provider_config))
                    )
                    return future.result(timeout=self.timeout + 5)
            else:
                return loop.run_until_complete(self.test_provider_connection(provider_config))
        except Exception as e:
            logger.error(f"Error in sync connection test: {str(e)}")
            return ConnectionTestResult(
                service=provider_config.provider,
                success=False,
                error_message=f"Connection test failed: {str(e)}"
            )
            if langfuse_config.enabled:
                if not langfuse_config.secret_key:
                    result.add_warning("Langfuse is enabled but secret_key is missing")
                if not langfuse_config.public_key:
                    result.add_warning("Langfuse is enabled but public_key is missing")
        
        # Validate MCP servers
        for server_name, server_config in config.mcp_servers.items():
            if not server_config.command:
                result.add_error(f"MCP server '{server_name}' missing command")
        
        # Validate TTS configuration
        if config.kokoro_model_path:
            model_path = Path(config.kokoro_model_path)
            if not model_path.exists():
                result.add_warning(f"Kokoro model path does not exist: {model_path}")
        
        if config.kokoro_voices_path:
            voices_path = Path(config.kokoro_voices_path)
            if not voices_path.exists():
                result.add_warning(f"Kokoro voices path does not exist: {voices_path}")
    
    def validate_api_key_sync(self, provider: str, api_key: str) -> ValidationResult:
        """Synchronous wrapper for API key validation.
        
        Args:
            provider: Provider name
            api_key: API key to validate
            
        Returns:
            ValidationResult with validation status
        """
        try:
            # Try to get the current event loop
            try:
                loop = asyncio.get_running_loop()
                # If we're already in an event loop, create a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(self.validate_api_key(provider, api_key))
                    )
                    return future.result(timeout=self.timeout + 5)
            except RuntimeError:
                # No running event loop, we can create one
                return asyncio.run(self.validate_api_key(provider, api_key))
        except Exception as e:
            result = ValidationResult(valid=False)
            result.add_error(f"API key validation failed: {str(e)}")
            return result
    
    def test_provider_connection_sync(self, provider_config: LLMProviderConfig) -> ConnectionTestResult:
        """Synchronous wrapper for provider connection testing.
        
        Args:
            provider_config: Provider configuration to test
            
        Returns:
            ConnectionTestResult with connection status
        """
        try:
            # Try to get the current event loop
            try:
                loop = asyncio.get_running_loop()
                # If we're already in an event loop, create a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(self.test_provider_connection(provider_config))
                    )
                    return future.result(timeout=self.timeout + 5)
            except RuntimeError:
                # No running event loop, we can create one
                return asyncio.run(self.test_provider_connection(provider_config))
        except Exception as e:
            return ConnectionTestResult(
                service=provider_config.provider,
                success=False,
                error_message=f"Connection test failed: {str(e)}"
            )
    
    def get_supported_providers(self) -> List[str]:
        """Get list of supported providers for validation.
        
        Returns:
            List of supported provider names
        """
        return list(self.validation_endpoints.keys())
    
    def get_provider_models(self, provider: str) -> Set[str]:
        """Get known models for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Set of known model names
        """
        return self.provider_models.get(provider, set())
    
    def get_embedding_dimensions(self, model_name: str) -> Optional[int]:
        """Get expected dimensions for an embedding model.
        
        Args:
            model_name: Embedding model name
            
        Returns:
            Expected dimensions or None if unknown
        """
        return self.embedding_dimensions.get(model_name)