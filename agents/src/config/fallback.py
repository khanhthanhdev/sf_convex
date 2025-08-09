"""
Fallback configuration system for handling missing or invalid configuration.

This module provides fallback mechanisms to ensure the system can continue
operating even when configuration is missing or invalid.
"""

import logging
import os
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import json

from .models import (
    SystemConfig, LLMProviderConfig, EmbeddingConfig, VectorStoreConfig,
    RAGConfig, AgentConfig, MonitoringConfig, LangfuseConfig,
    WorkflowConfig, HumanLoopConfig, DoclingConfig, Context7Config
)
from .error_handling import ConfigErrorCategory, ConfigErrorSeverity, handle_config_error


logger = logging.getLogger(__name__)


class FallbackConfigurationProvider:
    """Provides fallback configurations for system components."""
    
    def __init__(self):
        """Initialize the fallback configuration provider."""
        self.fallback_configs: Dict[str, Dict[str, Any]] = {}
        self.fallback_factories: Dict[str, Callable[[], Any]] = {}
        self._register_default_fallbacks()
        
        logger.info("FallbackConfigurationProvider initialized")
    
    def _register_default_fallbacks(self):
        """Register default fallback configurations."""
        # Register fallback factories for each component
        self.fallback_factories.update({
            'llm_providers': self._create_fallback_llm_providers,
            'rag_config': self._create_fallback_rag_config,
            'agent_configs': self._create_fallback_agent_configs,
            'monitoring_config': self._create_fallback_monitoring_config,
            'workflow_config': self._create_fallback_workflow_config,
            'human_loop_config': self._create_fallback_human_loop_config,
            'docling_config': self._create_fallback_docling_config,
            'context7_config': self._create_fallback_context7_config,
            'system_config': self._create_fallback_system_config
        })
    
    def get_fallback_config(self, component: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Get fallback configuration for a component.
        
        Args:
            component: Component name
            context: Additional context for fallback generation
            
        Returns:
            Fallback configuration
        """
        try:
            # Check if we have a cached fallback
            if component in self.fallback_configs:
                logger.info(f"Using cached fallback configuration for {component}")
                return self.fallback_configs[component]
            
            # Check if we have a factory for this component
            if component in self.fallback_factories:
                logger.warning(f"Creating fallback configuration for {component}")
                fallback = self.fallback_factories[component]()
                
                # Cache the fallback
                self.fallback_configs[component] = fallback
                
                # Log fallback usage for audit
                handle_config_error(
                    category=ConfigErrorCategory.LOADING,
                    severity=ConfigErrorSeverity.MEDIUM,
                    message=f"Using fallback configuration for {component}",
                    component="fallback_provider",
                    context=context or {},
                    suggested_fix=f"Check {component} configuration and fix any issues"
                )
                
                return fallback
            
            logger.error(f"No fallback configuration available for {component}")
            return None
            
        except Exception as e:
            handle_config_error(
                category=ConfigErrorCategory.LOADING,
                severity=ConfigErrorSeverity.HIGH,
                message=f"Failed to create fallback configuration for {component}: {str(e)}",
                component="fallback_provider",
                exception=e,
                context=context or {},
                suggested_fix=f"Check fallback configuration implementation for {component}"
            )
            return None
    
    def _create_fallback_llm_providers(self) -> Dict[str, LLMProviderConfig]:
        """Create fallback LLM provider configurations."""
        fallback_providers = {}
        
        # OpenAI fallback (if API key is available)
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            fallback_providers['openai'] = LLMProviderConfig(
                provider='openai',
                api_key=openai_key,
                models=['gpt-4o-mini', 'gpt-3.5-turbo'],
                default_model='gpt-4o-mini',
                enabled=True,
                timeout=30,
                max_retries=3
            )
        
        # Gemini fallback (if API key is available)
        gemini_key = os.getenv('GEMINI_API_KEY')
        if gemini_key:
            fallback_providers['gemini'] = LLMProviderConfig(
                provider='gemini',
                api_key=gemini_key,
                models=['gemini-1.5-flash', 'gemini-1.5-pro'],
                default_model='gemini-1.5-flash',
                enabled=True,
                timeout=30,
                max_retries=3
            )
        
        # If no API keys available, create a minimal local provider
        if not fallback_providers:
            fallback_providers['local'] = LLMProviderConfig(
                provider='local',
                api_key=None,
                models=['local-model'],
                default_model='local-model',
                enabled=False,  # Disabled by default since it's not functional
                timeout=30,
                max_retries=1
            )
        
        logger.info(f"Created fallback LLM providers: {list(fallback_providers.keys())}")
        return fallback_providers
    
    def _create_fallback_rag_config(self) -> RAGConfig:
        """Create fallback RAG configuration."""
        # Create minimal embedding config
        embedding_config = EmbeddingConfig(
            provider='local',
            model_name='local-embeddings',
            dimensions=384,  # Common small embedding size
            batch_size=32,
            timeout=30,
            max_retries=3,
            device='cpu'
        )
        
        # Create minimal vector store config
        vector_store_config = VectorStoreConfig(
            provider='chroma',
            collection_name='fallback_collection',
            connection_params={
                'persist_directory': './fallback_chroma_db'
            },
            max_results=10,
            distance_metric='cosine',
            timeout=30,
            max_retries=3
        )
        
        # Create RAG config with conservative settings
        rag_config = RAGConfig(
            enabled=False,  # Disabled by default in fallback
            embedding_config=embedding_config,
            vector_store_config=vector_store_config,
            chunk_size=500,
            chunk_overlap=50,
            min_chunk_size=100,
            default_k_value=3,
            similarity_threshold=0.5,
            enable_query_expansion=False,
            enable_semantic_search=True,
            enable_caching=False,
            cache_ttl=1800,
            max_cache_size=100,
            enable_quality_monitoring=False,
            quality_threshold=0.5
        )
        
        logger.info("Created fallback RAG configuration (disabled)")
        return rag_config
    
    def _create_fallback_agent_configs(self) -> Dict[str, AgentConfig]:
        """Create fallback agent configurations."""
        # Determine available model
        available_model = None
        if os.getenv('OPENAI_API_KEY'):
            available_model = 'openai/gpt-4o-mini'
        elif os.getenv('GEMINI_API_KEY'):
            available_model = 'gemini/gemini-1.5-flash'
        else:
            available_model = 'local/local-model'
        
        # Create minimal agent configs
        agent_configs = {}
        
        required_agents = ['planner_agent', 'code_generator_agent', 'renderer_agent']
        for agent_name in required_agents:
            agent_configs[agent_name] = AgentConfig(
                name=agent_name,
                llm_config={},
                tools=[],
                max_retries=2,
                timeout_seconds=180,
                enable_human_loop=False,
                planner_model=available_model,
                scene_model=available_model,
                helper_model=available_model,
                temperature=0.7,
                print_cost=False,
                verbose=False,
                enabled=available_model != 'local/local-model',  # Disable if no real API key
                system_prompt=f"You are a {agent_name.replace('_', ' ')} assistant."
            )
        
        logger.info(f"Created fallback agent configurations for {list(agent_configs.keys())}")
        return agent_configs
    
    def _create_fallback_monitoring_config(self) -> MonitoringConfig:
        """Create fallback monitoring configuration."""
        # Create minimal Langfuse config
        langfuse_config = None
        if os.getenv('LANGFUSE_SECRET_KEY') and os.getenv('LANGFUSE_PUBLIC_KEY'):
            langfuse_config = LangfuseConfig(
                enabled=True,
                secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
                public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
                host=os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')
            )
        
        monitoring_config = MonitoringConfig(
            enabled=langfuse_config is not None,
            langfuse_config=langfuse_config,
            log_level='INFO',
            metrics_collection_interval=600,  # 10 minutes
            performance_tracking=True,
            error_tracking=True,
            execution_tracing=langfuse_config is not None,
            cpu_threshold=90.0,
            memory_threshold=90.0,
            execution_time_threshold=600.0,
            history_retention_hours=12
        )
        
        logger.info("Created fallback monitoring configuration")
        return monitoring_config
    
    def _create_fallback_workflow_config(self) -> WorkflowConfig:
        """Create fallback workflow configuration."""
        workflow_config = WorkflowConfig(
            max_workflow_retries=2,
            workflow_timeout_seconds=1800,  # 30 minutes
            enable_checkpoints=True,
            checkpoint_interval=300,  # 5 minutes
            output_dir='output',
            max_scene_concurrency=2,
            max_topic_concurrency=1,
            max_concurrent_renders=2,
            default_quality='medium',
            use_gpu_acceleration=False,
            preview_mode=True  # Enable preview mode for safety
        )
        
        logger.info("Created fallback workflow configuration")
        return workflow_config
    
    def _create_fallback_human_loop_config(self) -> HumanLoopConfig:
        """Create fallback human loop configuration."""
        human_loop_config = HumanLoopConfig(
            enabled=True,
            enable_interrupts=True,
            timeout_seconds=300,
            auto_approve_low_risk=False
        )
        
        logger.info("Created fallback human loop configuration")
        return human_loop_config
    
    def _create_fallback_docling_config(self) -> DoclingConfig:
        """Create fallback Docling configuration."""
        docling_config = DoclingConfig(
            enabled=False,  # Disabled by default in fallback
            max_file_size_mb=10,
            supported_formats=['txt', 'md'],
            timeout_seconds=60
        )
        
        logger.info("Created fallback Docling configuration (disabled)")
        return docling_config
    
    def _create_fallback_context7_config(self) -> Context7Config:
        """Create fallback Context7 configuration."""
        context7_config = Context7Config(
            enabled=False,  # Disabled by default in fallback
            default_tokens=5000,
            timeout_seconds=30,
            cache_responses=False,
            cache_ttl=1800
        )
        
        logger.info("Created fallback Context7 configuration (disabled)")
        return context7_config
    
    def _create_fallback_system_config(self) -> SystemConfig:
        """Create complete fallback system configuration."""
        try:
            system_config = SystemConfig(
                environment='development',
                debug=True,
                default_llm_provider=self._get_default_provider(),
                llm_providers=self._create_fallback_llm_providers(),
                rag_config=self._create_fallback_rag_config(),
                agent_configs=self._create_fallback_agent_configs(),
                monitoring_config=self._create_fallback_monitoring_config(),
                workflow_config=self._create_fallback_workflow_config(),
                human_loop_config=self._create_fallback_human_loop_config(),
                docling_config=self._create_fallback_docling_config(),
                context7_config=self._create_fallback_context7_config(),
                mcp_servers={},
                kokoro_model_path=None,
                kokoro_voices_path=None,
                kokoro_default_voice='af',
                kokoro_default_speed=1.0,
                kokoro_default_lang='en-us'
            )
            
            logger.info("Created complete fallback system configuration")
            return system_config
            
        except Exception as e:
            logger.error(f"Failed to create fallback system configuration: {e}")
            # Return absolute minimal config
            return self._create_absolute_minimal_config()
    
    def _get_default_provider(self) -> str:
        """Determine the default provider based on available API keys."""
        if os.getenv('OPENAI_API_KEY'):
            return 'openai'
        elif os.getenv('GEMINI_API_KEY'):
            return 'gemini'
        else:
            return 'local'
    
    def _create_absolute_minimal_config(self) -> SystemConfig:
        """Create absolute minimal configuration that should always work."""
        try:
            return SystemConfig(
                environment='development',
                debug=True,
                default_llm_provider='local',
                llm_providers={
                    'local': LLMProviderConfig(
                        provider='local',
                        api_key=None,
                        models=['local-model'],
                        default_model='local-model',
                        enabled=False
                    )
                },
                agent_configs={
                    'planner_agent': AgentConfig(
                        name='planner_agent',
                        enabled=False
                    )
                }
            )
        except Exception as e:
            logger.critical(f"Failed to create absolute minimal configuration: {e}")
            raise RuntimeError("Cannot create any configuration - system is in critical state")
    
    def register_custom_fallback(self, component: str, fallback_factory: Callable[[], Any]):
        """Register a custom fallback factory for a component.
        
        Args:
            component: Component name
            fallback_factory: Factory function that creates fallback configuration
        """
        self.fallback_factories[component] = fallback_factory
        logger.info(f"Registered custom fallback factory for {component}")
    
    def clear_cached_fallbacks(self):
        """Clear cached fallback configurations."""
        cleared_count = len(self.fallback_configs)
        self.fallback_configs.clear()
        logger.info(f"Cleared {cleared_count} cached fallback configurations")
    
    def get_fallback_status(self) -> Dict[str, Any]:
        """Get status of fallback configurations.
        
        Returns:
            Dictionary with fallback status information
        """
        return {
            'available_fallbacks': list(self.fallback_factories.keys()),
            'cached_fallbacks': list(self.fallback_configs.keys()),
            'api_keys_available': {
                'openai': bool(os.getenv('OPENAI_API_KEY')),
                'gemini': bool(os.getenv('GEMINI_API_KEY')),
                'langfuse': bool(os.getenv('LANGFUSE_SECRET_KEY') and os.getenv('LANGFUSE_PUBLIC_KEY'))
            },
            'recommended_provider': self._get_default_provider()
        }
    
    def validate_fallback_config(self, component: str) -> bool:
        """Validate that a fallback configuration can be created.
        
        Args:
            component: Component name
            
        Returns:
            True if fallback can be created, False otherwise
        """
        try:
            if component not in self.fallback_factories:
                return False
            
            # Try to create the fallback
            fallback = self.fallback_factories[component]()
            return fallback is not None
            
        except Exception as e:
            logger.error(f"Fallback validation failed for {component}: {e}")
            return False
    
    def create_emergency_config(self) -> SystemConfig:
        """Create emergency configuration for critical system failures.
        
        Returns:
            Emergency SystemConfig that should always work
        """
        logger.critical("Creating emergency configuration due to critical system failure")
        
        try:
            # Create the most basic configuration possible
            emergency_config = SystemConfig(
                environment='development',
                debug=True,
                default_llm_provider='emergency',
                llm_providers={
                    'emergency': LLMProviderConfig(
                        provider='emergency',
                        api_key=None,
                        models=['emergency-model'],
                        default_model='emergency-model',
                        enabled=False
                    )
                },
                agent_configs={
                    'emergency_agent': AgentConfig(
                        name='emergency_agent',
                        enabled=False,
                        timeout_seconds=60,
                        max_retries=1
                    )
                }
            )
            
            # Log emergency config creation for audit
            handle_config_error(
                category=ConfigErrorCategory.LOADING,
                severity=ConfigErrorSeverity.CRITICAL,
                message="Emergency configuration created due to critical system failure",
                component="fallback_provider",
                suggested_fix="Investigate and fix the underlying configuration issues immediately"
            )
            
            return emergency_config
            
        except Exception as e:
            logger.critical(f"Failed to create emergency configuration: {e}")
            raise RuntimeError("System is in critical state - cannot create any configuration")


# Global fallback provider instance
_fallback_provider: Optional[FallbackConfigurationProvider] = None


def get_fallback_provider() -> FallbackConfigurationProvider:
    """Get the global fallback configuration provider."""
    global _fallback_provider
    
    if _fallback_provider is None:
        _fallback_provider = FallbackConfigurationProvider()
    
    return _fallback_provider


def get_fallback_config(component: str, context: Optional[Dict[str, Any]] = None) -> Any:
    """Get fallback configuration for a component.
    
    Args:
        component: Component name
        context: Additional context for fallback generation
        
    Returns:
        Fallback configuration
    """
    return get_fallback_provider().get_fallback_config(component, context)


def create_emergency_config() -> SystemConfig:
    """Create emergency configuration for critical failures.
    
    Returns:
        Emergency SystemConfig
    """
    return get_fallback_provider().create_emergency_config()