"""
Configuration factory for building configuration models from environment variables.

This module provides utilities to parse the existing .env structure and build
the Pydantic configuration models with proper validation.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from .models import (
    SystemConfig, LLMProviderConfig, EmbeddingConfig, VectorStoreConfig,
    RAGConfig, AgentConfig, LangfuseConfig, MonitoringConfig, ValidationResult,
    DoclingConfig, MCPServerConfig, Context7Config, HumanLoopConfig, WorkflowConfig
)
from .error_handling import (
    ConfigErrorCategory, ConfigErrorSeverity, handle_config_error
)
from .logging_config import get_config_logger


logger = logging.getLogger(__name__)


class ConfigurationFactory:
    """Factory class for building configuration models from environment variables."""
    
    @staticmethod
    def _get_env_value(key: str, default: Any = None, cast_type: type = str) -> Any:
        """Get environment variable value with type casting."""
        value = os.getenv(key)
        if value is None:
            return default
        
        if cast_type == bool:
            if isinstance(value, bool):
                return value
            return str(value).lower() in ('true', '1', 'yes', 'on')
        elif cast_type == int:
            try:
                return int(value)
            except (ValueError, TypeError):
                return default
        elif cast_type == float:
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        elif cast_type == list:
            if isinstance(value, str):
                # Handle comma-separated values
                items = [item.strip() for item in value.split(',') if item.strip()]
                return items if items else default if default is not None else []
            elif isinstance(value, list):
                return value
            else:
                return [value] if value else (default if default is not None else [])
        
        return value
    
    @staticmethod
    def build_llm_provider_configs() -> Dict[str, LLMProviderConfig]:
        """Build LLM provider configurations from environment variables."""
        providers = {}
        
        # OpenAI Configuration
        openai_api_key = ConfigurationFactory._get_env_value('OPENAI_API_KEY')
        if openai_api_key and openai_api_key.strip():
            openai_models = ConfigurationFactory._get_env_value(
                'OPENAI_MODELS', 
                ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'], 
                list
            )
            providers['openai'] = LLMProviderConfig(
                provider='openai',
                api_key=openai_api_key,
                base_url=ConfigurationFactory._get_env_value('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
                models=openai_models,
                default_model=ConfigurationFactory._get_env_value('OPENAI_DEFAULT_MODEL', 'gpt-4o'),
                enabled=True,
                timeout=ConfigurationFactory._get_env_value('OPENAI_TIMEOUT', 30, int),
                max_retries=ConfigurationFactory._get_env_value('OPENAI_MAX_RETRIES', 3, int)
            )
        
       
        
        
        # Gemini Configuration
        gemini_api_key = ConfigurationFactory._get_env_value('GEMINI_API_KEY')
        if gemini_api_key:
            gemini_models = ConfigurationFactory._get_env_value(
                'GEMINI_MODELS',
                ['gemini-1.5-pro', 'gemini-1.5-flash'],
                list
            )
            providers['gemini'] = LLMProviderConfig(
                provider='gemini',
                api_key=gemini_api_key,
                base_url=ConfigurationFactory._get_env_value('GEMINI_BASE_URL'),
                models=gemini_models,
                default_model=ConfigurationFactory._get_env_value('GEMINI_DEFAULT_MODEL', 'gemini-1.5-pro'),
                enabled=True,
                timeout=ConfigurationFactory._get_env_value('GEMINI_TIMEOUT', 30, int),
                max_retries=ConfigurationFactory._get_env_value('GEMINI_MAX_RETRIES', 3, int)
            )
        
        # AWS Bedrock Configuration
        aws_access_key = ConfigurationFactory._get_env_value('AWS_ACCESS_KEY_ID')
        if aws_access_key:
            bedrock_models = ConfigurationFactory._get_env_value(
                'AWS_BEDROCK_MODELS',
                ['bedrock/amazon.nova-pro-v1:0', 'bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0'],
                list
            )
            default_bedrock_model = ConfigurationFactory._get_env_value('AWS_BEDROCK_MODEL', 'bedrock/amazon.nova-pro-v1:0')
            
            # Ensure the default model is in the models list
            if default_bedrock_model not in bedrock_models:
                bedrock_models.append(default_bedrock_model)
            
            providers['bedrock'] = LLMProviderConfig(
                provider='bedrock',
                api_key=aws_access_key,  # Using access key as api_key
                base_url=None,  # Bedrock doesn't use base_url
                models=bedrock_models,
                default_model=default_bedrock_model,
                enabled=True,
                timeout=ConfigurationFactory._get_env_value('AWS_BEDROCK_TIMEOUT', 30, int),
                max_retries=ConfigurationFactory._get_env_value('AWS_BEDROCK_MAX_RETRIES', 3, int)
            )
        
        # If no providers are configured, add a default OpenAI provider for basic functionality
        if not providers:
            providers['openai'] = LLMProviderConfig(
                provider='openai',
                api_key=None,  # Will need to be configured later
                base_url='https://api.openai.com/v1',
                models=['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'],
                default_model='gpt-4o',
                enabled=False,  # Disabled until API key is provided
                timeout=30,
                max_retries=3
            )
        
        return providers
    
    @staticmethod
    def build_embedding_config() -> Optional[EmbeddingConfig]:
        """Build embedding configuration from environment variables."""
        provider = ConfigurationFactory._get_env_value('EMBEDDING_PROVIDER', 'jina')
        
        if provider == 'jina':
            return EmbeddingConfig(
                provider='jina',
                model_name=ConfigurationFactory._get_env_value('JINA_EMBEDDING_MODEL', 'jina-embeddings-v3'),
                api_key=ConfigurationFactory._get_env_value('JINA_API_KEY'),
                api_url=ConfigurationFactory._get_env_value('JINA_API_URL', 'https://api.jina.ai/v1/embeddings'),
                dimensions=ConfigurationFactory._get_env_value('EMBEDDING_DIMENSION', 1024, int),
                batch_size=ConfigurationFactory._get_env_value('EMBEDDING_BATCH_SIZE', 100, int),
                timeout=ConfigurationFactory._get_env_value('EMBEDDING_TIMEOUT', 30, int),
                max_retries=ConfigurationFactory._get_env_value('JINA_MAX_RETRIES', 3, int)
            )
        elif provider == 'openai':
            return EmbeddingConfig(
                provider='openai',
                model_name=ConfigurationFactory._get_env_value('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-large'),
                api_key=ConfigurationFactory._get_env_value('OPENAI_API_KEY'),
                dimensions=ConfigurationFactory._get_env_value('OPENAI_EMBEDDING_DIMENSION', 3072, int),
                batch_size=ConfigurationFactory._get_env_value('EMBEDDING_BATCH_SIZE', 100, int),
                timeout=ConfigurationFactory._get_env_value('EMBEDDING_TIMEOUT', 30, int),
                max_retries=ConfigurationFactory._get_env_value('OPENAI_MAX_RETRIES', 3, int)
            )
        elif provider == 'local':
            return EmbeddingConfig(
                provider='local',
                model_name=ConfigurationFactory._get_env_value('LOCAL_EMBEDDING_MODEL', 'hf:ibm-granite/granite-embedding-30m-english'),
                dimensions=ConfigurationFactory._get_env_value('EMBEDDING_DIMENSION', 384, int),
                batch_size=ConfigurationFactory._get_env_value('EMBEDDING_BATCH_SIZE', 100, int),
                timeout=ConfigurationFactory._get_env_value('EMBEDDING_TIMEOUT', 30, int),
                device=ConfigurationFactory._get_env_value('LOCAL_EMBEDDING_DEVICE', 'cpu'),
                cache_dir=ConfigurationFactory._get_env_value('LOCAL_EMBEDDING_CACHE_DIR', 'models/embeddings')
            )
        elif provider == 'gemini':
            return EmbeddingConfig(
                provider='gemini',
                model_name=ConfigurationFactory._get_env_value('GEMINI_EMBEDDING_MODEL', 'text-embedding-004'),
                api_key=ConfigurationFactory._get_env_value('GEMINI_API_KEY'),
                dimensions=ConfigurationFactory._get_env_value('GEMINI_EMBEDDING_DIMENSION', 768, int),
                batch_size=ConfigurationFactory._get_env_value('EMBEDDING_BATCH_SIZE', 100, int),
                timeout=ConfigurationFactory._get_env_value('EMBEDDING_TIMEOUT', 30, int),
                max_retries=ConfigurationFactory._get_env_value('GEMINI_MAX_RETRIES', 3, int)
            )
        
        return None
    
    @staticmethod
    def build_vector_store_config() -> Optional[VectorStoreConfig]:
        """Build vector store configuration from environment variables."""
        provider = ConfigurationFactory._get_env_value('VECTOR_STORE_PROVIDER', 'astradb')
        
        connection_params = {}
        
        if provider == 'astradb':
            connection_params = {
                'api_endpoint': ConfigurationFactory._get_env_value('ASTRADB_API_ENDPOINT'),
                'application_token': ConfigurationFactory._get_env_value('ASTRADB_APPLICATION_TOKEN'),
                'keyspace': ConfigurationFactory._get_env_value('ASTRADB_KEYSPACE', 'default_keyspace'),
                'region': ConfigurationFactory._get_env_value('ASTRADB_REGION', 'us-east-2')
            }
        elif provider == 'chroma':
            connection_params = {
                'db_path': ConfigurationFactory._get_env_value('CHROMA_DB_PATH', 'data/rag/chroma_db'),
                'persist_directory': ConfigurationFactory._get_env_value('CHROMA_PERSIST_DIRECTORY', 'data/rag/chroma_persist')
            }
        elif provider == 'pinecone':
            connection_params = {
                'api_key': ConfigurationFactory._get_env_value('PINECONE_API_KEY'),
                'environment': ConfigurationFactory._get_env_value('PINECONE_ENVIRONMENT'),
                'index_name': ConfigurationFactory._get_env_value('PINECONE_INDEX_NAME')
            }
        
        return VectorStoreConfig(
            provider=provider,
            collection_name=ConfigurationFactory._get_env_value('VECTOR_STORE_COLLECTION', 'manim_docs_jina_1024'),
            connection_params=connection_params,
            max_results=ConfigurationFactory._get_env_value('VECTOR_STORE_MAX_RESULTS', 50, int),
            distance_metric=ConfigurationFactory._get_env_value('VECTOR_STORE_DISTANCE_METRIC', 'cosine'),
            timeout=ConfigurationFactory._get_env_value('ASTRADB_TIMEOUT', 30, int),
            max_retries=ConfigurationFactory._get_env_value('ASTRADB_MAX_RETRIES', 3, int)
        )
    
    @staticmethod
    def build_rag_config() -> Optional[RAGConfig]:
        """Build RAG configuration from environment variables."""
        if not ConfigurationFactory._get_env_value('RAG_ENABLED', True, bool):
            return None
        
        embedding_config = ConfigurationFactory.build_embedding_config()
        vector_store_config = ConfigurationFactory.build_vector_store_config()
        
        if not embedding_config or not vector_store_config:
            return None
        
        return RAGConfig(
            enabled=True,
            embedding_config=embedding_config,
            vector_store_config=vector_store_config,
            chunk_size=ConfigurationFactory._get_env_value('RAG_CHUNK_SIZE', 1000, int),
            chunk_overlap=ConfigurationFactory._get_env_value('RAG_CHUNK_OVERLAP', 200, int),
            min_chunk_size=ConfigurationFactory._get_env_value('RAG_MIN_CHUNK_SIZE', 100, int),
            default_k_value=ConfigurationFactory._get_env_value('RAG_DEFAULT_K_VALUE', 5, int),
            similarity_threshold=ConfigurationFactory._get_env_value('RAG_SIMILARITY_THRESHOLD', 0.7, float),
            enable_query_expansion=ConfigurationFactory._get_env_value('RAG_ENABLE_QUERY_EXPANSION', True, bool),
            enable_semantic_search=ConfigurationFactory._get_env_value('RAG_ENABLE_SEMANTIC_SEARCH', True, bool),
            enable_caching=ConfigurationFactory._get_env_value('RAG_ENABLE_CACHING', True, bool),
            cache_ttl=ConfigurationFactory._get_env_value('RAG_CACHE_TTL', 3600, int),
            max_cache_size=ConfigurationFactory._get_env_value('RAG_MAX_CACHE_SIZE', 1000, int),
            enable_quality_monitoring=ConfigurationFactory._get_env_value('RAG_ENABLE_QUALITY_MONITORING', True, bool),
            quality_threshold=ConfigurationFactory._get_env_value('RAG_QUALITY_THRESHOLD', 0.7, float)
        )
    
    @staticmethod
    def build_agent_configs() -> Dict[str, AgentConfig]:
        """Build agent configurations from environment variables."""
        agents = {}
        
        # Define default agent configurations compatible with langgraph system
        default_agents = {
            'planner_agent': {
                'llm_config': {
                    'temperature': 0.7,
                    'max_tokens': 4000,
                    'timeout': 300
                },
                'tools': ['scene_planning', 'plugin_detection'],
                'planner_model': 'openai/gpt-4o',
                'helper_model': 'openai/gpt-4o-mini',
                'temperature': 0.7,
                'system_prompt': 'You are a video planning agent that creates detailed plans for educational videos.'
            },
            'code_generator_agent': {
                'llm_config': {
                    'temperature': 0.7,
                    'max_tokens': 4000,
                    'timeout': 300
                },
                'tools': ['code_generation', 'error_fixing', 'rag_query'],
                'scene_model': 'openai/gpt-4o',
                'helper_model': 'openai/gpt-4o-mini',
                'temperature': 0.7,
                'enable_human_loop': True,
                'max_retries': 5,
                'timeout_seconds': 600,
                'system_prompt': 'You are a code generation agent that creates Manim animations.'
            },
            'renderer_agent': {
                'llm_config': {
                    'temperature': 0.3,
                    'max_tokens': 2000,
                    'timeout': 600
                },
                'tools': ['video_rendering', 'optimization'],
                'temperature': 0.3,
                'timeout_seconds': 1200,
                'system_prompt': 'You are a video rendering agent that optimizes and renders Manim animations.'
            },
            'visual_analysis_agent': {
                'llm_config': {
                    'temperature': 0.5,
                    'max_tokens': 3000,
                    'timeout': 300
                },
                'tools': ['visual_analysis', 'error_detection'],
                'helper_model': 'openai/gpt-4o',
                'temperature': 0.5,
                'enable_human_loop': True,
                'system_prompt': 'You are a visual analysis agent that detects errors in rendered videos.'
            },
            'rag_agent': {
                'llm_config': {
                    'temperature': 0.3,
                    'max_tokens': 2000,
                    'timeout': 180
                },
                'tools': ['rag_query', 'context_retrieval', 'document_processing'],
                'helper_model': 'openai/gpt-4o-mini',
                'temperature': 0.3,
                'timeout_seconds': 180,
                'max_retries': 2,
                'system_prompt': 'You are a RAG agent that retrieves relevant documentation and context.'
            },
            'error_handler_agent': {
                'llm_config': {
                    'temperature': 0.3,
                    'max_tokens': 2000,
                    'timeout': 120
                },
                'tools': ['error_classification', 'recovery_routing'],
                'helper_model': 'openai/gpt-4o-mini',
                'temperature': 0.3,
                'timeout_seconds': 120,
                'max_retries': 1,
                'enable_human_loop': True,
                'system_prompt': 'You are an error handling agent that classifies and routes error recovery.'
            },
            'monitoring_agent': {
                'llm_config': {
                    'temperature': 0.1,
                    'max_tokens': 1000,
                    'timeout': 60
                },
                'tools': ['performance_monitoring', 'diagnostics'],
                'temperature': 0.1,
                'timeout_seconds': 60,
                'max_retries': 1,
                'print_cost': False,
                'system_prompt': 'You are a monitoring agent that tracks system performance and diagnostics.'
            },
            'human_loop_agent': {
                'llm_config': {
                    'temperature': 0.5,
                    'max_tokens': 1500,
                    'timeout': 30
                },
                'tools': ['human_interaction', 'decision_presentation'],
                'temperature': 0.5,
                'timeout_seconds': 30,
                'max_retries': 1,
                'enable_human_loop': True,
                'print_cost': False,
                'verbose': True,
                'system_prompt': 'You are a human loop agent that facilitates human-AI interaction.'
            }
        }
        
        for agent_name, config in default_agents.items():
            agents[agent_name] = AgentConfig(
                name=agent_name,
                llm_config=config.get('llm_config', {}),
                tools=config.get('tools', []),
                max_retries=config.get('max_retries', 3),
                timeout_seconds=config.get('timeout_seconds', 300),
                enable_human_loop=config.get('enable_human_loop', False),
                planner_model=config.get('planner_model'),
                scene_model=config.get('scene_model'),
                helper_model=config.get('helper_model'),
                temperature=config.get('temperature', 0.7),
                print_cost=config.get('print_cost', True),
                verbose=config.get('verbose', False),
                enabled=ConfigurationFactory._get_env_value(f'AGENT_{agent_name.upper()}_ENABLED', True, bool),
                system_prompt=config.get('system_prompt')
            )
        
        return agents
    
    @staticmethod
    def build_monitoring_config() -> MonitoringConfig:
        """Build monitoring configuration from environment variables."""
        langfuse_config = None
        
        if ConfigurationFactory._get_env_value('LANGFUSE_SECRET_KEY'):
            langfuse_config = LangfuseConfig(
                enabled=ConfigurationFactory._get_env_value('MONITORING_ENABLED', True, bool),
                secret_key=ConfigurationFactory._get_env_value('LANGFUSE_SECRET_KEY'),
                public_key=ConfigurationFactory._get_env_value('LANGFUSE_PUBLIC_KEY'),
                host=ConfigurationFactory._get_env_value('LANGFUSE_HOST', 'https://cloud.langfuse.com')
            )
        
        return MonitoringConfig(
            enabled=ConfigurationFactory._get_env_value('MONITORING_ENABLED', True, bool),
            langfuse_config=langfuse_config,
            log_level=ConfigurationFactory._get_env_value('LOG_LEVEL', 'INFO'),
            metrics_collection_interval=ConfigurationFactory._get_env_value('METRICS_COLLECTION_INTERVAL', 300, int)
        )
    
    @staticmethod
    def build_docling_config() -> DoclingConfig:
        """Build Docling configuration from environment variables."""
        supported_formats_str = ConfigurationFactory._get_env_value('DOCLING_SUPPORTED_FORMATS', 'pdf,docx,txt,md')
        supported_formats = [fmt.strip() for fmt in supported_formats_str.split(',') if fmt.strip()]
        
        return DoclingConfig(
            enabled=ConfigurationFactory._get_env_value('DOCLING_ENABLED', True, bool),
            max_file_size_mb=ConfigurationFactory._get_env_value('DOCLING_MAX_FILE_SIZE_MB', 50, int),
            supported_formats=supported_formats,
            timeout_seconds=ConfigurationFactory._get_env_value('DOCLING_TIMEOUT_SECONDS', 120, int)
        )
    
    @staticmethod
    def build_mcp_servers() -> Dict[str, MCPServerConfig]:
        """Build MCP server configurations from environment variables."""
        servers = {}
        
        # Context7 MCP server
        servers['context7'] = MCPServerConfig(
            command='uvx',
            args=['context7-mcp-server@latest'],
            env={'FASTMCP_LOG_LEVEL': 'ERROR'},
            disabled=ConfigurationFactory._get_env_value('MCP_CONTEXT7_DISABLED', False, bool),
            auto_approve=['resolve_library_id', 'get_library_docs']
        )
        
        # Docling MCP server
        servers['docling'] = MCPServerConfig(
            command='uvx',
            args=['docling-mcp-server@latest'],
            env={'FASTMCP_LOG_LEVEL': 'ERROR'},
            disabled=ConfigurationFactory._get_env_value('MCP_DOCLING_DISABLED', False, bool),
            auto_approve=['process_document']
        )
        
        return servers
    
    @staticmethod
    def build_context7_config() -> Context7Config:
        """Build Context7 configuration from environment variables."""
        return Context7Config(
            enabled=ConfigurationFactory._get_env_value('CONTEXT7_ENABLED', True, bool),
            default_tokens=ConfigurationFactory._get_env_value('CONTEXT7_DEFAULT_TOKENS', 10000, int),
            timeout_seconds=ConfigurationFactory._get_env_value('CONTEXT7_TIMEOUT_SECONDS', 30, int),
            cache_responses=ConfigurationFactory._get_env_value('CONTEXT7_CACHE_RESPONSES', True, bool),
            cache_ttl=ConfigurationFactory._get_env_value('CONTEXT7_CACHE_TTL', 3600, int)
        )
    
    @staticmethod
    def build_human_loop_config() -> HumanLoopConfig:
        """Build human loop configuration from environment variables."""
        return HumanLoopConfig(
            enabled=ConfigurationFactory._get_env_value('HUMAN_LOOP_ENABLED', True, bool),
            enable_interrupts=ConfigurationFactory._get_env_value('HUMAN_LOOP_ENABLE_INTERRUPTS', True, bool),
            timeout_seconds=ConfigurationFactory._get_env_value('HUMAN_LOOP_TIMEOUT_SECONDS', 300, int),
            auto_approve_low_risk=ConfigurationFactory._get_env_value('HUMAN_LOOP_AUTO_APPROVE_LOW_RISK', False, bool)
        )
    
    @staticmethod
    def build_workflow_config() -> WorkflowConfig:
        """Build workflow configuration from environment variables."""
        return WorkflowConfig(
            max_workflow_retries=ConfigurationFactory._get_env_value('MAX_WORKFLOW_RETRIES', 3, int),
            workflow_timeout_seconds=ConfigurationFactory._get_env_value('WORKFLOW_TIMEOUT_SECONDS', 3600, int),
            enable_checkpoints=ConfigurationFactory._get_env_value('ENABLE_CHECKPOINTS', True, bool),
            checkpoint_interval=ConfigurationFactory._get_env_value('CHECKPOINT_INTERVAL', 300, int),
            output_dir=ConfigurationFactory._get_env_value('OUTPUT_DIR', 'output'),
            max_scene_concurrency=ConfigurationFactory._get_env_value('MAX_SCENE_CONCURRENCY', 5, int),
            max_topic_concurrency=ConfigurationFactory._get_env_value('MAX_TOPIC_CONCURRENCY', 1, int),
            max_concurrent_renders=ConfigurationFactory._get_env_value('MAX_CONCURRENT_RENDERS', 4, int),
            default_quality=ConfigurationFactory._get_env_value('DEFAULT_QUALITY', 'medium'),
            use_gpu_acceleration=ConfigurationFactory._get_env_value('USE_GPU_ACCELERATION', False, bool),
            preview_mode=ConfigurationFactory._get_env_value('PREVIEW_MODE', False, bool)
        )
    
    @staticmethod
    def build_system_config() -> SystemConfig:
        """Build complete system configuration from environment variables."""
        return SystemConfig(
            environment=ConfigurationFactory._get_env_value('ENVIRONMENT', 'development'),
            debug=ConfigurationFactory._get_env_value('DEBUG', False, bool),
            default_llm_provider=ConfigurationFactory._get_env_value('DEFAULT_LLM_PROVIDER', 'openai'),
            llm_providers=ConfigurationFactory.build_llm_provider_configs(),
            rag_config=ConfigurationFactory.build_rag_config(),
            agent_configs=ConfigurationFactory.build_agent_configs(),
            docling_config=ConfigurationFactory.build_docling_config(),
            mcp_servers=ConfigurationFactory.build_mcp_servers(),
            context7_config=ConfigurationFactory.build_context7_config(),
            monitoring_config=ConfigurationFactory.build_monitoring_config(),
            human_loop_config=ConfigurationFactory.build_human_loop_config(),
            workflow_config=ConfigurationFactory.build_workflow_config(),
            kokoro_model_path=ConfigurationFactory._get_env_value('KOKORO_MODEL_PATH'),
            kokoro_voices_path=ConfigurationFactory._get_env_value('KOKORO_VOICES_PATH'),
            kokoro_default_voice=ConfigurationFactory._get_env_value('KOKORO_DEFAULT_VOICE', 'af'),
            kokoro_default_speed=ConfigurationFactory._get_env_value('KOKORO_DEFAULT_SPEED', 1.0, float),
            kokoro_default_lang=ConfigurationFactory._get_env_value('KOKORO_DEFAULT_LANG', 'en-us')
        )
    
    @staticmethod
    def create_minimal_config() -> SystemConfig:
        """Create a minimal working configuration as fallback.
        
        Returns:
            Minimal SystemConfig that should always work
        """
        config_logger = get_config_logger()
        
        try:
            config_logger.warning("Creating minimal configuration as fallback")
            
            # Create minimal LLM provider config
            minimal_providers = {}
            
            # Check for available API keys
            if os.getenv('OPENAI_API_KEY'):
                minimal_providers['openai'] = LLMProviderConfig(
                    provider='openai',
                    api_key=os.getenv('OPENAI_API_KEY'),
                    models=['gpt-4o-mini'],
                    default_model='gpt-4o-mini',
                    enabled=True
                )
            elif os.getenv('GEMINI_API_KEY'):
                minimal_providers['gemini'] = LLMProviderConfig(
                    provider='gemini',
                    api_key=os.getenv('GEMINI_API_KEY'),
                    models=['gemini-1.5-flash'],
                    default_model='gemini-1.5-flash',
                    enabled=True
                )
            else:
                # No API keys available - create disabled local provider
                minimal_providers['local'] = LLMProviderConfig(
                    provider='local',
                    api_key=None,
                    models=['local-model'],
                    default_model='local-model',
                    enabled=False
                )
            
            # Determine default provider
            default_provider = list(minimal_providers.keys())[0]
            
            # Create minimal agent configs
            minimal_agents = {}
            available_model = f"{default_provider}/{minimal_providers[default_provider].default_model}"
            
            for agent_name in ['planner_agent', 'code_generator_agent', 'renderer_agent']:
                minimal_agents[agent_name] = AgentConfig(
                    name=agent_name,
                    planner_model=available_model,
                    scene_model=available_model,
                    helper_model=available_model,
                    enabled=minimal_providers[default_provider].enabled,
                    timeout_seconds=180,
                    max_retries=2,
                    temperature=0.7
                )
            
            # Create minimal RAG config (disabled)
            minimal_rag = RAGConfig(
                enabled=False,
                embedding_config=EmbeddingConfig(
                    provider='local',
                    model_name='local-embeddings',
                    dimensions=384
                ),
                vector_store_config=VectorStoreConfig(
                    provider='chroma',
                    collection_name='minimal_collection',
                    connection_params={'persist_directory': './minimal_chroma_db'}
                )
            )
            
            # Create minimal monitoring config
            minimal_monitoring = MonitoringConfig(
                enabled=False,
                langfuse_config=None
            )
            
            # Create minimal workflow config
            minimal_workflow = WorkflowConfig(
                max_workflow_retries=2,
                workflow_timeout_seconds=1800,
                output_dir='output',
                max_scene_concurrency=1,
                max_topic_concurrency=1,
                max_concurrent_renders=1,
                preview_mode=True
            )
            
            # Create the minimal system config
            minimal_config = SystemConfig(
                environment='development',
                debug=True,
                default_llm_provider=default_provider,
                llm_providers=minimal_providers,
                rag_config=minimal_rag,
                agent_configs=minimal_agents,
                monitoring_config=minimal_monitoring,
                workflow_config=minimal_workflow,
                human_loop_config=HumanLoopConfig(enabled=True),
                docling_config=DoclingConfig(enabled=False),
                context7_config=Context7Config(enabled=False),
                mcp_servers={}
            )
            
            config_logger.info("Successfully created minimal configuration")
            return minimal_config
            
        except Exception as e:
            handle_config_error(
                category=ConfigErrorCategory.LOADING,
                severity=ConfigErrorSeverity.CRITICAL,
                message=f"Failed to create minimal configuration: {str(e)}",
                component='configuration_factory',
                exception=e,
                suggested_fix="Check system dependencies and environment setup"
            )
            
            # Absolute last resort - create truly minimal config
            return SystemConfig(
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
                        enabled=False
                    )
                }
            )