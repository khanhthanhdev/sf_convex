"""
Core configuration models using Pydantic for the centralized configuration system.

This module defines all the configuration models that support environment variable
loading with nested delimiter support and proper validation.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Literal, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class LLMProviderConfig(BaseModel):
    """Configuration for an LLM provider."""
    
    provider: str = Field(..., description="Provider name (e.g., 'openai', 'anthropic')")
    api_key: Optional[str] = Field(None, description="API key for the provider")
    base_url: Optional[str] = Field(None, description="Base URL for API calls")
    models: List[str] = Field(default_factory=list, description="Available models for this provider")
    default_model: str = Field(..., description="Default model to use")
    enabled: bool = Field(True, description="Whether this provider is enabled")
    timeout: int = Field(30, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retries")
    
    @field_validator('models')
    @classmethod
    def validate_models_not_empty(cls, v):
        if not v:
            raise ValueError("At least one model must be specified")
        return v
    
    @model_validator(mode='after')
    def validate_default_model_in_models(self):
        if self.default_model not in self.models:
            raise ValueError(f"Default model '{self.default_model}' must be in the models list")
        return self


class EmbeddingConfig(BaseModel):
    """Configuration for embedding providers."""
    
    model_config = ConfigDict(protected_namespaces=())
    
    provider: Literal["jina", "gemini", "local", "openai"] = Field(..., description="Embedding provider")
    model_name: str = Field(..., description="Model name for embeddings")
    api_key: Optional[str] = Field(None, description="API key for the provider")
    api_url: Optional[str] = Field(None, description="API URL for the provider")
    dimensions: int = Field(..., description="Embedding dimensions")
    batch_size: int = Field(100, description="Batch size for processing")
    timeout: int = Field(30, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retries")
    device: Optional[str] = Field("cpu", description="Device for local embeddings")
    cache_dir: Optional[str] = Field(None, description="Cache directory for local models")
    
    @field_validator('dimensions')
    @classmethod
    def validate_dimensions_positive(cls, v):
        if v <= 0:
            raise ValueError("Dimensions must be positive")
        return v
    
    @field_validator('batch_size')
    @classmethod
    def validate_batch_size_positive(cls, v):
        if v <= 0:
            raise ValueError("Batch size must be positive")
        return v


class VectorStoreConfig(BaseModel):
    """Configuration for vector store providers."""
    
    provider: Literal["astradb", "chroma", "pinecone"] = Field(..., description="Vector store provider")
    collection_name: str = Field(..., description="Collection/index name")
    connection_params: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific connection parameters")
    max_results: int = Field(50, description="Maximum results to return")
    distance_metric: str = Field("cosine", description="Distance metric for similarity")
    timeout: int = Field(30, description="Connection timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retries")
    
    @field_validator('max_results')
    @classmethod
    def validate_max_results_positive(cls, v):
        if v <= 0:
            raise ValueError("Max results must be positive")
        return v


class RAGConfig(BaseModel):
    """Configuration for RAG (Retrieval-Augmented Generation) system."""
    
    enabled: bool = Field(True, description="Whether RAG is enabled")
    embedding_config: EmbeddingConfig = Field(..., description="Embedding configuration")
    vector_store_config: VectorStoreConfig = Field(..., description="Vector store configuration")
    
    # Document processing settings
    chunk_size: int = Field(1000, description="Document chunk size")
    chunk_overlap: int = Field(200, description="Chunk overlap size")
    min_chunk_size: int = Field(100, description="Minimum chunk size")
    
    # Query processing settings
    default_k_value: int = Field(5, description="Default number of results to retrieve")
    similarity_threshold: float = Field(0.7, description="Similarity threshold for results")
    enable_query_expansion: bool = Field(True, description="Enable query expansion")
    enable_semantic_search: bool = Field(True, description="Enable semantic search")
    
    # Performance settings
    enable_caching: bool = Field(True, description="Enable result caching")
    cache_ttl: int = Field(3600, description="Cache TTL in seconds")
    max_cache_size: int = Field(1000, description="Maximum cache size")
    
    # Quality monitoring
    enable_quality_monitoring: bool = Field(True, description="Enable quality monitoring")
    quality_threshold: float = Field(0.7, description="Quality threshold")
    
    @field_validator('chunk_size', 'chunk_overlap', 'min_chunk_size')
    @classmethod
    def validate_positive_sizes(cls, v):
        if v <= 0:
            raise ValueError("Size values must be positive")
        return v
    
    @field_validator('similarity_threshold', 'quality_threshold')
    @classmethod
    def validate_threshold_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        return v
    
    @model_validator(mode='after')
    def validate_overlap_less_than_chunk_size(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        return self


class AgentConfig(BaseModel):
    """Configuration for individual agents with LangGraph compatibility."""
    
    name: str = Field(..., description="Agent name")
    llm_config: Dict[str, Any] = Field(default_factory=dict, description="LLM configuration dict")
    tools: List[str] = Field(default_factory=list, description="Available tools for the agent")
    max_retries: int = Field(3, description="Maximum number of retries")
    timeout_seconds: int = Field(300, description="Timeout in seconds")
    enable_human_loop: bool = Field(False, description="Whether to enable human loop")
    
    # LLM Provider configurations - compatible with existing structure
    planner_model: Optional[str] = Field(None, description="Planner model (format: provider/model)")
    scene_model: Optional[str] = Field(None, description="Scene model (format: provider/model)")
    helper_model: Optional[str] = Field(None, description="Helper model (format: provider/model)")
    
    # Model wrapper settings - preserving existing patterns
    temperature: float = Field(0.7, description="Temperature for generation")
    print_cost: bool = Field(True, description="Whether to print cost information")
    verbose: bool = Field(False, description="Verbose output")
    
    # Additional agent settings
    enabled: bool = Field(True, description="Whether this agent is enabled")
    system_prompt: Optional[str] = Field(None, description="System prompt for the agent")
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature_range(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v
    
    @field_validator('timeout_seconds')
    @classmethod
    def validate_timeout_positive(cls, v):
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v


class LangfuseConfig(BaseModel):
    """Configuration for Langfuse monitoring."""
    
    enabled: bool = Field(True, description="Whether Langfuse is enabled")
    secret_key: Optional[str] = Field(None, description="Langfuse secret key")
    public_key: Optional[str] = Field(None, description="Langfuse public key")
    host: str = Field("https://cloud.langfuse.com", description="Langfuse host URL")
    
    @model_validator(mode='after')
    def validate_keys_if_enabled(self):
        if self.enabled and (not self.secret_key or not self.public_key):
            raise ValueError("Secret key and public key are required when Langfuse is enabled")
        return self


class MonitoringConfig(BaseModel):
    """Configuration for system monitoring."""
    
    enabled: bool = Field(True, description="Whether monitoring is enabled")
    langfuse_config: Optional[LangfuseConfig] = Field(None, description="Langfuse configuration")
    log_level: str = Field("INFO", description="Logging level")
    metrics_collection_interval: int = Field(300, description="Metrics collection interval in seconds")
    
    # Performance tracking settings
    performance_tracking: bool = Field(True, description="Whether to track performance metrics")
    error_tracking: bool = Field(True, description="Whether to track errors")
    execution_tracing: bool = Field(True, description="Whether to trace execution")
    
    # Performance thresholds
    cpu_threshold: float = Field(80.0, description="CPU usage threshold percentage")
    memory_threshold: float = Field(85.0, description="Memory usage threshold percentage")
    execution_time_threshold: float = Field(300.0, description="Execution time threshold in seconds")
    
    # History retention
    history_retention_hours: int = Field(24, description="History retention in hours")
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @field_validator('cpu_threshold', 'memory_threshold')
    @classmethod
    def validate_percentage_thresholds(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Threshold must be between 0 and 100")
        return v


class DoclingConfig(BaseModel):
    """Configuration for Docling document processing."""
    
    enabled: bool = Field(True, description="Whether Docling is enabled")
    max_file_size_mb: int = Field(50, description="Maximum file size in MB")
    supported_formats: List[str] = Field(default_factory=lambda: ["pdf", "docx", "txt", "md"], description="Supported file formats")
    timeout_seconds: int = Field(120, description="Processing timeout in seconds")
    
    @field_validator('max_file_size_mb')
    @classmethod
    def validate_file_size_positive(cls, v):
        if v <= 0:
            raise ValueError("Max file size must be positive")
        return v


class MCPServerConfig(BaseModel):
    """Configuration for MCP (Model Context Protocol) servers."""
    
    command: str = Field(..., description="Command to run the server")
    args: List[str] = Field(default_factory=list, description="Command arguments")
    env: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    disabled: bool = Field(False, description="Whether the server is disabled")
    auto_approve: List[str] = Field(default_factory=list, description="Auto-approved tool names")


class Context7Config(BaseModel):
    """Configuration for Context7 library documentation service."""
    
    enabled: bool = Field(True, description="Whether Context7 is enabled")
    default_tokens: int = Field(10000, description="Default token limit")
    timeout_seconds: int = Field(30, description="Request timeout in seconds")
    cache_responses: bool = Field(True, description="Whether to cache responses")
    cache_ttl: int = Field(3600, description="Cache TTL in seconds")


class HumanLoopConfig(BaseModel):
    """Configuration for human-in-the-loop functionality."""
    
    enabled: bool = Field(True, description="Whether human loop is enabled")
    enable_interrupts: bool = Field(True, description="Whether to enable interrupts")
    timeout_seconds: int = Field(300, description="Human response timeout in seconds")
    auto_approve_low_risk: bool = Field(False, description="Auto-approve low-risk operations")


class WorkflowConfig(BaseModel):
    """Configuration for workflow execution."""
    
    max_workflow_retries: int = Field(3, description="Maximum workflow retries")
    workflow_timeout_seconds: int = Field(3600, description="Workflow timeout in seconds")
    enable_checkpoints: bool = Field(True, description="Whether to enable checkpoints")
    checkpoint_interval: int = Field(300, description="Checkpoint interval in seconds")
    
    # Video generation specific settings
    output_dir: str = Field("output", description="Output directory")
    max_scene_concurrency: int = Field(5, description="Maximum concurrent scenes")
    max_topic_concurrency: int = Field(1, description="Maximum concurrent topics")
    max_concurrent_renders: int = Field(4, description="Maximum concurrent renders")
    
    # Quality and performance settings
    default_quality: str = Field("medium", description="Default video quality")
    use_gpu_acceleration: bool = Field(False, description="Whether to use GPU acceleration")
    preview_mode: bool = Field(False, description="Whether to run in preview mode")
    
    @field_validator('max_workflow_retries', 'max_scene_concurrency', 'max_topic_concurrency', 'max_concurrent_renders')
    @classmethod
    def validate_positive_values(cls, v):
        if v <= 0:
            raise ValueError("Value must be positive")
        return v
    
    @field_validator('default_quality')
    @classmethod
    def validate_quality(cls, v):
        valid_qualities = ["low", "medium", "high", "ultra"]
        if v not in valid_qualities:
            raise ValueError(f"Quality must be one of: {valid_qualities}")
        return v


class SystemConfig(BaseSettings):
    """Main system configuration that loads from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        env_nested_delimiter='__',
        case_sensitive=False,
        extra='ignore'  # Ignore extra environment variables
    )
    
    # Core system settings
    environment: str = Field("development", description="Environment (development, staging, production)")
    debug: bool = Field(False, description="Debug mode")
    
    # LLM Configuration
    default_llm_provider: str = Field("openai", description="Default LLM provider")
    llm_providers: Dict[str, LLMProviderConfig] = Field(default_factory=dict, description="LLM provider configurations")
    
    # RAG Configuration
    rag_config: Optional[RAGConfig] = Field(None, description="RAG system configuration")
    
    # Agent Configuration
    agent_configs: Dict[str, AgentConfig] = Field(default_factory=dict, description="Agent configurations")
    
    # External tool configurations
    docling_config: DoclingConfig = Field(default_factory=DoclingConfig, description="Docling configuration")
    mcp_servers: Dict[str, MCPServerConfig] = Field(default_factory=dict, description="MCP server configurations")
    context7_config: Context7Config = Field(default_factory=Context7Config, description="Context7 configuration")
    
    # System monitoring
    monitoring_config: MonitoringConfig = Field(default_factory=MonitoringConfig, description="Monitoring configuration")
    human_loop_config: HumanLoopConfig = Field(default_factory=HumanLoopConfig, description="Human loop configuration")
    
    # Workflow settings
    workflow_config: WorkflowConfig = Field(default_factory=WorkflowConfig, description="Workflow configuration")
    
    # TTS Configuration (Kokoro)
    kokoro_model_path: Optional[str] = Field(None, description="Path to Kokoro TTS model")
    kokoro_voices_path: Optional[str] = Field(None, description="Path to Kokoro voices file")
    kokoro_default_voice: str = Field("af", description="Default voice for TTS")
    kokoro_default_speed: float = Field(1.0, description="Default TTS speed")
    kokoro_default_lang: str = Field("en-us", description="Default TTS language")
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v):
        valid_envs = ["development", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of: {valid_envs}")
        return v
    
    @field_validator('kokoro_default_speed')
    @classmethod
    def validate_speed_range(cls, v):
        if not 0.1 <= v <= 3.0:
            raise ValueError("TTS speed must be between 0.1 and 3.0")
        return v
    
    @model_validator(mode='after')
    def validate_default_provider_exists(self):
        if self.llm_providers and self.default_llm_provider and self.default_llm_provider not in self.llm_providers:
            raise ValueError(f"Default LLM provider '{self.default_llm_provider}' not found in configured providers")
        return self


class ValidationResult(BaseModel):
    """Result of configuration validation."""
    
    valid: bool = Field(..., description="Whether the configuration is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    timestamp: datetime = Field(default_factory=datetime.now, description="Validation timestamp")
    
    def add_error(self, error: str):
        """Add an error to the validation result."""
        self.errors.append(error)
        self.valid = False
    
    def add_warning(self, warning: str):
        """Add a warning to the validation result."""
        self.warnings.append(warning)
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0