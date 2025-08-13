"""
MCP (Model Context Protocol) Integration Module

This module provides MCP client functionality for connecting to and managing
MCP servers, with specific support for Context7 integration with Manim Community documentation.
"""

from .client import (
    MCPClient,
    MCPConnectionError,
    MCPServerNotFoundError,
    create_mcp_client,
    mcp_client_context,
    test_mcp_connection
)

from .context7_docs import (
    Context7DocsRetriever,
    DocumentationError,
    LibraryNotFoundError,
    CodeSnippet,
    DocumentationSection,
    DocumentationResponse,
    resolve_manim_library_id,
    get_manim_docs,
    extract_manim_code_examples
)

from .agent import (
    MCPAgent,
    AgentError,
    LLMProviderError,
    AgentConfig,
    AgentResponse,
    create_manim_agent,
    quick_manim_query
)

from .cache import (
    MemoryEfficientCache,
    DocumentationCache,
    ConnectionPool,
    CacheStats,
    get_documentation_cache,
    get_connection_pool,
    initialize_caching,
    shutdown_caching
)

from .optimization import (
    PerformanceMonitor,
    OptimizationManager,
    OptimizationLevel,
    PerformanceMetrics,
    OptimizationRecommendation,
    get_optimization_manager,
    optimize_mcp_system
)

__all__ = [
    'MCPClient',
    'MCPConnectionError', 
    'MCPServerNotFoundError',
    'create_mcp_client',
    'mcp_client_context',
    'test_mcp_connection',
    'Context7DocsRetriever',
    'DocumentationError',
    'LibraryNotFoundError',
    'CodeSnippet',
    'DocumentationSection',
    'DocumentationResponse',
    'resolve_manim_library_id',
    'get_manim_docs',
    'extract_manim_code_examples',
    'MCPAgent',
    'AgentError',
    'LLMProviderError',
    'AgentConfig',
    'AgentResponse',
    'create_manim_agent',
    'quick_manim_query',
    'MemoryEfficientCache',
    'DocumentationCache',
    'ConnectionPool',
    'CacheStats',
    'get_documentation_cache',
    'get_connection_pool',
    'initialize_caching',
    'shutdown_caching',
    'PerformanceMonitor',
    'OptimizationManager',
    'OptimizationLevel',
    'PerformanceMetrics',
    'OptimizationRecommendation',
    'get_optimization_manager',
    'optimize_mcp_system'
]