"""
RAG (Retrieval-Augmented Generation) module.

This module provides components for document retrieval, embedding generation,
vector storage, and integration with various embedding and vector store providers.
"""

from .embedding_providers import (
    EmbeddingProvider,
    EmbeddingConfig,
    EmbeddingGenerationError,
    ProviderConfigurationError,
    ConfigurationManager
)

from .jina_embedding_provider import (
    JinaEmbeddingProvider,
    AsyncJinaEmbeddingProvider,
    RateLimitInfo
)

from .provider_factory import (
    EmbeddingProviderFactory,
    create_embedding_provider,
    get_default_provider,
    test_all_providers
)

from .vector_store_providers import (
    VectorStoreProvider,
    VectorStoreConfig,
    VectorStoreError,
    VectorStoreConnectionError,
    VectorStoreOperationError,
    VectorStoreConfigurationError,
    SearchResult,
    DocumentInput
)

from .vector_store_factory import (
    VectorStoreFactory,
    create_vector_store,
    get_default_vector_store,
    test_vector_store_providers
)

__all__ = [
    # Embedding providers
    'EmbeddingProvider',
    'EmbeddingConfig', 
    'EmbeddingGenerationError',
    'ProviderConfigurationError',
    'ConfigurationManager',
    'JinaEmbeddingProvider',
    'AsyncJinaEmbeddingProvider',
    'RateLimitInfo',
    'EmbeddingProviderFactory',
    'create_embedding_provider',
    'get_default_provider',
    'test_all_providers',
    
    # Vector store providers
    'VectorStoreProvider',
    'VectorStoreConfig',
    'VectorStoreError',
    'VectorStoreConnectionError',
    'VectorStoreOperationError',
    'VectorStoreConfigurationError',
    'SearchResult',
    'DocumentInput',
    'VectorStoreFactory',
    'create_vector_store',
    'get_default_vector_store',
    'test_vector_store_providers'
]

# Import AstraDB providers (optional, will be registered if available)
try:
    from .astradb_vector_store import AstraDBVectorStore, AstraDBWithFallback
    from .astradb_fallback_handler import AstraDBFallbackHandler, FallbackConfig
    __all__.extend([
        'AstraDBVectorStore',
        'AstraDBWithFallback', 
        'AstraDBFallbackHandler',
        'FallbackConfig'
    ])
except ImportError:
    pass