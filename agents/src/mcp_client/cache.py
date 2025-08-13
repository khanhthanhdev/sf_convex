"""
Caching and Optimization Features for MCP Client

This module provides comprehensive caching functionality for MCP operations including:
- Documentation caching with TTL management
- Connection pooling for MCP servers
- Memory-efficient documentation storage
- Cache invalidation and performance optimization
"""

import asyncio
import hashlib
import json
import logging
import time
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from collections import OrderedDict
from threading import RLock
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached item with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl_seconds: Optional[float] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds
    
    def touch(self) -> None:
        """Update last accessed time and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            **asdict(self),
            'hit_rate': self.hit_rate
        }


class CacheEvictionPolicy(ABC):
    """Abstract base class for cache eviction policies."""
    
    @abstractmethod
    def select_for_eviction(self, entries: Dict[str, CacheEntry]) -> List[str]:
        """Select cache entries for eviction."""
        pass


class LRUEvictionPolicy(CacheEvictionPolicy):
    """Least Recently Used eviction policy."""
    
    def select_for_eviction(self, entries: Dict[str, CacheEntry]) -> List[str]:
        """Select least recently used entries for eviction."""
        if not entries:
            return []
        
        # Sort by last accessed time (oldest first)
        sorted_entries = sorted(
            entries.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Return keys of oldest entries (up to 25% of cache)
        evict_count = max(1, len(entries) // 4)
        return [key for key, _ in sorted_entries[:evict_count]]


class LFUEvictionPolicy(CacheEvictionPolicy):
    """Least Frequently Used eviction policy."""
    
    def select_for_eviction(self, entries: Dict[str, CacheEntry]) -> List[str]:
        """Select least frequently used entries for eviction."""
        if not entries:
            return []
        
        # Sort by access count (lowest first)
        sorted_entries = sorted(
            entries.items(),
            key=lambda x: x[1].access_count
        )
        
        # Return keys of least used entries (up to 25% of cache)
        evict_count = max(1, len(entries) // 4)
        return [key for key, _ in sorted_entries[:evict_count]]


class MemoryEfficientCache:
    """
    Memory-efficient cache with TTL, size limits, and configurable eviction policies.
    
    Features:
    - TTL-based expiration
    - Size-based eviction
    - LRU/LFU eviction policies
    - Memory usage tracking
    - Performance statistics
    """
    
    def __init__(
        self,
        max_size_bytes: int = 100 * 1024 * 1024,  # 100MB default
        max_entries: int = 10000,
        default_ttl_seconds: float = 3600,  # 1 hour default
        eviction_policy: CacheEvictionPolicy = None
    ):
        """
        Initialize memory-efficient cache.
        
        Args:
            max_size_bytes: Maximum cache size in bytes
            max_entries: Maximum number of cache entries
            default_ttl_seconds: Default TTL for cache entries
            eviction_policy: Cache eviction policy (defaults to LRU)
        """
        self.max_size_bytes = max_size_bytes
        self.max_entries = max_entries
        self.default_ttl_seconds = default_ttl_seconds
        self.eviction_policy = eviction_policy or LRUEvictionPolicy()
        
        self._entries: Dict[str, CacheEntry] = {}
        self._stats = CacheStats()
        self._lock = RLock()
        
        logger.info(f"Initialized cache with max_size={max_size_bytes} bytes, "
                   f"max_entries={max_entries}, default_ttl={default_ttl_seconds}s")
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (dict, list)):
                return len(json.dumps(value, default=str).encode('utf-8'))
            elif hasattr(value, '__sizeof__'):
                return value.__sizeof__()
            else:
                return len(str(value).encode('utf-8'))
        except Exception:
            # Fallback to string representation size
            return len(str(value).encode('utf-8'))
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self._entries.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
            logger.debug(f"Removed expired cache entry: {key}")
    
    def _enforce_size_limits(self) -> None:
        """Enforce cache size and entry count limits."""
        # Remove expired entries first
        self._cleanup_expired()
        
        # Check if we need to evict entries
        while (self._stats.total_size_bytes > self.max_size_bytes or 
               self._stats.entry_count > self.max_entries):
            
            if not self._entries:
                break
            
            # Use eviction policy to select entries for removal
            keys_to_evict = self.eviction_policy.select_for_eviction(self._entries)
            
            if not keys_to_evict:
                # Fallback: remove oldest entry
                oldest_key = min(self._entries.keys(), 
                               key=lambda k: self._entries[k].last_accessed)
                keys_to_evict = [oldest_key]
            
            for key in keys_to_evict:
                if key in self._entries:
                    self._remove_entry(key)
                    self._stats.evictions += 1
                    logger.debug(f"Evicted cache entry: {key}")
                
                # Check if we've freed enough space
                if (self._stats.total_size_bytes <= self.max_size_bytes and 
                    self._stats.entry_count <= self.max_entries):
                    break
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache and update stats."""
        if key in self._entries:
            entry = self._entries[key]
            self._stats.total_size_bytes -= entry.size_bytes
            self._stats.entry_count -= 1
            del self._entries[key]
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._entries:
                self._stats.misses += 1
                return None
            
            entry = self._entries[key]
            
            # Check if expired
            if entry.is_expired():
                self._remove_entry(key)
                self._stats.misses += 1
                return None
            
            # Update access info
            entry.touch()
            self._stats.hits += 1
            
            return entry.value
    
    def put(
        self, 
        key: str, 
        value: Any, 
        ttl_seconds: Optional[float] = None
    ) -> None:
        """
        Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: TTL for this entry (uses default if None)
        """
        with self._lock:
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
            size_bytes = self._calculate_size(value)
            current_time = time.time()
            
            # Remove existing entry if present
            if key in self._entries:
                self._remove_entry(key)
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                access_count=1,
                ttl_seconds=ttl,
                size_bytes=size_bytes
            )
            
            # Add to cache
            self._entries[key] = entry
            self._stats.total_size_bytes += size_bytes
            self._stats.entry_count += 1
            
            # Enforce limits
            self._enforce_size_limits()
    
    def invalidate(self, key: str) -> bool:
        """
        Invalidate specific cache entry.
        
        Args:
            key: Cache key to invalidate
            
        Returns:
            True if entry was removed, False if not found
        """
        with self._lock:
            if key in self._entries:
                self._remove_entry(key)
                return True
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate cache entries matching pattern.
        
        Args:
            pattern: Pattern to match (simple substring match)
            
        Returns:
            Number of entries invalidated
        """
        with self._lock:
            keys_to_remove = [
                key for key in self._entries.keys() 
                if pattern in key
            ]
            
            for key in keys_to_remove:
                self._remove_entry(key)
            
            return len(keys_to_remove)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._entries.clear()
            self._stats = CacheStats()
    
    def get_stats(self) -> CacheStats:
        """Get cache performance statistics."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                total_size_bytes=self._stats.total_size_bytes,
                entry_count=self._stats.entry_count
            )
    
    def cache_decorator(self, ttl_seconds: Optional[float] = None):
        """
        Decorator for caching function results.
        
        Args:
            ttl_seconds: TTL for cached results
            
        Returns:
            Decorator function
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = f"{func.__name__}:{self._generate_key(*args, **kwargs)}"
                
                # Try to get from cache
                result = self.get(cache_key)
                if result is not None:
                    return result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.put(cache_key, result, ttl_seconds)
                
                return result
            
            return wrapper
        return decorator


class DocumentationCache(MemoryEfficientCache):
    """
    Specialized cache for documentation with topic-aware invalidation.
    """
    
    def __init__(self, **kwargs):
        """Initialize documentation cache with optimized defaults."""
        defaults = {
            'max_size_bytes': 50 * 1024 * 1024,  # 50MB for docs
            'max_entries': 5000,
            'default_ttl_seconds': 1800,  # 30 minutes for docs
        }
        defaults.update(kwargs)
        super().__init__(**defaults)
        
        # Track topics for efficient invalidation
        self._topic_keys: Dict[str, Set[str]] = {}
    
    def put_documentation(
        self,
        library_id: str,
        topic: Optional[str],
        documentation: Any,
        ttl_seconds: Optional[float] = None
    ) -> None:
        """
        Cache documentation with topic tracking.
        
        Args:
            library_id: Library identifier
            topic: Documentation topic (None for general docs)
            documentation: Documentation data to cache
            ttl_seconds: TTL for this entry
        """
        key = self._generate_doc_key(library_id, topic)
        self.put(key, documentation, ttl_seconds)
        
        # Track topic association
        topic_key = topic or "general"
        if topic_key not in self._topic_keys:
            self._topic_keys[topic_key] = set()
        self._topic_keys[topic_key].add(key)
    
    def get_documentation(
        self,
        library_id: str,
        topic: Optional[str]
    ) -> Optional[Any]:
        """
        Get cached documentation.
        
        Args:
            library_id: Library identifier
            topic: Documentation topic
            
        Returns:
            Cached documentation or None
        """
        key = self._generate_doc_key(library_id, topic)
        return self.get(key)
    
    def invalidate_topic(self, topic: str) -> int:
        """
        Invalidate all documentation for a specific topic.
        
        Args:
            topic: Topic to invalidate
            
        Returns:
            Number of entries invalidated
        """
        with self._lock:
            if topic not in self._topic_keys:
                return 0
            
            keys_to_remove = list(self._topic_keys[topic])
            count = 0
            
            for key in keys_to_remove:
                if self.invalidate(key):
                    count += 1
            
            # Clean up topic tracking
            del self._topic_keys[topic]
            
            return count
    
    def invalidate_library(self, library_id: str) -> int:
        """
        Invalidate all documentation for a specific library.
        
        Args:
            library_id: Library to invalidate
            
        Returns:
            Number of entries invalidated
        """
        return self.invalidate_pattern(library_id)
    
    def _generate_doc_key(self, library_id: str, topic: Optional[str]) -> str:
        """Generate cache key for documentation."""
        return f"doc:{library_id}:{topic or 'general'}"


class ConnectionPool:
    """
    Connection pool for MCP servers with automatic cleanup and health monitoring.
    """
    
    def __init__(
        self,
        max_connections_per_server: int = 5,
        connection_timeout: float = 30.0,
        idle_timeout: float = 300.0,  # 5 minutes
        health_check_interval: float = 60.0  # 1 minute
    ):
        """
        Initialize connection pool.
        
        Args:
            max_connections_per_server: Maximum connections per server
            connection_timeout: Timeout for establishing connections
            idle_timeout: Timeout for idle connections
            health_check_interval: Interval for health checks
        """
        self.max_connections_per_server = max_connections_per_server
        self.connection_timeout = connection_timeout
        self.idle_timeout = idle_timeout
        self.health_check_interval = health_check_interval
        
        # Connection pools per server
        self._pools: Dict[str, List[Any]] = {}
        self._pool_locks: Dict[str, asyncio.Lock] = {}
        self._connection_stats: Dict[str, Dict[str, Any]] = {}
        
        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        logger.info(f"Initialized connection pool with max_connections={max_connections_per_server}")
    
    async def get_connection(self, server_name: str, create_func) -> Any:
        """
        Get connection from pool or create new one.
        
        Args:
            server_name: Name of the server
            create_func: Async function to create new connection
            
        Returns:
            Connection object
        """
        if server_name not in self._pool_locks:
            self._pool_locks[server_name] = asyncio.Lock()
        
        async with self._pool_locks[server_name]:
            # Initialize pool for server if needed
            if server_name not in self._pools:
                self._pools[server_name] = []
                self._connection_stats[server_name] = {
                    'created': 0,
                    'reused': 0,
                    'errors': 0,
                    'last_used': time.time()
                }
            
            pool = self._pools[server_name]
            stats = self._connection_stats[server_name]
            
            # Try to reuse existing connection
            while pool:
                connection = pool.pop(0)
                
                # Check if connection is still valid
                if await self._is_connection_healthy(connection):
                    stats['reused'] += 1
                    stats['last_used'] = time.time()
                    logger.debug(f"Reused connection for server: {server_name}")
                    return connection
                else:
                    # Connection is unhealthy, close it
                    await self._close_connection(connection)
            
            # Create new connection
            try:
                connection = await asyncio.wait_for(
                    create_func(),
                    timeout=self.connection_timeout
                )
                stats['created'] += 1
                stats['last_used'] = time.time()
                logger.debug(f"Created new connection for server: {server_name}")
                return connection
                
            except Exception as e:
                stats['errors'] += 1
                logger.error(f"Failed to create connection for {server_name}: {e}")
                raise
    
    async def return_connection(self, server_name: str, connection: Any) -> None:
        """
        Return connection to pool.
        
        Args:
            server_name: Name of the server
            connection: Connection to return
        """
        if self._shutdown or server_name not in self._pool_locks:
            await self._close_connection(connection)
            return
        
        async with self._pool_locks[server_name]:
            pool = self._pools.get(server_name, [])
            
            # Check pool size limit
            if len(pool) >= self.max_connections_per_server:
                await self._close_connection(connection)
                return
            
            # Check connection health before returning to pool
            if await self._is_connection_healthy(connection):
                pool.append(connection)
                logger.debug(f"Returned connection to pool for server: {server_name}")
            else:
                await self._close_connection(connection)
    
    async def _is_connection_healthy(self, connection: Any) -> bool:
        """Check if connection is healthy."""
        try:
            # Basic health check - this would be customized based on connection type
            if hasattr(connection, 'is_connected'):
                return connection.is_connected()
            elif hasattr(connection, 'ping'):
                await connection.ping()
                return True
            else:
                # Assume healthy if no specific check available
                return True
        except Exception:
            return False
    
    async def _close_connection(self, connection: Any) -> None:
        """Close connection safely."""
        try:
            if hasattr(connection, 'close'):
                await connection.close()
            elif hasattr(connection, 'disconnect'):
                await connection.disconnect()
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")
    
    async def start_health_monitoring(self) -> None:
        """Start background health monitoring task."""
        if self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            logger.info("Started connection pool health monitoring")
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._cleanup_idle_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    async def _cleanup_idle_connections(self) -> None:
        """Clean up idle connections."""
        current_time = time.time()
        
        for server_name in list(self._pools.keys()):
            if server_name not in self._pool_locks:
                continue
                
            async with self._pool_locks[server_name]:
                pool = self._pools[server_name]
                stats = self._connection_stats[server_name]
                
                # Check if server has been idle too long
                if current_time - stats['last_used'] > self.idle_timeout:
                    # Close all connections for this server
                    while pool:
                        connection = pool.pop()
                        await self._close_connection(connection)
                    
                    logger.debug(f"Cleaned up idle connections for server: {server_name}")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        stats = {}
        
        for server_name, pool in self._pools.items():
            server_stats = self._connection_stats.get(server_name, {})
            stats[server_name] = {
                'active_connections': len(pool),
                'max_connections': self.max_connections_per_server,
                **server_stats
            }
        
        return stats
    
    async def shutdown(self) -> None:
        """Shutdown connection pool and cleanup resources."""
        self._shutdown = True
        
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        for server_name in list(self._pools.keys()):
            if server_name in self._pool_locks:
                async with self._pool_locks[server_name]:
                    pool = self._pools[server_name]
                    while pool:
                        connection = pool.pop()
                        await self._close_connection(connection)
        
        self._pools.clear()
        self._pool_locks.clear()
        self._connection_stats.clear()
        
        logger.info("Connection pool shutdown completed")


# Global cache instances
_documentation_cache: Optional[DocumentationCache] = None
_connection_pool: Optional[ConnectionPool] = None


def get_documentation_cache() -> DocumentationCache:
    """Get global documentation cache instance."""
    global _documentation_cache
    if _documentation_cache is None:
        _documentation_cache = DocumentationCache()
    return _documentation_cache


def get_connection_pool() -> ConnectionPool:
    """Get global connection pool instance."""
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = ConnectionPool()
    return _connection_pool


async def initialize_caching() -> None:
    """Initialize global caching components."""
    # Initialize connection pool health monitoring
    pool = get_connection_pool()
    await pool.start_health_monitoring()
    
    logger.info("Caching and optimization components initialized")


async def shutdown_caching() -> None:
    """Shutdown global caching components."""
    global _documentation_cache, _connection_pool
    
    if _connection_pool:
        await _connection_pool.shutdown()
        _connection_pool = None
    
    if _documentation_cache:
        _documentation_cache.clear()
        _documentation_cache = None
    
    logger.info("Caching and optimization components shutdown")