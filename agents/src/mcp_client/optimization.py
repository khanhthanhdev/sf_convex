"""
Optimization and Performance Monitoring Module

This module provides comprehensive optimization features and performance monitoring
for the MCP client and agent system, including:
- Cache performance analysis
- Connection pool monitoring
- Memory usage tracking
- Performance recommendations
- Automated optimization strategies
"""

import asyncio
import logging
import time
import psutil
import gc
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from .cache import get_documentation_cache, get_connection_pool
from .client import MCPClient
from .agent import MCPAgent

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Optimization levels for automatic tuning."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    timestamp: float
    memory_usage_mb: float
    cache_hit_rate: float
    avg_query_time: float
    connection_pool_utilization: float
    total_queries: int
    active_connections: int
    cache_size_mb: float


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation."""
    category: str
    priority: str  # 'high', 'medium', 'low'
    description: str
    action: str
    expected_improvement: str
    implementation_complexity: str  # 'low', 'medium', 'high'


class PerformanceMonitor:
    """
    Performance monitoring and optimization system.
    
    Provides real-time monitoring of cache performance, memory usage,
    connection pool utilization, and generates optimization recommendations.
    """
    
    def __init__(
        self,
        monitoring_interval: float = 60.0,  # 1 minute
        history_size: int = 100,
        auto_optimize: bool = False,
        optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    ):
        """
        Initialize performance monitor.
        
        Args:
            monitoring_interval: Interval between performance measurements (seconds)
            history_size: Number of historical measurements to keep
            auto_optimize: Whether to automatically apply optimizations
            optimization_level: Level of automatic optimization
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.auto_optimize = auto_optimize
        self.optimization_level = optimization_level
        
        # Performance history
        self.metrics_history: List[PerformanceMetrics] = []
        self.recommendations: List[OptimizationRecommendation] = []
        
        # Monitoring state
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Thresholds for recommendations
        self.thresholds = {
            'low_cache_hit_rate': 0.7,
            'high_memory_usage_mb': 500,
            'slow_query_time_ms': 5000,
            'high_pool_utilization': 0.8,
            'large_cache_size_mb': 100
        }
        
        logger.info(f"Performance monitor initialized with {monitoring_interval}s interval")
    
    async def start_monitoring(self) -> None:
        """Start background performance monitoring."""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Started performance monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop background performance monitoring."""
        self._shutdown = True
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
        
        logger.info("Stopped performance monitoring")
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.monitoring_interval)
                
                # Collect performance metrics
                metrics = await self.collect_metrics()
                self.metrics_history.append(metrics)
                
                # Trim history to size limit
                if len(self.metrics_history) > self.history_size:
                    self.metrics_history = self.metrics_history[-self.history_size:]
                
                # Generate recommendations
                await self._analyze_performance()
                
                # Apply automatic optimizations if enabled
                if self.auto_optimize:
                    await self._apply_auto_optimizations()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
    
    async def collect_metrics(self) -> PerformanceMetrics:
        """
        Collect current performance metrics.
        
        Returns:
            PerformanceMetrics with current system state
        """
        try:
            # Memory usage
            process = psutil.Process()
            memory_usage_mb = process.memory_info().rss / 1024 / 1024
            
            # Cache statistics
            doc_cache = get_documentation_cache()
            cache_stats = doc_cache.get_stats()
            cache_hit_rate = cache_stats.hit_rate
            cache_size_mb = cache_stats.total_size_bytes / 1024 / 1024
            
            # Connection pool statistics
            conn_pool = get_connection_pool()
            pool_stats = conn_pool.get_pool_stats()
            
            # Calculate pool utilization
            total_connections = sum(
                stats.get('active_connections', 0) 
                for stats in pool_stats.values()
            )
            max_connections = sum(
                stats.get('max_connections', 0) 
                for stats in pool_stats.values()
            )
            pool_utilization = total_connections / max(max_connections, 1)
            
            # Average query time (from recent history)
            avg_query_time = 0.0
            if len(self.metrics_history) > 0:
                recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
                avg_query_time = sum(m.avg_query_time for m in recent_metrics) / len(recent_metrics)
            
            return PerformanceMetrics(
                timestamp=time.time(),
                memory_usage_mb=memory_usage_mb,
                cache_hit_rate=cache_hit_rate,
                avg_query_time=avg_query_time,
                connection_pool_utilization=pool_utilization,
                total_queries=cache_stats.hits + cache_stats.misses,
                active_connections=total_connections,
                cache_size_mb=cache_size_mb
            )
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
            # Return default metrics on error
            return PerformanceMetrics(
                timestamp=time.time(),
                memory_usage_mb=0.0,
                cache_hit_rate=0.0,
                avg_query_time=0.0,
                connection_pool_utilization=0.0,
                total_queries=0,
                active_connections=0,
                cache_size_mb=0.0
            )
    
    async def _analyze_performance(self) -> None:
        """Analyze performance and generate recommendations."""
        if not self.metrics_history:
            return
        
        current_metrics = self.metrics_history[-1]
        self.recommendations.clear()
        
        # Analyze cache performance
        if current_metrics.cache_hit_rate < self.thresholds['low_cache_hit_rate']:
            self.recommendations.append(OptimizationRecommendation(
                category="Cache Performance",
                priority="high",
                description=f"Cache hit rate is low ({current_metrics.cache_hit_rate:.2%})",
                action="Increase cache TTL or cache size, review caching strategy",
                expected_improvement="20-40% faster response times",
                implementation_complexity="low"
            ))
        
        # Analyze memory usage
        if current_metrics.memory_usage_mb > self.thresholds['high_memory_usage_mb']:
            self.recommendations.append(OptimizationRecommendation(
                category="Memory Usage",
                priority="medium",
                description=f"High memory usage ({current_metrics.memory_usage_mb:.1f} MB)",
                action="Reduce cache size limits or implement more aggressive eviction",
                expected_improvement="Reduced memory footprint",
                implementation_complexity="medium"
            ))
        
        # Analyze query performance
        if current_metrics.avg_query_time > self.thresholds['slow_query_time_ms']:
            self.recommendations.append(OptimizationRecommendation(
                category="Query Performance",
                priority="high",
                description=f"Slow average query time ({current_metrics.avg_query_time:.0f}ms)",
                action="Enable pre-fetching, optimize documentation retrieval",
                expected_improvement="30-50% faster queries",
                implementation_complexity="medium"
            ))
        
        # Analyze connection pool utilization
        if current_metrics.connection_pool_utilization > self.thresholds['high_pool_utilization']:
            self.recommendations.append(OptimizationRecommendation(
                category="Connection Pool",
                priority="medium",
                description=f"High connection pool utilization ({current_metrics.connection_pool_utilization:.1%})",
                action="Increase max connections per server or implement connection sharing",
                expected_improvement="Better concurrency handling",
                implementation_complexity="low"
            ))
        
        # Analyze cache size
        if current_metrics.cache_size_mb > self.thresholds['large_cache_size_mb']:
            self.recommendations.append(OptimizationRecommendation(
                category="Cache Size",
                priority="low",
                description=f"Large cache size ({current_metrics.cache_size_mb:.1f} MB)",
                action="Implement more aggressive cache eviction or reduce cache limits",
                expected_improvement="Lower memory usage",
                implementation_complexity="low"
            ))
        
        logger.debug(f"Generated {len(self.recommendations)} performance recommendations")
    
    async def _apply_auto_optimizations(self) -> None:
        """Apply automatic optimizations based on current recommendations."""
        if not self.recommendations:
            return
        
        applied_optimizations = []
        
        for rec in self.recommendations:
            if rec.priority == "high" and rec.implementation_complexity == "low":
                try:
                    success = await self._apply_optimization(rec)
                    if success:
                        applied_optimizations.append(rec.description)
                except Exception as e:
                    logger.error(f"Failed to apply optimization '{rec.description}': {e}")
        
        if applied_optimizations:
            logger.info(f"Applied {len(applied_optimizations)} automatic optimizations")
    
    async def _apply_optimization(self, recommendation: OptimizationRecommendation) -> bool:
        """
        Apply a specific optimization recommendation.
        
        Args:
            recommendation: Optimization to apply
            
        Returns:
            True if optimization was applied successfully
        """
        try:
            if "cache hit rate" in recommendation.description.lower():
                # Increase cache TTL
                doc_cache = get_documentation_cache()
                doc_cache.default_ttl_seconds *= 1.5
                logger.info("Increased cache TTL to improve hit rate")
                return True
            
            elif "connection pool utilization" in recommendation.description.lower():
                # Increase connection pool size
                conn_pool = get_connection_pool()
                conn_pool.max_connections_per_server += 2
                logger.info("Increased connection pool size")
                return True
            
            elif "large cache size" in recommendation.description.lower():
                # Trigger cache cleanup
                doc_cache = get_documentation_cache()
                doc_cache._enforce_size_limits()
                logger.info("Triggered cache cleanup to reduce size")
                return True
            
        except Exception as e:
            logger.error(f"Error applying optimization: {e}")
            return False
        
        return False
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent performance metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, limit: Optional[int] = None) -> List[PerformanceMetrics]:
        """
        Get performance metrics history.
        
        Args:
            limit: Maximum number of metrics to return
            
        Returns:
            List of historical performance metrics
        """
        if limit:
            return self.metrics_history[-limit:]
        return self.metrics_history.copy()
    
    def get_recommendations(self, priority: Optional[str] = None) -> List[OptimizationRecommendation]:
        """
        Get current optimization recommendations.
        
        Args:
            priority: Filter by priority ('high', 'medium', 'low')
            
        Returns:
            List of optimization recommendations
        """
        if priority:
            return [rec for rec in self.recommendations if rec.priority == priority]
        return self.recommendations.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.
        
        Returns:
            Dictionary containing performance summary
        """
        if not self.metrics_history:
            return {"status": "no_data", "message": "No performance data available"}
        
        current = self.metrics_history[-1]
        
        # Calculate trends if we have enough history
        trends = {}
        if len(self.metrics_history) >= 10:
            recent = self.metrics_history[-10:]
            older = self.metrics_history[-20:-10] if len(self.metrics_history) >= 20 else recent
            
            trends = {
                'cache_hit_rate_trend': (
                    sum(m.cache_hit_rate for m in recent) / len(recent) -
                    sum(m.cache_hit_rate for m in older) / len(older)
                ),
                'memory_usage_trend': (
                    sum(m.memory_usage_mb for m in recent) / len(recent) -
                    sum(m.memory_usage_mb for m in older) / len(older)
                ),
                'query_time_trend': (
                    sum(m.avg_query_time for m in recent) / len(recent) -
                    sum(m.avg_query_time for m in older) / len(older)
                )
            }
        
        return {
            'status': 'healthy' if len(self.recommendations) == 0 else 'needs_attention',
            'current_metrics': asdict(current),
            'trends': trends,
            'recommendations_count': len(self.recommendations),
            'high_priority_recommendations': len([r for r in self.recommendations if r.priority == 'high']),
            'monitoring_duration_hours': (time.time() - self.metrics_history[0].timestamp) / 3600 if self.metrics_history else 0,
            'data_points': len(self.metrics_history)
        }


class OptimizationManager:
    """
    High-level optimization manager for MCP client and agent systems.
    
    Provides automated optimization strategies and performance tuning.
    """
    
    def __init__(self):
        """Initialize optimization manager."""
        self.monitor = PerformanceMonitor()
        self._optimization_strategies = {
            OptimizationLevel.CONSERVATIVE: self._conservative_optimization,
            OptimizationLevel.BALANCED: self._balanced_optimization,
            OptimizationLevel.AGGRESSIVE: self._aggressive_optimization
        }
        
        logger.info("Optimization manager initialized")
    
    async def optimize_system(
        self,
        mcp_client: MCPClient,
        agent: Optional[MCPAgent] = None,
        level: OptimizationLevel = OptimizationLevel.BALANCED
    ) -> Dict[str, Any]:
        """
        Perform comprehensive system optimization.
        
        Args:
            mcp_client: MCP client to optimize
            agent: Optional agent to optimize
            level: Optimization level to apply
            
        Returns:
            Dictionary containing optimization results
        """
        logger.info(f"Starting system optimization at {level.value} level")
        
        optimization_results = {
            'level': level.value,
            'timestamp': time.time(),
            'optimizations_applied': [],
            'performance_before': {},
            'performance_after': {},
            'recommendations': []
        }
        
        try:
            # Collect baseline metrics
            baseline_metrics = await self.monitor.collect_metrics()
            optimization_results['performance_before'] = asdict(baseline_metrics)
            
            # Apply optimization strategy
            strategy_func = self._optimization_strategies[level]
            applied_optimizations = await strategy_func(mcp_client, agent)
            optimization_results['optimizations_applied'] = applied_optimizations
            
            # Force garbage collection
            gc.collect()
            
            # Wait a moment for changes to take effect
            await asyncio.sleep(2)
            
            # Collect post-optimization metrics
            post_metrics = await self.monitor.collect_metrics()
            optimization_results['performance_after'] = asdict(post_metrics)
            
            # Generate recommendations for further improvements
            await self.monitor._analyze_performance()
            optimization_results['recommendations'] = [
                asdict(rec) for rec in self.monitor.get_recommendations()
            ]
            
            logger.info(f"System optimization completed: {len(applied_optimizations)} optimizations applied")
            
        except Exception as e:
            logger.error(f"System optimization failed: {e}")
            optimization_results['error'] = str(e)
        
        return optimization_results
    
    async def _conservative_optimization(
        self,
        mcp_client: MCPClient,
        agent: Optional[MCPAgent]
    ) -> List[str]:
        """Apply conservative optimization strategies."""
        applied = []
        
        try:
            # Clear expired cache entries
            doc_cache = get_documentation_cache()
            doc_cache._cleanup_expired()
            applied.append("Cleared expired cache entries")
            
            # Optimize connection pool health checks
            conn_pool = get_connection_pool()
            await conn_pool._cleanup_idle_connections()
            applied.append("Cleaned up idle connections")
            
        except Exception as e:
            logger.error(f"Conservative optimization error: {e}")
        
        return applied
    
    async def _balanced_optimization(
        self,
        mcp_client: MCPClient,
        agent: Optional[MCPAgent]
    ) -> List[str]:
        """Apply balanced optimization strategies."""
        applied = await self._conservative_optimization(mcp_client, agent)
        
        try:
            # Optimize cache settings
            doc_cache = get_documentation_cache()
            
            # Adjust cache size based on usage
            cache_stats = doc_cache.get_stats()
            if cache_stats.hit_rate < 0.7:
                # Increase cache size for better hit rate
                doc_cache.max_size_bytes = int(doc_cache.max_size_bytes * 1.2)
                applied.append("Increased cache size for better hit rate")
            
            # Optimize eviction policy based on access patterns
            if cache_stats.entry_count > doc_cache.max_entries * 0.8:
                doc_cache._enforce_size_limits()
                applied.append("Enforced cache size limits")
            
            # Optimize connection pool settings
            conn_pool = get_connection_pool()
            pool_stats = conn_pool.get_pool_stats()
            
            for server_name, stats in pool_stats.items():
                utilization = stats.get('active_connections', 0) / max(stats.get('max_connections', 1), 1)
                if utilization > 0.8:
                    # Increase connection limit for high-utilization servers
                    conn_pool.max_connections_per_server += 1
                    applied.append(f"Increased connection limit for {server_name}")
            
        except Exception as e:
            logger.error(f"Balanced optimization error: {e}")
        
        return applied
    
    async def _aggressive_optimization(
        self,
        mcp_client: MCPClient,
        agent: Optional[MCPAgent]
    ) -> List[str]:
        """Apply aggressive optimization strategies."""
        applied = await self._balanced_optimization(mcp_client, agent)
        
        try:
            # Aggressive cache optimization
            doc_cache = get_documentation_cache()
            
            # Switch to more efficient eviction policy
            from .cache import LFUEvictionPolicy
            doc_cache.eviction_policy = LFUEvictionPolicy()
            applied.append("Switched to LFU eviction policy")
            
            # Increase cache TTL for stable documentation
            doc_cache.default_ttl_seconds = min(doc_cache.default_ttl_seconds * 2, 7200)  # Max 2 hours
            applied.append("Increased cache TTL for better retention")
            
            # Aggressive connection pool optimization
            conn_pool = get_connection_pool()
            
            # Reduce idle timeout for faster cleanup
            conn_pool.idle_timeout = max(conn_pool.idle_timeout * 0.7, 60)  # Min 1 minute
            applied.append("Reduced connection idle timeout")
            
            # Increase health check frequency
            conn_pool.health_check_interval = max(conn_pool.health_check_interval * 0.5, 30)  # Min 30 seconds
            applied.append("Increased health check frequency")
            
            # Agent-specific optimizations
            if agent:
                # Enable more aggressive documentation pre-fetching
                if hasattr(agent, 'config'):
                    agent.config.auto_retrieve_docs = True
                    agent.config.documentation_max_tokens = min(agent.config.documentation_max_tokens * 1.5, 20000)
                    applied.append("Enabled aggressive documentation pre-fetching")
            
        except Exception as e:
            logger.error(f"Aggressive optimization error: {e}")
        
        return applied
    
    async def start_monitoring(self) -> None:
        """Start performance monitoring."""
        await self.monitor.start_monitoring()
    
    async def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        await self.monitor.stop_monitoring()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from monitor."""
        return self.monitor.get_performance_summary()


# Global optimization manager instance
_optimization_manager: Optional[OptimizationManager] = None


def get_optimization_manager() -> OptimizationManager:
    """Get global optimization manager instance."""
    global _optimization_manager
    if _optimization_manager is None:
        _optimization_manager = OptimizationManager()
    return _optimization_manager


async def optimize_mcp_system(
    mcp_client: MCPClient,
    agent: Optional[MCPAgent] = None,
    level: OptimizationLevel = OptimizationLevel.BALANCED,
    start_monitoring: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to optimize MCP system.
    
    Args:
        mcp_client: MCP client to optimize
        agent: Optional agent to optimize
        level: Optimization level
        start_monitoring: Whether to start performance monitoring
        
    Returns:
        Optimization results
    """
    manager = get_optimization_manager()
    
    if start_monitoring:
        await manager.start_monitoring()
    
    return await manager.optimize_system(mcp_client, agent, level)