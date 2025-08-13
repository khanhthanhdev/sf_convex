"""
MCP Client Implementation for Context7 Integration

This module provides MCPClient functionality for connecting to and managing
MCP servers, specifically optimized for Context7 integration with Manim Community documentation.
"""

import asyncio
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager

import mcp
from mcp import stdio_client

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.mcp_config import MCPConfigLoader, MCPConfig
from config.models import MCPServerConfig, ValidationResult
from .cache import get_connection_pool, get_documentation_cache, initialize_caching, shutdown_caching

logger = logging.getLogger(__name__)


class MCPConnectionError(Exception):
    """Exception raised when MCP server connection fails."""
    pass


class MCPServerNotFoundError(Exception):
    """Exception raised when requested MCP server is not configured."""
    pass


class MCPClient:
    """
    MCP Client for managing connections to MCP servers with Context7 integration.
    
    This client provides functionality to:
    - Initialize connections to configured MCP servers
    - Test server connectivity and validation
    - Handle connection failures with retry mechanisms
    - Manage Context7-specific operations for Manim documentation
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None, enable_connection_pooling: bool = True):
        """
        Initialize MCPClient with configuration.
        
        Args:
            config_path: Path to MCP configuration file. If None, uses default locations.
            enable_connection_pooling: Whether to enable connection pooling for better performance
        """
        self.config_loader = MCPConfigLoader(config_path)
        self.config: Optional[MCPConfig] = None
        self.sessions: Dict[str, ClientSession] = {}
        self.server_processes: Dict[str, subprocess.Popen] = {}
        self._connection_status: Dict[str, bool] = {}
        
        # Connection pooling and caching
        self.enable_connection_pooling = enable_connection_pooling
        self._connection_pool = get_connection_pool() if enable_connection_pooling else None
        self._doc_cache = get_documentation_cache()
        self._initialized_caching = False
        
        logger.info(f"MCPClient initialized with config path: {self.config_loader.config_path}, "
                   f"connection pooling: {'enabled' if enable_connection_pooling else 'disabled'}")
    
    @classmethod
    def from_config_file(cls, config_path: Union[str, Path]) -> "MCPClient":
        """
        Create MCPClient instance from configuration file.
        
        Args:
            config_path: Path to MCP configuration file
            
        Returns:
            Configured MCPClient instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If config format is invalid
        """
        return cls(config_path)
    
    async def initialize(self) -> None:
        """
        Initialize the MCP client by loading configuration and establishing connections.
        
        Raises:
            MCPConnectionError: If initialization fails
            FileNotFoundError: If config file doesn't exist
        """
        try:
            # Initialize caching and connection pooling
            if not self._initialized_caching:
                await initialize_caching()
                self._initialized_caching = True
                logger.info("Initialized caching and connection pooling")
            
            # Load configuration
            self.config = self.config_loader.load_config()
            logger.info(f"Loaded MCP configuration with {len(self.config.mcp_servers)} servers")
            
            # Initialize connections to enabled servers
            for server_name, server_config in self.config.mcp_servers.items():
                if not server_config.disabled:
                    try:
                        await self._connect_server(server_name, server_config)
                        logger.info(f"Successfully connected to MCP server: {server_name}")
                    except Exception as e:
                        logger.error(f"Failed to connect to MCP server '{server_name}': {e}")
                        self._connection_status[server_name] = False
                        # Continue with other servers rather than failing completely
                else:
                    logger.info(f"Skipping disabled MCP server: {server_name}")
                    self._connection_status[server_name] = False
            
            logger.info("MCP client initialization completed")
            
        except Exception as e:
            logger.error(f"MCP client initialization failed: {e}")
            raise MCPConnectionError(f"Failed to initialize MCP client: {e}")
    
    async def _connect_server(self, server_name: str, server_config: MCPServerConfig) -> None:
        """
        Connect to a specific MCP server with connection pooling support.
        
        Args:
            server_name: Name of the server
            server_config: Server configuration
            
        Raises:
            MCPConnectionError: If connection fails
        """
        try:
            if self._connection_pool:
                # Use connection pool for better resource management
                async def create_connection():
                    server_params = mcp.StdioServerParameters(
                        command=server_config.command,
                        args=server_config.args,
                        env=server_config.env or {}
                    )
                    
                    # Create stdio client
                    read, write = await stdio_client(server_params).__aenter__()
                    session = await mcp.ClientSession(read, write).__aenter__()
                    await session.initialize()
                    return session
                
                # Get connection from pool
                session = await self._connection_pool.get_connection(server_name, create_connection)
                self.sessions[server_name] = session
                self._connection_status[server_name] = True
                
                logger.info(f"Successfully established pooled connection to {server_name}")
            else:
                # Fallback to direct connection
                server_params = mcp.StdioServerParameters(
                    command=server_config.command,
                    args=server_config.args,
                    env=server_config.env or {}
                )
                
                # Create and start the client session
                async with stdio_client(server_params) as (read, write):
                    async with mcp.ClientSession(read, write) as session:
                        # Initialize the session
                        await session.initialize()
                        
                        # Store the session for later use
                        self.sessions[server_name] = session
                        self._connection_status[server_name] = True
                        
                        logger.info(f"Successfully established direct connection to {server_name}")
                    
        except Exception as e:
            logger.error(f"Failed to connect to server '{server_name}': {e}")
            self._connection_status[server_name] = False
            raise MCPConnectionError(f"Connection to '{server_name}' failed: {e}")
    
    async def test_connection(self, server_name: Optional[str] = None) -> ValidationResult:
        """
        Test connection to MCP servers.
        
        Args:
            server_name: Specific server to test, or None to test all servers
            
        Returns:
            ValidationResult with connection test results
        """
        result = ValidationResult(valid=True)
        
        if not self.config:
            result.add_error("MCP client not initialized - call initialize() first")
            return result
        
        servers_to_test = [server_name] if server_name else list(self.config.mcp_servers.keys())
        
        for name in servers_to_test:
            if name not in self.config.mcp_servers:
                result.add_error(f"Server '{name}' not found in configuration")
                continue
            
            server_config = self.config.mcp_servers[name]
            
            if server_config.disabled:
                result.add_warning(f"Server '{name}' is disabled")
                continue
            
            try:
                # Test basic connectivity
                await self._test_server_connectivity(name, server_config, result)
                
            except Exception as e:
                result.add_error(f"Connection test failed for server '{name}': {e}")
                logger.error(f"Connection test error for {name}: {e}")
        
        logger.info(f"Connection test completed: {'Passed' if result.valid else 'Failed'}")
        return result
    
    async def _test_server_connectivity(self, server_name: str, server_config: MCPServerConfig, result: ValidationResult) -> None:
        """
        Test connectivity to a specific server.
        
        Args:
            server_name: Name of the server
            server_config: Server configuration
            result: ValidationResult to update
        """
        try:
            # Check if we already have a connection
            if server_name in self.sessions and self._connection_status.get(server_name, False):
                result.add_warning(f"Server '{server_name}' already connected")
                return
            
            # Try to establish a test connection
            server_params = mcp.StdioServerParameters(
                command=server_config.command,
                args=server_config.args,
                env=server_config.env or {}
            )
            
            # Test connection with timeout
            async with asyncio.timeout(30):  # 30 second timeout
                async with stdio_client(server_params) as (read, write):
                    async with mcp.ClientSession(read, write) as session:
                        await session.initialize()
                        
                        # Test basic functionality by listing tools
                        tools = await session.list_tools()
                        logger.info(f"Server '{server_name}' has {len(tools.tools)} tools available")
                        
                        # For Context7, verify specific tools are available
                        if server_name == 'context7':
                            self._validate_context7_tools(tools.tools, result)
            
            self._connection_status[server_name] = True
            logger.info(f"Connection test passed for server: {server_name}")
            
        except asyncio.TimeoutError:
            result.add_error(f"Connection timeout for server '{server_name}'")
            self._connection_status[server_name] = False
        except Exception as e:
            result.add_error(f"Connection test failed for server '{server_name}': {e}")
            self._connection_status[server_name] = False
    
    def _validate_context7_tools(self, tools: List[Any], result: ValidationResult) -> None:
        """
        Validate that Context7 server has required tools for Manim integration.
        
        Args:
            tools: List of available tools
            result: ValidationResult to update
        """
        tool_names = [tool.name for tool in tools]
        
        required_tools = [
            'mcp_context7_resolve_library_id',
            'mcp_context7_get_library_docs'
        ]
        
        missing_tools = [tool for tool in required_tools if tool not in tool_names]
        
        if missing_tools:
            result.add_error(f"Context7 server missing required tools: {missing_tools}")
        else:
            result.add_warning("Context7 server has all required tools for Manim integration")
        
        logger.info(f"Context7 tools validation: {len(tool_names)} tools available")
    
    async def get_server_session(self, server_name: str) -> mcp.ClientSession:
        """
        Get active session for a specific server.
        
        Args:
            server_name: Name of the server
            
        Returns:
            Active ClientSession for the server
            
        Raises:
            MCPServerNotFoundError: If server is not configured
            MCPConnectionError: If server is not connected
        """
        if not self.config:
            raise MCPConnectionError("MCP client not initialized")
        
        if server_name not in self.config.mcp_servers:
            raise MCPServerNotFoundError(f"Server '{server_name}' not found in configuration")
        
        if server_name not in self.sessions or not self._connection_status.get(server_name, False):
            # Try to reconnect
            server_config = self.config.mcp_servers[server_name]
            await self._connect_server(server_name, server_config)
        
        if server_name not in self.sessions:
            raise MCPConnectionError(f"No active session for server '{server_name}'")
        
        return self.sessions[server_name]
    
    async def get_context7_session(self) -> mcp.ClientSession:
        """
        Get Context7 server session for Manim documentation access.
        
        Returns:
            Active ClientSession for Context7 server
            
        Raises:
            MCPServerNotFoundError: If Context7 is not configured
            MCPConnectionError: If Context7 is not connected
        """
        return await self.get_server_session('context7')
    
    def is_server_connected(self, server_name: str) -> bool:
        """
        Check if a server is currently connected.
        
        Args:
            server_name: Name of the server
            
        Returns:
            True if server is connected, False otherwise
        """
        return self._connection_status.get(server_name, False)
    
    def get_connection_status(self) -> Dict[str, bool]:
        """
        Get connection status for all configured servers.
        
        Returns:
            Dictionary mapping server names to connection status
        """
        return self._connection_status.copy()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        stats = {}
        
        if self._doc_cache:
            cache_stats = self._doc_cache.get_stats()
            stats['documentation_cache'] = cache_stats.to_dict()
        
        if self._connection_pool:
            pool_stats = self._connection_pool.get_pool_stats()
            stats['connection_pool'] = pool_stats
        
        return stats
    
    def clear_cache(self, cache_type: Optional[str] = None) -> None:
        """
        Clear cache entries.
        
        Args:
            cache_type: Type of cache to clear ('documentation', 'all', or None for all)
        """
        if cache_type in (None, 'all', 'documentation') and self._doc_cache:
            self._doc_cache.clear()
            logger.info("Cleared documentation cache")
    
    def invalidate_cache_pattern(self, pattern: str, cache_type: Optional[str] = None) -> int:
        """
        Invalidate cache entries matching a pattern.
        
        Args:
            pattern: Pattern to match for invalidation
            cache_type: Type of cache to invalidate ('documentation', 'all', or None for all)
            
        Returns:
            Number of entries invalidated
        """
        total_invalidated = 0
        
        if cache_type in (None, 'all', 'documentation') and self._doc_cache:
            invalidated = self._doc_cache.invalidate_pattern(pattern)
            total_invalidated += invalidated
            logger.info(f"Invalidated {invalidated} documentation cache entries matching pattern: {pattern}")
        
        return total_invalidated
    
    async def reconnect_server(self, server_name: str) -> bool:
        """
        Reconnect to a specific server.
        
        Args:
            server_name: Name of the server to reconnect
            
        Returns:
            True if reconnection successful, False otherwise
        """
        if not self.config:
            logger.error("Cannot reconnect: MCP client not initialized")
            return False
        
        if server_name not in self.config.mcp_servers:
            logger.error(f"Cannot reconnect: Server '{server_name}' not configured")
            return False
        
        try:
            # Close existing connection if any
            if server_name in self.sessions:
                try:
                    # Note: ClientSession doesn't have explicit close method in mcp library
                    # The connection will be closed when the context manager exits
                    del self.sessions[server_name]
                except Exception as e:
                    logger.warning(f"Error closing existing session for '{server_name}': {e}")
            
            # Attempt reconnection
            server_config = self.config.mcp_servers[server_name]
            await self._connect_server(server_name, server_config)
            
            logger.info(f"Successfully reconnected to server: {server_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reconnect to server '{server_name}': {e}")
            self._connection_status[server_name] = False
            return False
    
    async def close(self) -> None:
        """
        Close all server connections and cleanup resources.
        """
        logger.info("Closing MCP client connections...")
        
        # Return connections to pool or close them
        for server_name in list(self.sessions.keys()):
            try:
                session = self.sessions[server_name]
                
                if self._connection_pool:
                    # Return connection to pool
                    await self._connection_pool.return_connection(server_name, session)
                    logger.info(f"Returned connection to pool for server: {server_name}")
                else:
                    # Direct close (sessions will be closed when their context managers exit)
                    logger.info(f"Closed direct connection to server: {server_name}")
                
                del self.sessions[server_name]
                self._connection_status[server_name] = False
                
            except Exception as e:
                logger.warning(f"Error closing session for '{server_name}': {e}")
        
        # Terminate any server processes
        for server_name, process in self.server_processes.items():
            try:
                if process.poll() is None:  # Process is still running
                    process.terminate()
                    # Give it a moment to terminate gracefully
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    logger.info(f"Terminated server process: {server_name}")
            except Exception as e:
                logger.warning(f"Error terminating process for '{server_name}': {e}")
        
        self.server_processes.clear()
        self.sessions.clear()
        self._connection_status.clear()
        
        # Shutdown caching if this was the last client
        if self._initialized_caching:
            await shutdown_caching()
            self._initialized_caching = False
        
        logger.info("MCP client cleanup completed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Convenience functions for common operations

async def create_mcp_client(config_path: Optional[Union[str, Path]] = None) -> MCPClient:
    """
    Create and initialize an MCP client.
    
    Args:
        config_path: Path to MCP configuration file
        
    Returns:
        Initialized MCPClient instance
        
    Raises:
        MCPConnectionError: If initialization fails
    """
    client = MCPClient(config_path)
    await client.initialize()
    return client


@asynccontextmanager
async def mcp_client_context(config_path: Optional[Union[str, Path]] = None):
    """
    Async context manager for MCP client with automatic cleanup.
    
    Args:
        config_path: Path to MCP configuration file
        
    Yields:
        Initialized MCPClient instance
    """
    client = MCPClient(config_path)
    try:
        await client.initialize()
        yield client
    finally:
        await client.close()


async def test_mcp_connection(config_path: Optional[Union[str, Path]] = None, server_name: Optional[str] = None) -> ValidationResult:
    """
    Test MCP server connections.
    
    Args:
        config_path: Path to MCP configuration file
        server_name: Specific server to test, or None for all servers
        
    Returns:
        ValidationResult with test results
    """
    async with mcp_client_context(config_path) as client:
        return await client.test_connection(server_name)