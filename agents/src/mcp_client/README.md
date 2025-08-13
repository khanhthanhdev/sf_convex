# MCPClient - Model Context Protocol Client

This module provides a robust MCP client implementation for connecting to and managing MCP servers, with specific support for Context7 integration with Manim Community documentation.

## Features

- ✅ **Configuration Management**: Load MCP server configurations from JSON files
- ✅ **Connection Management**: Initialize and manage connections to multiple MCP servers
- ✅ **Connection Testing**: Test server connectivity with comprehensive validation
- ✅ **Error Handling**: Robust error handling for server connection failures
- ✅ **Context7 Integration**: Specialized support for Context7 documentation services
- ✅ **Async Context Manager**: Automatic resource cleanup with context managers
- ✅ **Reconnection Logic**: Automatic reconnection capabilities for failed servers

## Quick Start

### Basic Usage with Context Manager (Recommended)

```python
import asyncio
from mcp_client import mcp_client_context

async def main():
    async with mcp_client_context() as client:
        # Check connection status
        status = client.get_connection_status()
        print(f"Connection status: {status}")
        
        # Test connections
        result = await client.test_connection()
        print(f"Connection test: {result.valid}")
        
        # Get Context7 session if available
        if client.is_server_connected('context7'):
            session = await client.get_context7_session()
            # Use session for documentation queries
            tools = await session.list_tools()
            print(f"Available tools: {len(tools.tools)}")

asyncio.run(main())
```

### Manual Lifecycle Management

```python
import asyncio
from mcp_client import MCPClient

async def main():
    client = MCPClient()
    try:
        await client.initialize()
        
        # Use client...
        status = client.get_connection_status()
        print(f"Status: {status}")
        
    finally:
        await client.close()

asyncio.run(main())
```

### Configuration Loading

```python
from config.mcp_config import MCPConfigLoader

# Load configuration
loader = MCPConfigLoader()
config = loader.load_config()

print(f"Loaded {len(config.mcp_servers)} servers")
for name, server in config.mcp_servers.items():
    print(f"Server '{name}': {server.command} {server.args}")
```

## Configuration

The MCPClient loads configuration from these locations (in order of preference):

1. `mcp_config.json` (project root)
2. `.kiro/settings/mcp.json` (workspace config)
3. `~/.kiro/settings/mcp.json` (user global config)
4. `.cursor/mcp.json` (Cursor IDE config)

### Example Configuration

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"],
      "env": {},
      "disabled": false,
      "autoApprove": [
        "mcp_context7_resolve_library_id",
        "mcp_context7_get_library_docs"
      ]
    }
  }
}
```

## API Reference

### MCPClient

Main client class for managing MCP server connections.

#### Methods

- `__init__(config_path=None)`: Initialize client with optional config path
- `initialize()`: Load configuration and establish server connections
- `test_connection(server_name=None)`: Test connection to servers
- `get_server_session(server_name)`: Get active session for a server
- `get_context7_session()`: Get Context7 server session
- `is_server_connected(server_name)`: Check if server is connected
- `get_connection_status()`: Get status of all servers
- `reconnect_server(server_name)`: Reconnect to a specific server
- `close()`: Close all connections and cleanup

### Convenience Functions

- `create_mcp_client(config_path=None)`: Create and initialize client
- `mcp_client_context(config_path=None)`: Async context manager
- `test_mcp_connection(config_path=None, server_name=None)`: Test connections

### Exception Classes

- `MCPConnectionError`: Raised when connection operations fail
- `MCPServerNotFoundError`: Raised when requested server is not configured

## Context7 Integration

The MCPClient provides specialized support for Context7, which enables access to library documentation. The `Context7DocsRetriever` class provides high-level functions for Manim Community documentation retrieval.

### Basic Usage

```python
from mcp_client import mcp_client_context, get_manim_docs

async with mcp_client_context() as client:
    if client.is_server_connected('context7'):
        # Get general Manim documentation
        docs = await get_manim_docs(client, max_tokens=5000)
        print(f"Retrieved {docs['total_sections']} sections")
        print(f"Found {len(docs['code_examples'])} code examples")
```

### Topic-Specific Documentation

```python
from mcp_client import get_manim_docs, extract_manim_code_examples

async with mcp_client_context() as client:
    # Get documentation for specific topics
    animation_docs = await get_manim_docs(client, topic="animations", max_tokens=3000)
    
    # Extract code examples
    code_snippets = await extract_manim_code_examples(client, topic="scenes")
    print(f"Extracted {len(code_snippets)} code snippets")
```

### Advanced Usage with Context7DocsRetriever

```python
from mcp_client import Context7DocsRetriever

async with mcp_client_context() as client:
    retriever = Context7DocsRetriever(client)
    
    # Resolve Manim library ID
    library_id = await retriever.resolve_manim_library_id()
    
    # Get documentation with custom parameters
    doc_response = await retriever.retrieve_documentation(
        topic="geometry",
        max_tokens=4000,
        library_id=library_id
    )
    
    # Extract and format code snippets
    all_snippets = []
    for section in doc_response.sections:
        all_snippets.extend(section.code_snippets)
    
    formatted_code = retriever.format_code_snippets(all_snippets, "markdown")
    print(formatted_code)
```

### Context7 Documentation Functions

#### High-Level Functions

- `get_manim_docs(client, topic=None, max_tokens=10000)`: Get Manim documentation with code examples
- `extract_manim_code_examples(client, topic=None)`: Extract code snippets from Manim documentation
- `resolve_manim_library_id(client)`: Resolve Manim Community library ID

#### Context7DocsRetriever Class

- `resolve_manim_library_id()`: Resolve library ID for Manim Community
- `retrieve_documentation(topic=None, max_tokens=10000, library_id=None)`: Retrieve documentation
- `extract_code_snippets(text)`: Extract code snippets from text
- `format_code_snippets(snippets, format_type="markdown")`: Format code snippets
- `get_manim_documentation(topic=None, include_code_examples=True, max_tokens=10000)`: High-level documentation retrieval

#### Data Classes

- `CodeSnippet`: Represents a code snippet with language, code, and description
- `DocumentationSection`: Represents a documentation section with title, content, and code snippets
- `DocumentationResponse`: Complete response with library ID, topic, sections, and metadata

### Required Context7 Tools

- `mcp_context7_resolve_library_id`: Resolve library names to Context7 IDs
- `mcp_context7_get_library_docs`: Fetch library documentation

### Error Handling

```python
from mcp_client import DocumentationError, LibraryNotFoundError, MCPConnectionError

try:
    docs = await get_manim_docs(client, topic="animations")
except DocumentationError as e:
    print(f"Documentation retrieval failed: {e}")
except LibraryNotFoundError as e:
    print(f"Library not found: {e}")
except MCPConnectionError as e:
    print(f"Context7 server not available: {e}")
```

## Error Handling

The MCPClient provides comprehensive error handling:

```python
from mcp_client import MCPClient, MCPConnectionError, MCPServerNotFoundError

try:
    async with mcp_client_context() as client:
        session = await client.get_server_session('my_server')
        # Use session...
        
except MCPConnectionError as e:
    print(f"Connection failed: {e}")
    
except MCPServerNotFoundError as e:
    print(f"Server not found: {e}")
    
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Requirements

- Python 3.8+
- `mcp` library for MCP protocol support
- `pydantic` for configuration validation
- `asyncio` for async operations

For Context7 integration:
- Node.js and npm/npx
- `@upstash/context7-mcp` package

## Testing

Run the test suite:

```bash
python agents/src/mcp_client/test_client.py
```

Run the demonstration:

```bash
python agents/src/mcp_client/demo.py
```

## Development

The MCPClient is designed to be:

- **Robust**: Comprehensive error handling and validation
- **Flexible**: Support for multiple MCP servers and configurations
- **Async-first**: Built with asyncio for concurrent operations
- **Context7-optimized**: Specialized support for documentation services
- **Production-ready**: Proper resource management and cleanup

## Troubleshooting

### Common Issues

1. **"No such file or directory: 'npx'"**
   - Install Node.js and npm
   - Ensure npx is in your PATH

2. **"MCP configuration file not found"**
   - Create a configuration file in one of the expected locations
   - Use absolute path when initializing MCPClient

3. **"Connection timeout"**
   - Check if the MCP server is responsive
   - Verify network connectivity
   - Increase timeout in server configuration

4. **"Server not connected"**
   - Check server configuration
   - Verify server command and arguments
   - Test connection manually

### Debug Logging

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will show detailed information about:
- Configuration loading
- Connection attempts
- Server communication
- Error details