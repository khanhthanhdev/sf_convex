# Context7 Integration with CodeGenerator

This document describes the integration of Context7 MCP server with the CodeGenerator agent, replacing traditional RAG approaches with real-time Manim Community documentation retrieval.

## Overview

The Context7 integration allows CodeGenerator to:
- Access up-to-date Manim Community documentation in real-time
- Automatically detect relevant topics from implementation plans
- Retrieve targeted documentation and code examples
- Use current documentation for both code generation and error fixing

## Key Components

### 1. MCPClient Integration

The CodeGenerator now accepts an optional `mcp_client` parameter:

```python
from src.mcp_client.client import MCPClient
from src.core.code_generator import CodeGenerator

# Initialize MCP client
mcp_client = MCPClient.from_config_file("mcp_config.json")
await mcp_client.initialize()

# Create CodeGenerator with Context7 integration
generator = CodeGenerator(
    scene_model=your_scene_model,
    helper_model=your_helper_model,
    use_rag=True,  # Enable documentation retrieval
    mcp_client=mcp_client  # Provide MCP client
)
```

### 2. Context7DocsRetriever

The integration uses `Context7DocsRetriever` to:
- Resolve Manim Community library ID (`/manimcommunity/manim`)
- Retrieve documentation with topic filtering
- Extract and format code snippets
- Cache documentation for performance

### 3. Automatic Topic Detection

The system automatically detects relevant Manim topics from implementation text:

- **animation**: animate, animation, transform, morph, transition
- **mobject**: mobject, vmobject, text, shape, geometry  
- **scene**: scene, construct, play, wait
- **camera**: camera, zoom, frame, view
- **coordinate**: coordinate, axes, graph, plot
- **color**: color, fill, stroke, opacity
- **movement**: move, shift, rotate, scale
- **creation**: create, write, draw, show, fade

## Usage Examples

### Basic Code Generation

```python
# Generate code with Context7 documentation
code, response = generator.generate_manim_code(
    topic="geometric_animations",
    description="Create animated geometric shapes",
    scene_outline="Circle to square transformation",
    scene_implementation="Transform a circle into a square with animation",
    scene_number=1
)
```

### Error Fixing

```python
# Fix code errors using Context7 documentation
fixed_code, response = generator.fix_code_errors(
    implementation_plan="Original implementation plan",
    code="Buggy code with errors",
    error="Error message from Manim",
    scene_trace_id="trace_id",
    topic="animations",
    scene_number=1,
    session_id="session_id"
)
```

## Benefits Over Traditional RAG

### Real-time Documentation
- Always uses current Manim Community documentation
- No need to maintain local documentation copies
- Automatic updates when Manim documentation changes

### Improved Accuracy
- Context7 provides structured, high-quality documentation
- Better code examples and explanations
- Reduced hallucination from outdated training data

### Performance Optimization
- Intelligent caching with TTL management
- Connection pooling for MCP servers
- Memory-efficient documentation storage

## Configuration

### MCP Configuration

Ensure your `mcp_config.json` includes Context7 server:

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"],
      "disabled": false
    }
  }
}
```

### Environment Variables

Set up your LLM provider API keys:

```bash
export OPENAI_API_KEY="your-openai-key"
# or
export ANTHROPIC_API_KEY="your-anthropic-key"
# or  
export GROQ_API_KEY="your-groq-key"
```

## Fallback Behavior

The integration includes robust fallback mechanisms:

1. **MCP Unavailable**: Falls back to traditional RAG if configured
2. **Context7 Error**: Returns basic documentation placeholder
3. **Network Issues**: Uses cached documentation when available
4. **Import Errors**: Gracefully degrades to non-MCP functionality

## Caching Strategy

### Documentation Caching
- 30-minute TTL for documentation sections
- 24-hour TTL for library ID resolution
- Automatic cache invalidation on errors

### File-based Caching
- Scene-specific cache directories
- JSON format for easy debugging
- Separate caches for code generation and error fixing

## Error Handling

The integration includes comprehensive error handling:

- **Connection Errors**: Retry with exponential backoff
- **Documentation Retrieval Errors**: Fallback to cached or basic docs
- **Async Context Issues**: Thread pool execution for sync contexts
- **Import Errors**: Graceful degradation without MCP features

## Performance Considerations

### Async Execution
- Handles both sync and async contexts
- Thread pool execution when needed
- Proper event loop management

### Resource Management
- Connection pooling for MCP servers
- Memory-efficient documentation storage
- Automatic cleanup on client close

### Monitoring
- Cache hit/miss statistics
- Retrieval timing metrics
- Error rate tracking

## Testing

Run the integration tests:

```bash
cd agents
python test_context7_integration.py
```

Run the example demonstration:

```bash
cd agents
python examples/context7_code_generator_example.py
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure MCP dependencies are installed
2. **Connection Failures**: Check Context7 server configuration
3. **Timeout Errors**: Increase timeout settings in MCP client
4. **Cache Issues**: Clear cache directories if needed

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

Planned improvements include:
- Support for additional MCP documentation servers
- Enhanced topic detection with ML models
- Real-time documentation quality scoring
- Integration with Manim plugin documentation