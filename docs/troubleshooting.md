# Troubleshooting Guide: MCP-Use Context7 Integration

This guide helps you resolve common issues when using mcp-use with Context7 for Manim Community documentation access.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Configuration Problems](#configuration-problems)
3. [Connection Errors](#connection-errors)
4. [Agent Execution Issues](#agent-execution-issues)
5. [Documentation Retrieval Problems](#documentation-retrieval-problems)
6. [Performance Issues](#performance-issues)
7. [API Key and Authentication](#api-key-and-authentication)
8. [Common Error Messages](#common-error-messages)

## Installation Issues

### Problem: `mcp-use` not found after installation

**Symptoms:**
```bash
ModuleNotFoundError: No module named 'mcp_use'
```

**Solutions:**
1. Verify installation:
   ```bash
   pip list | grep mcp-use
   ```

2. Reinstall with correct package name:
   ```bash
   pip install mcp-use[langchain]
   ```

3. Check Python environment:
   ```bash
   which python
   pip --version
   ```

4. For virtual environments:
   ```bash
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   pip install mcp-use[langchain]
   ```

### Problem: Context7 MCP server not available

**Symptoms:**
```bash
Error: Context7 MCP server command not found
```

**Solutions:**
1. Install Node.js and npm:
   ```bash
   # Check if Node.js is installed
   node --version
   npm --version
   ```

2. Verify Context7 MCP server:
   ```bash
   npx -y @upstash/context7-mcp --help
   ```

3. Alternative installation:
   ```bash
   npm install -g @upstash/context7-mcp
   ```

## Configuration Problems

### Problem: Invalid MCP configuration file

**Symptoms:**
```python
FileNotFoundError: mcp_config.json not found
# or
JSONDecodeError: Invalid JSON format
```

**Solutions:**
1. Create proper configuration file:
   ```json
   {
     "mcpServers": {
       "context7": {
         "command": "npx",
         "args": ["-y", "@upstash/context7-mcp"]
       }
     }
   }
   ```

2. Validate JSON format:
   ```bash
   python -m json.tool mcp_config.json
   ```

3. Check file permissions:
   ```bash
   ls -la mcp_config.json
   chmod 644 mcp_config.json
   ```

### Problem: Wrong server configuration

**Symptoms:**
```python
Error: Server 'context7' not found in configuration
```

**Solutions:**
1. Verify server name matches exactly:
   ```json
   {
     "mcpServers": {
       "context7": {  // Must match exactly
         "command": "npx",
         "args": ["-y", "@upstash/context7-mcp"]
       }
     }
   }
   ```

2. Check for typos in configuration keys
3. Ensure proper JSON structure with nested objects

## Connection Errors

### Problem: MCP server connection timeout

**Symptoms:**
```python
TimeoutError: Connection to MCP server timed out
```

**Solutions:**
1. Increase timeout in client initialization:
   ```python
   client = MCPClient.from_config_file(
       "mcp_config.json",
       timeout=30  # Increase timeout
   )
   ```

2. Check network connectivity:
   ```bash
   ping google.com
   ```

3. Verify server command works independently:
   ```bash
   npx -y @upstash/context7-mcp
   ```

### Problem: Server process fails to start

**Symptoms:**
```python
Error: Failed to start MCP server process
```

**Solutions:**
1. Check server command manually:
   ```bash
   npx -y @upstash/context7-mcp --version
   ```

2. Verify Node.js version compatibility:
   ```bash
   node --version  # Should be >= 16
   ```

3. Clear npm cache:
   ```bash
   npm cache clean --force
   ```

4. Use absolute paths if needed:
   ```json
   {
     "mcpServers": {
       "context7": {
         "command": "/usr/local/bin/npx",
         "args": ["-y", "@upstash/context7-mcp"]
       }
     }
   }
   ```

## Agent Execution Issues

### Problem: Agent queries fail or hang

**Symptoms:**
```python
# Agent hangs indefinitely
# or
Error: Agent execution failed after max steps
```

**Solutions:**
1. Reduce max_steps for testing:
   ```python
   agent = MCPAgent(
       llm=llm,
       client=client,
       max_steps=5,  # Start with lower value
       timeout=30    # Add timeout
   )
   ```

2. Check LLM API connectivity:
   ```python
   from langchain_openai import ChatOpenAI
   llm = ChatOpenAI(model="gpt-4o")
   response = llm.invoke("Hello")
   print(response.content)
   ```

3. Enable debug logging:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

### Problem: Agent provides incorrect or outdated information

**Symptoms:**
- Agent responses don't match current Manim documentation
- Code examples use deprecated syntax

**Solutions:**
1. Verify Context7 is retrieving current docs:
   ```python
   # Test Context7 directly
   tools = client.list_tools()
   print([tool.name for tool in tools])
   ```

2. Check library ID resolution:
   ```python
   # Should resolve to /manimcommunity/manim
   result = client.call_tool("resolve-library-id", {"libraryName": "manim"})
   print(result)
   ```

3. Update system prompt to emphasize current documentation:
   ```python
   system_prompt = """You MUST use Context7 tools to retrieve current Manim documentation.
   Never rely on training data for Manim examples. Always fetch fresh documentation."""
   ```

## Documentation Retrieval Problems

### Problem: No documentation found for queries

**Symptoms:**
```python
Error: No relevant documentation found for query
```

**Solutions:**
1. Test library ID resolution:
   ```python
   result = client.call_tool("resolve-library-id", {"libraryName": "manim community"})
   ```

2. Try different query phrasings:
   ```python
   # Instead of: "How to animate?"
   # Try: "Manim animation methods"
   # Or: "Manim Community animation tutorial"
   ```

3. Check available libraries:
   ```python
   libraries = client.call_tool("resolve-library-id", {"libraryName": "manim"})
   print(libraries)
   ```

### Problem: Retrieved documentation is incomplete

**Symptoms:**
- Partial code examples
- Missing context or explanations

**Solutions:**
1. Increase documentation token limit:
   ```python
   result = client.call_tool("get-library-docs", {
       "context7CompatibleLibraryID": "/manimcommunity/manim",
       "tokens": 15000,  # Increase from default
       "topic": "animations"
   })
   ```

2. Use more specific topics:
   ```python
   # Instead of general queries, use specific topics
   topics = ["animations", "mobjects", "scenes", "transforms"]
   ```

## Performance Issues

### Problem: Slow response times

**Symptoms:**
- Queries take > 30 seconds
- High memory usage

**Solutions:**
1. Implement caching:
   ```python
   import functools
   
   @functools.lru_cache(maxsize=100)
   def cached_query(query_text):
       return agent.query(query_text)
   ```

2. Reduce documentation retrieval:
   ```python
   # Use smaller token limits
   result = client.call_tool("get-library-docs", {
       "context7CompatibleLibraryID": "/manimcommunity/manim",
       "tokens": 5000,  # Smaller limit
       "topic": "specific_topic"
   })
   ```

3. Optimize LLM settings:
   ```python
   llm = ChatOpenAI(
       model="gpt-4o-mini",  # Faster model
       temperature=0.1,      # Lower temperature
       max_tokens=1000       # Limit response length
   )
   ```

## API Key and Authentication

### Problem: OpenAI API key errors

**Symptoms:**
```python
AuthenticationError: Invalid API key
```

**Solutions:**
1. Verify API key format:
   ```bash
   echo $OPENAI_API_KEY  # Should start with sk-
   ```

2. Set environment variable properly:
   ```bash
   export OPENAI_API_KEY="sk-your-key-here"
   # or create .env file
   echo "OPENAI_API_KEY=sk-your-key-here" > .env
   ```

3. Test API key directly:
   ```python
   import openai
   client = openai.OpenAI()
   response = client.chat.completions.create(
       model="gpt-3.5-turbo",
       messages=[{"role": "user", "content": "Hello"}]
   )
   ```

### Problem: Rate limiting errors

**Symptoms:**
```python
RateLimitError: Rate limit exceeded
```

**Solutions:**
1. Implement retry with backoff:
   ```python
   import time
   import random
   
   def retry_with_backoff(func, max_retries=3):
       for attempt in range(max_retries):
           try:
               return func()
           except RateLimitError:
               wait_time = (2 ** attempt) + random.uniform(0, 1)
               time.sleep(wait_time)
       raise
   ```

2. Use different LLM provider:
   ```python
   from langchain_anthropic import ChatAnthropic
   llm = ChatAnthropic(model="claude-3-sonnet-20240229")
   ```

## Common Error Messages

### `ModuleNotFoundError: No module named 'langchain_openai'`

**Solution:**
```bash
pip install langchain-openai
```

### `JSONDecodeError: Expecting value: line 1 column 1 (char 0)`

**Solution:**
Check if mcp_config.json is empty or corrupted:
```bash
cat mcp_config.json
# Should show valid JSON, not empty file
```

### `PermissionError: [Errno 13] Permission denied`

**Solution:**
```bash
chmod +x examples/basic_usage.py
# or
python examples/basic_usage.py
```

### `ImportError: cannot import name 'MCPClient' from 'mcp_use'`

**Solution:**
Update to latest version:
```bash
pip install --upgrade mcp-use[langchain]
```

## Getting Help

If you're still experiencing issues:

1. **Check the logs**: Enable debug logging to see detailed error information
2. **Verify versions**: Ensure all dependencies are up to date
3. **Test components individually**: Test MCP client, LLM, and Context7 separately
4. **Check documentation**: Review the latest mcp-use and Context7 documentation
5. **Create minimal reproduction**: Isolate the problem with a simple test case

### Debug Information Collection

When reporting issues, include:

```python
import sys
import platform
print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")

# Package versions
import pkg_resources
packages = ['mcp-use', 'langchain', 'langchain-openai']
for package in packages:
    try:
        version = pkg_resources.get_distribution(package).version
        print(f"{package}: {version}")
    except:
        print(f"{package}: Not installed")
```

This information helps diagnose environment-specific issues.