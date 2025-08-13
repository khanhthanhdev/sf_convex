# Usage Guide: MCP-Use Context7 Integration

This guide provides comprehensive instructions for using the mcp-use library with Context7 integration to create AI agents that can access Manim Community documentation.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Basic Usage](#basic-usage)
3. [Advanced Usage](#advanced-usage)
4. [Configuration Options](#configuration-options)
5. [Best Practices](#best-practices)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- OpenAI API key (or other LLM provider)

### Installation

```bash
# Install mcp-use with LangChain support
pip install mcp-use[langchain]

# Verify Context7 MCP server is available
npx -y @upstash/context7-mcp --help
```

### Basic Setup

1. Create configuration file:
```bash
cp examples/configurations/basic_config.json mcp_config.json
```

2. Set environment variables:
```bash
export OPENAI_API_KEY="sk-your-openai-key"
```

3. Run basic example:
```bash
python examples/basic_usage.py
```

## Basic Usage

### Creating Your First Agent

```python
from mcp_use import MCPClient, MCPAgent
from langchain_openai import ChatOpenAI
import os

# Initialize MCP client
client = MCPClient.from_config_file("mcp_config.json")

# Create LLM instance
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Create agent
agent = MCPAgent(
    llm=llm,
    client=client,
    max_steps=10,
    system_prompt="You are a Manim expert with access to current documentation."
)

# Query the agent
response = agent.query("How do I create a circle animation in Manim?")
print(response)
```

### Simple Query Examples

```python
# Basic animation question
response = agent.query("Show me how to animate text in Manim")

# Specific functionality
response = agent.query("What's the syntax for Transform animation?")

# Troubleshooting
response = agent.query("Why isn't my Manim animation rendering?")
```

## Advanced Usage

### Multi-Step Conversations

```python
from examples.advanced_usage import AdvancedManimAgent

# Create advanced agent with conversation history
agent = AdvancedManimAgent()

# Multi-step conversation
responses = []
queries = [
    "I want to create a mathematical visualization of derivatives",
    "How do I start with a basic function graph?",
    "Now add a tangent line that moves along the curve",
    "Finally, add explanatory text labels"
]

for query in queries:
    response = agent.query_with_context(query)
    responses.append(response)
    print(f"Q: {query}")
    print(f"A: {response}\n")

# Get performance metrics
metrics = agent.get_performance_summary()
print(f"Average response time: {metrics['avg_execution_time']:.2f}s")
```

### Custom System Prompts

```python
# Specialized prompts for different use cases
prompts = {
    "beginner": """You are a patient Manim tutor for beginners. 
    Always provide step-by-step explanations and simple examples.""",
    
    "advanced": """You are an expert Manim consultant. 
    Provide optimized solutions and advanced techniques.""",
    
    "debugging": """You are a Manim debugging specialist. 
    Focus on identifying and fixing common issues."""
}

# Create specialized agents
agents = {}
for role, prompt in prompts.items():
    agents[role] = MCPAgent(
        llm=llm,
        client=client,
        system_prompt=prompt
    )

# Use appropriate agent for the task
beginner_response = agents["beginner"].query("How do I start with Manim?")
debug_response = agents["debugging"].query("My animation isn't working")
```

### Error Handling and Retries

```python
import time
import random
from typing import Optional

def robust_query(agent: MCPAgent, query: str, max_retries: int = 3) -> Optional[str]:
    """Query agent with retry logic and error handling."""
    
    for attempt in range(max_retries):
        try:
            response = agent.query(query)
            return response
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
            else:
                print(f"All {max_retries} attempts failed")
                return None
    
    return None

# Usage
response = robust_query(agent, "Complex Manim question")
if response:
    print(response)
else:
    print("Failed to get response after retries")
```

## Configuration Options

### Environment-Specific Configurations

Choose the appropriate configuration for your environment:

```bash
# Development
cp examples/configurations/development_config.json mcp_config.json

# Production
cp examples/configurations/production_config.json mcp_config.json

# High Performance
cp examples/configurations/performance_optimized_config.json mcp_config.json

# Maximum Security
cp examples/configurations/secure_config.json mcp_config.json
```

### Custom Configuration

Create your own configuration by combining options:

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"],
      "env": {
        "NODE_ENV": "production"
      }
    }
  },
  "agentDefaults": {
    "maxSteps": 10,
    "temperature": 0.1,
    "model": "gpt-4o",
    "enableCaching": true
  },
  "security": {
    "restrictedTools": ["shell", "filesystem_write"]
  }
}
```

### Multiple LLM Providers

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

# Configure multiple providers
llm_providers = {
    "openai": ChatOpenAI(model="gpt-4o"),
    "anthropic": ChatAnthropic(model="claude-3-sonnet-20240229"),
    "groq": ChatGroq(model="llama3-70b-8192")
}

# Create agents with different providers
agents = {}
for name, llm in llm_providers.items():
    agents[name] = MCPAgent(llm=llm, client=client)

# Use different agents for different tasks
quick_response = agents["groq"].query("Quick Manim question")
detailed_response = agents["openai"].query("Complex Manim explanation")
```

## Best Practices

### 1. Query Optimization

```python
# Good: Specific, actionable queries
"How do I animate a circle moving from left to right in Manim?"

# Better: Include context
"I'm creating a physics simulation. How do I animate a circle representing a ball moving with constant velocity in Manim?"

# Best: Specify requirements
"For a Manim physics simulation, show me how to animate a red circle moving horizontally with smooth motion, including the complete scene code."
```

### 2. Error Handling

```python
def safe_agent_query(agent, query, fallback_response="Unable to process query"):
    try:
        response = agent.query(query)
        if not response or len(response.strip()) < 10:
            return fallback_response
        return response
    except Exception as e:
        print(f"Agent query failed: {e}")
        return fallback_response
```

### 3. Performance Optimization

```python
# Use caching for repeated queries
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_manim_query(query_text):
    return agent.query(query_text)

# Batch similar queries
queries = [
    "Manim circle animation",
    "Manim square animation", 
    "Manim triangle animation"
]

responses = []
for query in queries:
    response = cached_manim_query(query)
    responses.append(response)
```

### 4. Resource Management

```python
# Use context managers for proper cleanup
class ManagedAgent:
    def __init__(self, config_file):
        self.client = None
        self.agent = None
        self.config_file = config_file
    
    def __enter__(self):
        self.client = MCPClient.from_config_file(self.config_file)
        self.agent = MCPAgent(llm=llm, client=self.client)
        return self.agent
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            self.client.close()

# Usage
with ManagedAgent("mcp_config.json") as agent:
    response = agent.query("Manim question")
    print(response)
```

## Examples

### Example 1: Interactive Manim Tutorial

```python
def interactive_manim_tutorial():
    """Interactive tutorial that adapts to user responses."""
    
    agent = MCPAgent(llm=llm, client=client)
    
    print("Welcome to the Interactive Manim Tutorial!")
    print("I'll help you learn Manim step by step.\n")
    
    # Assess user level
    level_query = input("What's your experience with Manim? (beginner/intermediate/advanced): ")
    
    # Customize approach based on level
    if level_query.lower() == "beginner":
        topics = [
            "What is Manim and how do I install it?",
            "How do I create my first simple animation?",
            "How do I add text to my animations?",
            "How do I animate shapes moving around?"
        ]
    else:
        topics = [
            "Advanced animation techniques in Manim",
            "Custom mobject creation",
            "Complex mathematical visualizations",
            "Performance optimization for large scenes"
        ]
    
    for i, topic in enumerate(topics, 1):
        print(f"\n--- Lesson {i}: {topic} ---")
        response = agent.query(f"Explain {topic} with a practical example")
        print(response)
        
        input("\nPress Enter to continue to the next lesson...")

# Run tutorial
interactive_manim_tutorial()
```

### Example 2: Manim Code Generator

```python
def generate_manim_scene(description):
    """Generate complete Manim scene code from description."""
    
    prompt = f"""
    Generate a complete, runnable Manim scene based on this description: {description}
    
    Requirements:
    - Include all necessary imports
    - Create a complete Scene class
    - Add proper construct method
    - Include comments explaining each step
    - Make sure the code is syntactically correct
    """
    
    agent = MCPAgent(
        llm=llm, 
        client=client,
        system_prompt="You are a Manim code generator. Always provide complete, runnable code."
    )
    
    return agent.query(prompt)

# Usage examples
scenes = [
    "A red circle that grows and then shrinks",
    "Mathematical function f(x) = x^2 with animated plotting",
    "Text that writes itself and then transforms into a shape"
]

for description in scenes:
    print(f"\n--- Scene: {description} ---")
    code = generate_manim_scene(description)
    print(code)
    print("\n" + "="*60)
```

### Example 3: Manim Debugging Assistant

```python
def debug_manim_code(code, error_message):
    """Help debug Manim code issues."""
    
    debug_prompt = f"""
    I'm having trouble with this Manim code:
    
    ```python
    {code}
    ```
    
    Error message: {error_message}
    
    Please help me identify and fix the issue. Provide:
    1. Explanation of what's wrong
    2. Corrected code
    3. Tips to avoid similar issues
    """
    
    agent = MCPAgent(
        llm=llm,
        client=client,
        system_prompt="You are a Manim debugging expert. Help identify and fix code issues."
    )
    
    return agent.query(debug_prompt)

# Usage
buggy_code = """
from manim import *

class MyScene(Scene):
    def construct(self):
        circle = Circle()
        self.play(Create(circle))
        self.play(circle.animate.shift(RIGHT * 2))
"""

error = "AttributeError: 'Circle' object has no attribute 'animate'"

solution = debug_manim_code(buggy_code, error)
print(solution)
```

## Troubleshooting

For detailed troubleshooting information, see [troubleshooting.md](troubleshooting.md).

### Quick Fixes

**Agent not responding:**
```python
# Check if client is connected
print(client.list_tools())

# Verify LLM is working
test_response = llm.invoke("Hello")
print(test_response.content)
```

**Slow responses:**
```python
# Use faster model
llm = ChatOpenAI(model="gpt-4o-mini")

# Reduce max_steps
agent = MCPAgent(llm=llm, client=client, max_steps=5)
```

**Documentation not found:**
```python
# Test Context7 directly
result = client.call_tool("resolve-library-id", {"libraryName": "manim"})
print(result)
```

## Next Steps

1. **Explore Examples**: Try running all examples in the `examples/` directory
2. **Customize Configuration**: Modify configurations for your specific needs
3. **Build Applications**: Create your own Manim-focused applications
4. **Contribute**: Share your configurations and examples with the community

For more advanced topics and API reference, see the complete documentation.