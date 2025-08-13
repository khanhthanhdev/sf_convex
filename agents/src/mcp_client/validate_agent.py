"""
Simple validation script for MCPAgent implementation.
Tests basic imports and initialization without requiring full MCP setup.
"""

import sys
import os
from pathlib import Path

# Add paths for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent.parent.parent))  # Add project root
sys.path.append(str(current_dir.parent))
sys.path.append(str(current_dir))

# Load environment variables from agents/.env
from dotenv import load_dotenv
env_path = current_dir.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test basic imports
        import sys
        sys.path.append('.')
        from agents.src.mcp_client.agent import MCPAgent, AgentConfig, AgentResponse, AgentError, LLMProviderError
        print("✓ Agent classes imported successfully")
        
        from agents.src.mcp_client.agent import create_manim_agent, quick_manim_query
        print("✓ Convenience functions imported successfully")
        
        # Test that we can create config
        config = AgentConfig(
            max_steps=5,
            timeout_seconds=60,
            temperature=0.1
        )
        print("✓ AgentConfig created successfully")
        print(f"  Max steps: {config.max_steps}")
        print(f"  Timeout: {config.timeout_seconds}")
        print(f"  Temperature: {config.temperature}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_langchain_dependencies():
    """Test LangChain dependencies."""
    print("\nTesting LangChain dependencies...")
    
    try:
        from langchain_openai import ChatOpenAI
        print("✓ LangChain OpenAI imported")
        
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        print("✓ LangChain agents imported")
        
        from langchain.tools import Tool
        print("✓ LangChain tools imported")
        
        from langchain_core.prompts import ChatPromptTemplate
        print("✓ LangChain prompts imported")
        
        return True
        
    except ImportError as e:
        print(f"✗ LangChain dependency missing: {e}")
        print("  Install with: pip install langchain-openai")
        return False


def test_agent_class_structure():
    """Test agent class structure and methods."""
    print("\nTesting agent class structure...")
    
    try:
        from agents.src.mcp_client.agent import MCPAgent, AgentConfig
        
        # Check class methods exist
        methods_to_check = [
            'create_with_openai',
            'create_with_anthropic', 
            'create_with_groq',
            'initialize',
            'query',
            'get_agent_stats',
            'close'
        ]
        
        for method in methods_to_check:
            if hasattr(MCPAgent, method):
                print(f"✓ Method {method} exists")
            else:
                print(f"✗ Method {method} missing")
                return False
        
        # Test system prompt
        if hasattr(MCPAgent, 'DEFAULT_MANIM_SYSTEM_PROMPT'):
            prompt = MCPAgent.DEFAULT_MANIM_SYSTEM_PROMPT
            if 'Manim' in prompt and 'Context7' in prompt:
                print("✓ Default system prompt contains Manim and Context7 references")
            else:
                print("✗ Default system prompt missing expected content")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Class structure test failed: {e}")
        return False


def test_environment_variables():
    """Test environment variable configuration."""
    print("\nTesting environment variables...")
    
    if os.getenv('OPENAI_API_KEY'):
        print("✓ OpenAI API key found")
        return True
    else:
        print("✗ OpenAI API key not set")
        print("  Set OPENAI_API_KEY environment variable")
        return False


def test_mcp_integration():
    """Test MCP integration components."""
    print("\nTesting MCP integration...")
    
    try:
        from agents.src.mcp_client.client import MCPClient
        print("✓ MCPClient imported")
        
        from agents.src.mcp_client.context7_docs import Context7DocsRetriever
        print("✓ Context7DocsRetriever imported")
        
        # Test that agent can be created with mock client
        from agents.src.mcp_client.agent import MCPAgent, AgentConfig
        
        # We can't test full initialization without MCP server,
        # but we can test class creation
        print("✓ MCP integration components available")
        
        return True
        
    except Exception as e:
        print(f"✗ MCP integration test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("MCPAgent Implementation Validation")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("LangChain Dependencies", test_langchain_dependencies),
        ("Agent Class Structure", test_agent_class_structure),
        ("Environment Variables", test_environment_variables),
        ("MCP Integration", test_mcp_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
    
    print(f"\n{'=' * 50}")
    print(f"Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validation tests passed!")
        print("\nNext steps:")
        print("1. Ensure MCP server is configured and running")
        print("2. Test with example_agent_usage.py")
        print("3. Run full integration tests")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        print("\nPlease fix the failing tests before proceeding.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)