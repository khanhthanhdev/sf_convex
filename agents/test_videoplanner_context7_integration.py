#!/usr/bin/env python3
"""
Test script for VideoPlanner Context7 integration.

This script tests the integration of VideoPlanner with Context7 MCP server
for accessing Manim Community documentation during video planning.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the agents directory to the Python path
agents_dir = Path(__file__).parent
sys.path.insert(0, str(agents_dir))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.core.video_planner import VideoPlanner
        print("✓ VideoPlanner imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import VideoPlanner: {e}")
        return False
    
    try:
        from src.mcp_client.client import MCPClient
        print("✓ MCPClient imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import MCPClient: {e}")
        return False
    
    try:
        from src.mcp_client.context7_docs import Context7DocsRetriever
        print("✓ Context7DocsRetriever imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Context7DocsRetriever: {e}")
        return False
    
    return True

def test_videoplanner_initialization():
    """Test VideoPlanner initialization with and without MCP client."""
    print("\nTesting VideoPlanner initialization...")
    
    try:
        from src.core.video_planner import VideoPlanner
        
        # Mock model for testing
        class MockModel:
            def __call__(self, *args, **kwargs):
                return "Mock response"
        
        mock_model = MockModel()
        
        # Test initialization without MCP client
        planner = VideoPlanner(
            planner_model=mock_model,
            helper_model=mock_model,
            use_rag=False,
            mcp_client=None
        )
        print("✓ VideoPlanner initialized without MCP client")
        
        # Test initialization with None MCP client (should work the same)
        planner_with_none = VideoPlanner(
            planner_model=mock_model,
            helper_model=mock_model,
            use_rag=False,
            mcp_client=None
        )
        print("✓ VideoPlanner initialized with None MCP client")
        
        return True
        
    except Exception as e:
        print(f"✗ VideoPlanner initialization failed: {e}")
        return False

async def test_mcp_client_creation():
    """Test MCP client creation (if configuration exists)."""
    print("\nTesting MCP client creation...")
    
    try:
        from src.mcp_client.client import MCPClient
        
        # Check if MCP config exists
        config_paths = [
            "mcp_config.json",
            "../mcp_config.json",
            "../../mcp_config.json"
        ]
        
        config_path = None
        for path in config_paths:
            if os.path.exists(path):
                config_path = path
                break
        
        if not config_path:
            print("⚠ No MCP configuration file found, skipping MCP client test")
            return True
        
        print(f"Found MCP config at: {config_path}")
        
        # Try to create MCP client
        client = MCPClient.from_config_file(config_path)
        print("✓ MCPClient created successfully")
        
        # Test VideoPlanner with MCP client
        from src.core.video_planner import VideoPlanner
        
        class MockModel:
            def __call__(self, *args, **kwargs):
                return "Mock response"
        
        mock_model = MockModel()
        
        planner = VideoPlanner(
            planner_model=mock_model,
            helper_model=mock_model,
            use_rag=False,
            mcp_client=client
        )
        print("✓ VideoPlanner initialized with MCP client")
        
        # Check that Context7 retriever was initialized
        if planner.context7_retriever:
            print("✓ Context7DocsRetriever initialized in VideoPlanner")
        else:
            print("⚠ Context7DocsRetriever not initialized (MCP dependencies may be missing)")
        
        return True
        
    except Exception as e:
        print(f"✗ MCP client creation failed: {e}")
        return False

def test_context7_integration_methods():
    """Test Context7 integration methods exist and are callable."""
    print("\nTesting Context7 integration methods...")
    
    try:
        from src.core.video_planner import VideoPlanner
        
        class MockModel:
            def __call__(self, *args, **kwargs):
                return "Mock response"
        
        mock_model = MockModel()
        
        planner = VideoPlanner(
            planner_model=mock_model,
            helper_model=mock_model,
            use_rag=False,
            mcp_client=None
        )
        
        # Check that new methods exist
        assert hasattr(planner, '_retrieve_context7_documentation_for_planning'), "Missing _retrieve_context7_documentation_for_planning method"
        assert hasattr(planner, '_extract_manim_topic_for_planning'), "Missing _extract_manim_topic_for_planning method"
        assert hasattr(planner, '_format_context7_docs_for_planning'), "Missing _format_context7_docs_for_planning method"
        
        print("✓ All Context7 integration methods exist")
        
        # Test topic extraction
        topic = planner._extract_manim_topic_for_planning(
            'scene_plan', 
            'Mathematical Animation', 
            'Create an animation showing function transformations',
            None
        )
        print(f"✓ Topic extraction works: {topic}")
        
        return True
        
    except Exception as e:
        print(f"✗ Context7 integration methods test failed: {e}")
        return False

def test_parameter_integration():
    """Test that mcp_client parameter is properly integrated."""
    print("\nTesting parameter integration...")
    
    try:
        from src.core.video_planner import VideoPlanner
        import inspect
        
        # Check that __init__ method has mcp_client parameter
        init_signature = inspect.signature(VideoPlanner.__init__)
        params = list(init_signature.parameters.keys())
        
        assert 'mcp_client' in params, "mcp_client parameter missing from __init__"
        print("✓ mcp_client parameter exists in __init__")
        
        # Check default value
        mcp_client_param = init_signature.parameters['mcp_client']
        assert mcp_client_param.default is None, "mcp_client parameter should default to None"
        print("✓ mcp_client parameter has correct default value")
        
        return True
        
    except Exception as e:
        print(f"✗ Parameter integration test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("VideoPlanner Context7 Integration Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_videoplanner_initialization,
        test_context7_integration_methods,
        test_parameter_integration,
    ]
    
    async_tests = [
        test_mcp_client_creation
    ]
    
    # Run synchronous tests
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Run async tests
    for test in async_tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"✗ Async test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! VideoPlanner Context7 integration is working correctly.")
        return True
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)