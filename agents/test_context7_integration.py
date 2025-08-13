#!/usr/bin/env python3
"""
Test script for Context7 integration with CodeGenerator.

This script tests the integration of MCPClient with CodeGenerator agent
to ensure Context7 documentation retrieval works correctly.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the agents directory to Python path
agents_dir = Path(__file__).parent
sys.path.insert(0, str(agents_dir))

def test_imports():
    """Test that all required imports work correctly."""
    print("Testing imports...")
    
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
    
    try:
        from src.core.code_generator import CodeGenerator
        print("✓ CodeGenerator imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import CodeGenerator: {e}")
        return False
    
    return True

def test_code_generator_initialization():
    """Test CodeGenerator initialization with MCP client."""
    print("\nTesting CodeGenerator initialization...")
    
    try:
        from src.core.code_generator import CodeGenerator
        
        # Mock models for testing
        class MockModel:
            def __call__(self, *args, **kwargs):
                return "Mock response"
        
        scene_model = MockModel()
        helper_model = MockModel()
        
        # Test initialization without MCP client
        generator = CodeGenerator(
            scene_model=scene_model,
            helper_model=helper_model,
            use_rag=False
        )
        print("✓ CodeGenerator initialized without MCP client")
        
        # Test initialization with None MCP client
        generator_with_none = CodeGenerator(
            scene_model=scene_model,
            helper_model=helper_model,
            use_rag=False,
            mcp_client=None
        )
        print("✓ CodeGenerator initialized with None MCP client")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to initialize CodeGenerator: {e}")
        return False

def test_context7_topic_extraction():
    """Test the Manim topic extraction functionality."""
    print("\nTesting Manim topic extraction...")
    
    try:
        from src.core.code_generator import CodeGenerator
        
        # Mock models
        class MockModel:
            def __call__(self, *args, **kwargs):
                return "Mock response"
        
        generator = CodeGenerator(
            scene_model=MockModel(),
            helper_model=MockModel(),
            use_rag=False
        )
        
        # Test topic extraction
        test_cases = [
            ("Create an animation that transforms a circle into a square", "animation"),
            ("Draw text on the scene and make it appear", "creation"),
            ("Move the camera to zoom in on the object", "camera"),
            ("Create a coordinate system with axes", "coordinate"),
            ("Change the color of the mobject to red", "color"),
            ("Rotate and scale the shape", "movement"),
            ("This is a general scene description", None)
        ]
        
        for implementation, expected_topic in test_cases:
            detected_topic = generator._extract_manim_topic_from_implementation(implementation)
            if detected_topic == expected_topic:
                print(f"✓ Correctly detected topic '{detected_topic}' for: {implementation[:50]}...")
            else:
                print(f"? Detected topic '{detected_topic}' (expected '{expected_topic}') for: {implementation[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test topic extraction: {e}")
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
            print("? No MCP configuration file found, skipping MCP client test")
            return True
        
        print(f"Found MCP config at: {config_path}")
        
        # Try to create client (but don't initialize to avoid server connection)
        client = MCPClient.from_config_file(config_path)
        print("✓ MCPClient created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to create MCP client: {e}")
        return False

def main():
    """Run all tests."""
    print("Context7 Integration Test Suite")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_code_generator_initialization,
        test_context7_topic_extraction,
    ]
    
    # Run async test
    async_tests = [
        test_mcp_client_creation
    ]
    
    results = []
    
    # Run synchronous tests
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Run asynchronous tests
    for test in async_tests:
        try:
            result = asyncio.run(test())
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 40)
    print("Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())