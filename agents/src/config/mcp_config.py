"""
MCP (Model Context Protocol) Configuration Management

This module provides functionality for loading, validating, and managing MCP server configurations,
specifically for Context7 integration with Manim Community documentation.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator, ValidationError

from .models import MCPServerConfig, ValidationResult

logger = logging.getLogger(__name__)


class MCPConfig(BaseModel):
    """Complete MCP configuration containing all server configurations."""
    
    mcp_servers: Dict[str, MCPServerConfig] = Field(
        default_factory=dict, 
        description="MCP server configurations"
    )
    
    @field_validator('mcp_servers')
    @classmethod
    def validate_servers_not_empty(cls, v):
        if not v:
            logger.warning("No MCP servers configured")
        return v


class MCPConfigLoader:
    """Loader for MCP configuration files with validation and Context7 integration."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize the MCP configuration loader.
        
        Args:
            config_path: Path to MCP configuration file. If None, uses default locations.
        """
        self.config_path = self._resolve_config_path(config_path)
    
    def _resolve_config_path(self, config_path: Optional[Union[str, Path]]) -> Path:
        """Resolve the configuration file path.
        
        Args:
            config_path: Provided config path or None for default
            
        Returns:
            Resolved Path object
        """
        if config_path:
            return Path(config_path)
        
        # Try default locations in order of preference
        default_paths = [
            Path("mcp_config.json"),  # Project root
            Path(".kiro/settings/mcp.json"),  # Kiro workspace config
            Path.home() / ".kiro/settings/mcp.json",  # User global config
            Path(".cursor/mcp.json")  # Cursor IDE config
        ]
        
        for path in default_paths:
            if path.exists():
                logger.info(f"Using MCP config from: {path}")
                return path
        
        # Return the first default if none exist
        logger.info(f"No existing MCP config found, will use: {default_paths[0]}")
        return default_paths[0]
    
    def load_config(self) -> MCPConfig:
        """Load MCP configuration from file.
        
        Returns:
            MCPConfig object with loaded configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If config format is invalid
            json.JSONDecodeError: If JSON is malformed
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"MCP configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Convert to our internal format if needed
            if 'mcpServers' in config_data:
                # Convert from external format to internal format
                servers = {}
                for name, server_config in config_data['mcpServers'].items():
                    servers[name] = MCPServerConfig(
                        command=server_config['command'],
                        args=server_config.get('args', []),
                        env=server_config.get('env', {}),
                        disabled=server_config.get('disabled', False),
                        auto_approve=server_config.get('autoApprove', [])
                    )
                config_data = {'mcp_servers': servers}
            
            config = MCPConfig(**config_data)
            logger.info(f"Successfully loaded MCP config from {self.config_path}")
            return config
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in MCP config file: {e}")
            raise
        except ValidationError as e:
            logger.error(f"Invalid MCP configuration format: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading MCP configuration: {e}")
            raise
    
    def save_config(self, config: MCPConfig) -> None:
        """Save MCP configuration to file.
        
        Args:
            config: MCPConfig object to save
            
        Raises:
            OSError: If file cannot be written
        """
        try:
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to external format for saving
            config_data = {
                "mcpServers": {}
            }
            
            for name, server_config in config.mcp_servers.items():
                config_data["mcpServers"][name] = {
                    "command": server_config.command,
                    "args": server_config.args,
                    "env": server_config.env,
                    "disabled": server_config.disabled,
                    "autoApprove": server_config.auto_approve
                }
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully saved MCP config to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error saving MCP configuration: {e}")
            raise
    
    def get_context7_config(self) -> Optional[MCPServerConfig]:
        """Get Context7 server configuration if available.
        
        Returns:
            MCPServerConfig for Context7 or None if not configured
        """
        try:
            config = self.load_config()
            return config.mcp_servers.get('context7')
        except Exception as e:
            logger.error(f"Error getting Context7 config: {e}")
            return None


class MCPConfigValidator:
    """Validator for MCP configurations with Context7-specific checks."""
    
    def __init__(self):
        """Initialize the MCP configuration validator."""
        pass
    
    def validate_config(self, config: MCPConfig) -> ValidationResult:
        """Validate complete MCP configuration.
        
        Args:
            config: MCP configuration to validate
            
        Returns:
            ValidationResult with validation status and details
        """
        result = ValidationResult(valid=True)
        
        try:
            # Basic validation
            if not config.mcp_servers:
                result.add_warning("No MCP servers configured")
                return result
            
            # Validate each server configuration
            for server_name, server_config in config.mcp_servers.items():
                self._validate_server_config(server_name, server_config, result)
            
            # Context7-specific validation
            self._validate_context7_integration(config, result)
            
            logger.info(f"MCP configuration validation completed: {'Valid' if result.valid else 'Invalid'}")
            
        except Exception as e:
            result.add_error(f"MCP configuration validation failed: {str(e)}")
            logger.error(f"MCP configuration validation error: {e}")
        
        return result
    
    def _validate_server_config(self, server_name: str, server_config: MCPServerConfig, result: ValidationResult):
        """Validate individual server configuration.
        
        Args:
            server_name: Name of the server
            server_config: Server configuration to validate
            result: ValidationResult to update
        """
        # Validate command exists
        if not server_config.command:
            result.add_error(f"Server '{server_name}' has no command specified")
            return
        
        # Check if command is available (basic check)
        if server_config.command in ['npx', 'npm', 'node']:
            # These are common Node.js commands, assume they're available
            pass
        elif server_config.command.startswith('/') or server_config.command.startswith('~'):
            # Absolute path - check if file exists
            command_path = Path(server_config.command).expanduser()
            if not command_path.exists():
                result.add_error(f"Server '{server_name}' command not found: {server_config.command}")
        else:
            # Relative command - would need PATH checking, skip for now
            result.add_warning(f"Server '{server_name}' command availability not verified: {server_config.command}")
        
        # Validate arguments
        if not isinstance(server_config.args, list):
            result.add_error(f"Server '{server_name}' args must be a list")
        
        # Validate environment variables
        if not isinstance(server_config.env, dict):
            result.add_error(f"Server '{server_name}' env must be a dictionary")
        
        # Validate auto_approve list
        if not isinstance(server_config.auto_approve, list):
            result.add_error(f"Server '{server_name}' auto_approve must be a list")
    
    def _validate_context7_integration(self, config: MCPConfig, result: ValidationResult):
        """Validate Context7-specific integration requirements.
        
        Args:
            config: MCP configuration to validate
            result: ValidationResult to update
        """
        context7_config = config.mcp_servers.get('context7')
        
        if not context7_config:
            result.add_error("Context7 server not configured - required for Manim documentation integration")
            return
        
        if context7_config.disabled:
            result.add_error("Context7 server is disabled - required for Manim documentation integration")
            return
        
        # Validate Context7 command and arguments
        if context7_config.command != 'npx':
            result.add_warning(f"Context7 server uses non-standard command: {context7_config.command}")
        
        expected_args = ['-y', '@upstash/context7-mcp']
        if context7_config.args != expected_args:
            result.add_warning(f"Context7 server args differ from expected: {context7_config.args} vs {expected_args}")
        
        # Check for recommended auto-approve tools
        recommended_tools = [
            'mcp_context7_resolve_library_id',
            'mcp_context7_get_library_docs'
        ]
        
        missing_tools = [tool for tool in recommended_tools if tool not in context7_config.auto_approve]
        if missing_tools:
            result.add_warning(f"Context7 server missing recommended auto-approve tools: {missing_tools}")
        
        logger.info("Context7 integration validation completed")
    
    def validate_context7_manim_integration(self, config: MCPConfig) -> ValidationResult:
        """Validate Context7 configuration specifically for Manim Community integration.
        
        Args:
            config: MCP configuration to validate
            
        Returns:
            ValidationResult with Manim-specific validation status
        """
        result = ValidationResult(valid=True)
        
        context7_config = config.mcp_servers.get('context7')
        if not context7_config:
            result.add_error("Context7 server required for Manim Community documentation access")
            return result
        
        if context7_config.disabled:
            result.add_error("Context7 server must be enabled for Manim Community documentation access")
            return result
        
        # Validate that Context7 can access Manim Community library
        # This would require actual connection testing, which we'll note as a requirement
        result.add_warning("Context7 connection to Manim Community library (/manimcommunity/manim) should be tested")
        
        # Check for proper tool configuration
        required_tools = [
            'mcp_context7_resolve_library_id',
            'mcp_context7_get_library_docs'
        ]
        
        configured_tools = context7_config.auto_approve
        missing_tools = [tool for tool in required_tools if tool not in configured_tools]
        
        if missing_tools:
            result.add_error(f"Context7 missing required tools for Manim integration: {missing_tools}")
        
        logger.info("Context7 Manim integration validation completed")
        return result


def load_mcp_config(config_path: Optional[Union[str, Path]] = None) -> MCPConfig:
    """Convenience function to load MCP configuration.
    
    Args:
        config_path: Path to configuration file or None for default
        
    Returns:
        Loaded MCPConfig object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config format is invalid
    """
    loader = MCPConfigLoader(config_path)
    return loader.load_config()


def validate_mcp_config(config: MCPConfig) -> ValidationResult:
    """Convenience function to validate MCP configuration.
    
    Args:
        config: MCP configuration to validate
        
    Returns:
        ValidationResult with validation status
    """
    validator = MCPConfigValidator()
    return validator.validate_config(config)


def create_default_context7_config() -> MCPConfig:
    """Create a default MCP configuration with Context7 for Manim Community integration.
    
    Returns:
        MCPConfig with default Context7 configuration
    """
    context7_server = MCPServerConfig(
        command="npx",
        args=["-y", "@upstash/context7-mcp"],
        env={},
        disabled=False,
        auto_approve=[
            "mcp_context7_resolve_library_id",
            "mcp_context7_get_library_docs"
        ]
    )
    
    return MCPConfig(mcp_servers={"context7": context7_server})