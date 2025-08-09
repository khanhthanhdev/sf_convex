"""
Configuration Service for .env file handling.

This module provides the ConfigurationService class that handles loading and parsing
of .env files with support for nested environment variables, multiple environment files,
and proper validation and type conversion.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from dotenv import load_dotenv, dotenv_values
from pydantic import ValidationError

from .models import SystemConfig, ValidationResult


logger = logging.getLogger(__name__)


class ConfigurationService:
    """Service for loading and parsing configuration from .env files.
    
    This service handles:
    - Loading from multiple .env files (development, production, etc.)
    - Nested environment variable parsing using pydantic-settings
    - Environment variable validation and type conversion
    - Support for environment-specific configuration profiles
    """
    
    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """Initialize the configuration service.
        
        Args:
            base_path: Base path for configuration files. Defaults to current directory.
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.loaded_files: List[Path] = []
        self.config_cache: Dict[str, Any] = {}
        self._last_reload_time: Optional[float] = None
        
        logger.info(f"ConfigurationService initialized with base path: {self.base_path}")
    
    def load_env_config(self, env_file: str = '.env') -> Dict[str, Any]:
        """Load configuration from a specific .env file.
        
        Args:
            env_file: Path to the .env file relative to base_path
            
        Returns:
            Dictionary containing the loaded configuration
            
        Raises:
            FileNotFoundError: If the specified .env file doesn't exist
            ValueError: If the .env file contains invalid syntax
        """
        env_path = self.base_path / env_file
        
        if not env_path.exists():
            logger.warning(f"Environment file not found: {env_path}")
            return {}
        
        try:
            # Load environment variables from file
            config_dict = dotenv_values(env_path)
            
            # Filter out None values and empty strings
            filtered_config = {
                key: value for key, value in config_dict.items() 
                if value is not None and value.strip() != ''
            }
            
            # Parse nested configuration
            parsed_config = self.parse_nested_config(filtered_config)
            
            # Track loaded file
            if env_path not in self.loaded_files:
                self.loaded_files.append(env_path)
            
            logger.info(f"Successfully loaded configuration from: {env_path}")
            logger.debug(f"Loaded {len(filtered_config)} configuration values")
            
            return parsed_config
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {env_path}: {e}")
            raise ValueError(f"Invalid .env file syntax in {env_path}: {e}")
    
    def load_multiple_env_files(self, env_files: List[str]) -> Dict[str, Any]:
        """Load configuration from multiple .env files with precedence order.
        
        Files are loaded in order, with later files overriding earlier ones.
        
        Args:
            env_files: List of .env file paths in precedence order (lowest to highest)
            
        Returns:
            Merged configuration dictionary
        """
        merged_config = {}
        
        for env_file in env_files:
            try:
                file_config = self.load_env_config(env_file)
                merged_config.update(file_config)
                logger.debug(f"Merged configuration from: {env_file}")
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Skipping {env_file}: {e}")
                continue
        
        logger.info(f"Successfully merged configuration from {len(env_files)} files")
        return merged_config
    
    def load_environment_specific_config(self, environment: str = None) -> Dict[str, Any]:
        """Load configuration for a specific environment with fallback chain.
        
        Loading order (highest to lowest precedence):
        1. .env.{environment}.local (e.g., .env.development.local)
        2. .env.{environment} (e.g., .env.development)
        3. .env.local
        4. .env
        
        Args:
            environment: Environment name (development, staging, production).
                        If None, attempts to detect from ENVIRONMENT variable.
                        
        Returns:
            Merged configuration dictionary
        """
        if environment is None:
            environment = os.getenv('ENVIRONMENT', 'development')
        
        # Define file loading order (lowest to highest precedence)
        env_files = [
            '.env',
            '.env.local',
            f'.env.{environment}',
            f'.env.{environment}.local'
        ]
        
        logger.info(f"Loading configuration for environment: {environment}")
        config = self.load_multiple_env_files(env_files)
        
        # Add environment to config if not present
        if 'environment' not in config:
            config['environment'] = environment
        
        return config
    
    def parse_nested_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Parse nested environment variables using delimiter.
        
        Converts flat environment variables with delimiters into nested dictionaries.
        Example: 'LLM_PROVIDERS__OPENAI__API_KEY' becomes nested structure.
        
        Args:
            config_dict: Flat configuration dictionary
            
        Returns:
            Nested configuration dictionary
        """
        nested_config = {}
        delimiter = '__'  # Using double underscore as delimiter
        
        for key, value in config_dict.items():
            if delimiter in key:
                # Split key into parts
                parts = key.split(delimiter)
                
                # Navigate/create nested structure
                current_dict = nested_config
                for part in parts[:-1]:
                    part_lower = part.lower()
                    if part_lower not in current_dict:
                        current_dict[part_lower] = {}
                    current_dict = current_dict[part_lower]
                
                # Set the final value
                final_key = parts[-1].lower()
                current_dict[final_key] = self._convert_env_value(value)
            else:
                # Direct key-value pair
                nested_config[key.lower()] = self._convert_env_value(value)
        
        return nested_config
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate Python type.
        
        Args:
            value: String value from environment variable
            
        Returns:
            Converted value (bool, int, float, list, or str)
        """
        if not isinstance(value, str):
            return value
        
        value = value.strip()
        
        # Boolean conversion
        if value.lower() in ('true', '1', 'yes', 'on'):
            return True
        elif value.lower() in ('false', '0', 'no', 'off'):
            return False
        
        # List conversion (comma-separated values)
        if ',' in value:
            return [item.strip() for item in value.split(',') if item.strip()]
        
        # Numeric conversion
        try:
            # Try integer first
            if '.' not in value:
                return int(value)
            else:
                return float(value)
        except ValueError:
            pass
        
        # Return as string if no conversion applies
        return value
    
    def get_env_value(self, key: str, default: Any = None, cast_type: type = str) -> Any:
        """Get environment variable value with type casting and default.
        
        Args:
            key: Environment variable key
            default: Default value if key not found
            cast_type: Type to cast the value to
            
        Returns:
            Environment variable value cast to specified type
        """
        value = os.getenv(key)
        
        if value is None:
            return default
        
        try:
            if cast_type == bool:
                return self._convert_env_value(value) if isinstance(self._convert_env_value(value), bool) else bool(value)
            elif cast_type == int:
                return int(value)
            elif cast_type == float:
                return float(value)
            elif cast_type == list:
                converted = self._convert_env_value(value)
                return converted if isinstance(converted, list) else [value]
            else:
                return cast_type(value)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to cast {key}='{value}' to {cast_type.__name__}: {e}")
            return default
    
    def reload_env_config(self, env_file: str = '.env') -> bool:
        """Reload configuration from .env file and update environment.
        
        Args:
            env_file: Path to the .env file to reload
            
        Returns:
            True if reload was successful, False otherwise
        """
        try:
            env_path = self.base_path / env_file
            
            if not env_path.exists():
                logger.warning(f"Cannot reload - file not found: {env_path}")
                return False
            
            # Load new configuration
            load_dotenv(env_path, override=True)
            
            # Clear cache to force reload
            self.config_cache.clear()
            
            # Update last reload time
            import time
            self._last_reload_time = time.time()
            
            logger.info(f"Successfully reloaded configuration from: {env_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reload configuration from {env_file}: {e}")
            return False
    
    def validate_env_config(self, config_dict: Dict[str, Any]) -> ValidationResult:
        """Validate environment configuration against SystemConfig model.
        
        Args:
            config_dict: Configuration dictionary to validate
            
        Returns:
            ValidationResult with validation status and errors
        """
        result = ValidationResult(valid=True)
        
        try:
            # Attempt to create SystemConfig from the configuration
            # This will trigger Pydantic validation
            system_config = SystemConfig(**config_dict)
            
            # Additional custom validations
            self._validate_required_keys(config_dict, result)
            self._validate_provider_configurations(config_dict, result)
            self._validate_file_paths(config_dict, result)
            
            logger.info("Configuration validation completed successfully")
            
        except ValidationError as e:
            result.valid = False
            for error in e.errors():
                field_path = ' -> '.join(str(loc) for loc in error['loc'])
                error_msg = f"Field '{field_path}': {error['msg']}"
                result.add_error(error_msg)
            
            logger.error(f"Configuration validation failed: {len(result.errors)} errors")
            
        except Exception as e:
            result.valid = False
            result.add_error(f"Unexpected validation error: {str(e)}")
            logger.error(f"Unexpected validation error: {e}")
        
        return result
    
    def _validate_required_keys(self, config_dict: Dict[str, Any], result: ValidationResult):
        """Validate that required configuration keys are present."""
        # Check for at least one LLM provider
        has_llm_provider = any(
            key.endswith('_API_KEY') and config_dict.get(key)
            for key in ['OPENAI_API_KEY', 'GEMINI_API_KEY', 'AWS_ACCESS_KEY_ID']
        )
        
        if not has_llm_provider:
            result.add_warning("No LLM provider API keys found - system may not function properly")
        
        # Check for RAG configuration if enabled
        if config_dict.get('RAG_ENABLED', True):
            embedding_provider = config_dict.get('EMBEDDING_PROVIDER', 'jina')
            if embedding_provider == 'jina' and not config_dict.get('JINA_API_KEY'):
                result.add_warning("RAG is enabled but JINA_API_KEY is missing")
            elif embedding_provider == 'openai' and not config_dict.get('OPENAI_API_KEY'):
                result.add_warning("RAG is enabled with OpenAI embeddings but OPENAI_API_KEY is missing")
    
    def _validate_provider_configurations(self, config_dict: Dict[str, Any], result: ValidationResult):
        """Validate LLM provider configurations."""
        # Validate OpenAI configuration
        if config_dict.get('OPENAI_API_KEY'):
            models = config_dict.get('OPENAI_MODELS', 'gpt-4o,gpt-4o-mini,gpt-3.5-turbo')
            default_model = config_dict.get('OPENAI_DEFAULT_MODEL', 'gpt-4o')
            
            if isinstance(models, str):
                model_list = [m.strip() for m in models.split(',')]
            else:
                model_list = models
            
            if default_model not in model_list:
                result.add_error(f"OpenAI default model '{default_model}' not in available models: {model_list}")
        
        # Validate Gemini configuration
        if config_dict.get('GEMINI_API_KEY'):
            models = config_dict.get('GEMINI_MODELS', 'gemini-1.5-pro,gemini-1.5-flash')
            default_model = config_dict.get('GEMINI_DEFAULT_MODEL', 'gemini-1.5-pro')
            
            if isinstance(models, str):
                model_list = [m.strip() for m in models.split(',')]
            else:
                model_list = models
            
            if default_model not in model_list:
                result.add_error(f"Gemini default model '{default_model}' not in available models: {model_list}")
    
    def _validate_file_paths(self, config_dict: Dict[str, Any], result: ValidationResult):
        """Validate file paths in configuration."""
        # Validate Kokoro TTS paths
        kokoro_model_path = config_dict.get('KOKORO_MODEL_PATH')
        if kokoro_model_path and not Path(kokoro_model_path).exists():
            result.add_warning(f"Kokoro model path does not exist: {kokoro_model_path}")
        
        kokoro_voices_path = config_dict.get('KOKORO_VOICES_PATH')
        if kokoro_voices_path and not Path(kokoro_voices_path).exists():
            result.add_warning(f"Kokoro voices path does not exist: {kokoro_voices_path}")
        
        # Validate output directory
        output_dir = config_dict.get('OUTPUT_DIR', 'output')
        output_path = Path(output_dir)
        if not output_path.exists():
            try:
                output_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created output directory: {output_path}")
            except Exception as e:
                result.add_warning(f"Cannot create output directory {output_dir}: {e}")
    
    def get_loaded_files(self) -> List[Path]:
        """Get list of loaded configuration files.
        
        Returns:
            List of Path objects for loaded .env files
        """
        return self.loaded_files.copy()
    
    def clear_cache(self):
        """Clear the configuration cache."""
        self.config_cache.clear()
        logger.debug("Configuration cache cleared")
    
    def get_last_reload_time(self) -> Optional[float]:
        """Get timestamp of last configuration reload.
        
        Returns:
            Unix timestamp of last reload, or None if never reloaded
        """
        return self._last_reload_time
    
    def watch_config_files(self, callback=None) -> bool:
        """Enable file watching for configuration hot-reloading.
        
        Args:
            callback: Optional callback function to call on file changes
            
        Returns:
            True if watching was enabled, False otherwise
        """
        try:
            from .file_watcher import ConfigFileWatcher
            
            # Initialize file watcher if not already done
            if not hasattr(self, '_file_watcher'):
                self._file_watcher = ConfigFileWatcher(self.base_path)
            
            # Add callback if provided
            if callback:
                self._file_watcher.add_callback(callback)
            
            # Add default callback for reloading configuration
            self._file_watcher.add_callback(self._on_config_file_changed)
            
            # Start watching
            success = self._file_watcher.start_watching()
            
            if success:
                logger.info("Configuration file watching enabled")
            else:
                logger.warning("Failed to enable configuration file watching")
            
            return success
            
        except ImportError as e:
            logger.error(f"File watching requires watchdog library: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to enable file watching: {e}")
            return False
    
    def stop_watching_config_files(self):
        """Stop watching configuration files."""
        if hasattr(self, '_file_watcher'):
            self._file_watcher.stop_watching()
            logger.info("Configuration file watching stopped")
    
    def _on_config_file_changed(self, file_path: str):
        """Handle configuration file changes.
        
        Args:
            file_path: Path to the changed file
        """
        try:
            logger.info(f"Configuration file changed: {file_path}")
            
            # Reload the specific file
            file_name = Path(file_path).name
            self.reload_env_config(file_name)
            
            logger.info(f"Successfully reloaded configuration from: {file_name}")
            
        except Exception as e:
            logger.error(f"Failed to reload configuration after file change: {e}")
    
    def get_file_watcher_status(self) -> dict:
        """Get status of the file watcher.
        
        Returns:
            Dictionary with watcher status information
        """
        if hasattr(self, '_file_watcher'):
            return self._file_watcher.get_status()
        else:
            return {
                'is_watching': False,
                'development_mode': os.getenv('ENVIRONMENT', 'development') == 'development',
                'hot_reload_enabled': False,
                'watched_files': [],
                'callback_count': 0,
                'base_path': str(self.base_path)
            }