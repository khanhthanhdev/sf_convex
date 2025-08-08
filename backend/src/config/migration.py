"""
Configuration Migration Utilities for version updates and schema changes.

This module provides utilities for migrating configuration between different
versions of the system, handling schema changes, and ensuring backward compatibility.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from packaging import version

from .models import SystemConfig, ValidationResult


logger = logging.getLogger(__name__)


@dataclass
class MigrationResult:
    """Result of a configuration migration."""
    
    success: bool
    from_version: str
    to_version: str
    changes_made: List[str]
    warnings: List[str]
    errors: List[str]
    backup_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'success': self.success,
            'from_version': self.from_version,
            'to_version': self.to_version,
            'changes_made': self.changes_made,
            'warnings': self.warnings,
            'errors': self.errors,
            'backup_path': self.backup_path,
            'timestamp': datetime.now().isoformat()
        }


@dataclass
class MigrationStep:
    """A single migration step."""
    
    from_version: str
    to_version: str
    description: str
    migration_func: Callable[[Dict[str, Any]], Dict[str, Any]]
    validation_func: Optional[Callable[[Dict[str, Any]], bool]] = None
    
    def apply(self, config_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Apply the migration step.
        
        Args:
            config_data: Configuration data to migrate
            
        Returns:
            Tuple of (migrated_config, changes_made)
        """
        try:
            migrated_data = self.migration_func(config_data.copy())
            
            # Validate if validation function provided
            if self.validation_func and not self.validation_func(migrated_data):
                raise ValueError(f"Migration validation failed for {self.from_version} -> {self.to_version}")
            
            return migrated_data, [f"Applied migration: {self.description}"]
            
        except Exception as e:
            logger.error(f"Migration step failed ({self.from_version} -> {self.to_version}): {e}")
            raise


class ConfigurationMigrator:
    """Service for migrating configuration between versions.
    
    This service handles schema changes, version updates, and ensures
    backward compatibility when upgrading the system.
    """
    
    # Current configuration schema version
    CURRENT_VERSION = "1.0.0"
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize the configuration migrator.
        
        Args:
            config_dir: Configuration directory path
        """
        self.config_dir = config_dir or Path("config")
        self.config_dir.mkdir(exist_ok=True)
        
        # Migration steps registry
        self.migration_steps: List[MigrationStep] = []
        self._register_migration_steps()
        
        logger.info(f"ConfigurationMigrator initialized (current version: {self.CURRENT_VERSION})")
    
    def _register_migration_steps(self):
        """Register all available migration steps."""
        
        # Migration from 0.9.0 to 1.0.0 - Initial centralized config
        self.migration_steps.append(MigrationStep(
            from_version="0.9.0",
            to_version="1.0.0",
            description="Migrate to centralized configuration system",
            migration_func=self._migrate_0_9_0_to_1_0_0,
            validation_func=self._validate_1_0_0_config
        ))
        
        # Migration from 1.0.0 to 1.1.0 - Enhanced monitoring
        self.migration_steps.append(MigrationStep(
            from_version="1.0.0",
            to_version="1.1.0",
            description="Add enhanced monitoring configuration",
            migration_func=self._migrate_1_0_0_to_1_1_0,
            validation_func=self._validate_1_1_0_config
        ))
        
        # Migration from 1.1.0 to 1.2.0 - MCP and Context7 support
        self.migration_steps.append(MigrationStep(
            from_version="1.1.0",
            to_version="1.2.0",
            description="Add MCP servers and Context7 configuration",
            migration_func=self._migrate_1_1_0_to_1_2_0,
            validation_func=self._validate_1_2_0_config
        ))
        
        logger.info(f"Registered {len(self.migration_steps)} migration steps")
    
    def detect_config_version(self, config_data: Dict[str, Any]) -> str:
        """Detect the version of a configuration.
        
        Args:
            config_data: Configuration data to analyze
            
        Returns:
            Detected version string
        """
        # Check for explicit version field
        if 'config_version' in config_data:
            return config_data['config_version']
        
        # Detect version based on structure
        if 'mcp_servers' in config_data and 'context7_config' in config_data:
            return "1.2.0"
        elif 'monitoring_config' in config_data and 'performance_tracking' in config_data.get('monitoring_config', {}):
            return "1.1.0"
        elif 'llm_providers' in config_data and 'rag_config' in config_data:
            return "1.0.0"
        else:
            # Assume legacy configuration
            return "0.9.0"
    
    def needs_migration(self, config_data: Dict[str, Any]) -> bool:
        """Check if configuration needs migration.
        
        Args:
            config_data: Configuration data to check
            
        Returns:
            True if migration is needed
        """
        current_version = self.detect_config_version(config_data)
        return version.parse(current_version) < version.parse(self.CURRENT_VERSION)
    
    def migrate_configuration(self, config_data: Dict[str, Any], target_version: Optional[str] = None) -> MigrationResult:
        """Migrate configuration to target version.
        
        Args:
            config_data: Configuration data to migrate
            target_version: Target version (defaults to current version)
            
        Returns:
            MigrationResult with migration details
        """
        if target_version is None:
            target_version = self.CURRENT_VERSION
        
        current_version = self.detect_config_version(config_data)
        
        logger.info(f"Starting migration from {current_version} to {target_version}")
        
        result = MigrationResult(
            success=False,
            from_version=current_version,
            to_version=target_version,
            changes_made=[],
            warnings=[],
            errors=[]
        )
        
        try:
            # Check if migration is needed
            if version.parse(current_version) >= version.parse(target_version):
                result.success = True
                result.warnings.append(f"Configuration is already at version {current_version} (target: {target_version})")
                return result
            
            # Create backup
            backup_path = self._create_backup(config_data, current_version)
            result.backup_path = str(backup_path) if backup_path else None
            
            # Find migration path
            migration_path = self._find_migration_path(current_version, target_version)
            if not migration_path:
                result.errors.append(f"No migration path found from {current_version} to {target_version}")
                return result
            
            # Apply migrations step by step
            migrated_data = config_data.copy()
            
            for step in migration_path:
                logger.info(f"Applying migration step: {step.description}")
                
                try:
                    migrated_data, changes = step.apply(migrated_data)
                    result.changes_made.extend(changes)
                    
                except Exception as e:
                    result.errors.append(f"Migration step failed: {str(e)}")
                    return result
            
            # Add version field to migrated configuration
            migrated_data['config_version'] = target_version
            migrated_data['migration_timestamp'] = datetime.now().isoformat()
            
            # Final validation
            if not self._validate_migrated_config(migrated_data, target_version):
                result.errors.append("Final validation of migrated configuration failed")
                return result
            
            result.success = True
            logger.info(f"Migration completed successfully: {current_version} -> {target_version}")
            
        except Exception as e:
            result.errors.append(f"Migration failed with exception: {str(e)}")
            logger.error(f"Migration failed: {e}")
        
        return result
    
    def _find_migration_path(self, from_version: str, to_version: str) -> List[MigrationStep]:
        """Find the migration path between two versions.
        
        Args:
            from_version: Starting version
            to_version: Target version
            
        Returns:
            List of migration steps to apply
        """
        # Simple linear migration path for now
        # In a more complex system, you might need graph traversal
        
        path = []
        current = from_version
        
        while version.parse(current) < version.parse(to_version):
            # Find next migration step
            next_step = None
            for step in self.migration_steps:
                if step.from_version == current:
                    next_step = step
                    break
            
            if not next_step:
                logger.error(f"No migration step found from version {current}")
                return []
            
            path.append(next_step)
            current = next_step.to_version
        
        return path
    
    def _create_backup(self, config_data: Dict[str, Any], version: str) -> Optional[Path]:
        """Create a backup of the current configuration.
        
        Args:
            config_data: Configuration data to backup
            version: Current version
            
        Returns:
            Path to backup file or None if failed
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"config_backup_{version}_{timestamp}.json"
            backup_path = self.config_dir / "backup" / backup_filename
            
            # Create backup directory
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write backup
            with open(backup_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Configuration backup created: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Failed to create configuration backup: {e}")
            return None
    
    def _validate_migrated_config(self, config_data: Dict[str, Any], target_version: str) -> bool:
        """Validate migrated configuration.
        
        Args:
            config_data: Migrated configuration data
            target_version: Target version
            
        Returns:
            True if validation passes
        """
        try:
            # Basic structure validation
            required_fields = ['config_version', 'environment', 'llm_providers']
            for field in required_fields:
                if field not in config_data:
                    logger.error(f"Missing required field in migrated config: {field}")
                    return False
            
            # Version-specific validation
            if target_version == "1.0.0":
                return self._validate_1_0_0_config(config_data)
            elif target_version == "1.1.0":
                return self._validate_1_1_0_config(config_data)
            elif target_version == "1.2.0":
                return self._validate_1_2_0_config(config_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    # Migration functions for different version transitions
    
    def _migrate_0_9_0_to_1_0_0(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from legacy config (0.9.0) to centralized config (1.0.0)."""
        migrated = {
            'config_version': '1.0.0',
            'environment': config_data.get('environment', 'development'),
            'debug': config_data.get('debug', False),
            'default_llm_provider': 'openai',
            'llm_providers': {},
            'agent_configs': {},
            'workflow_config': {
                'max_workflow_retries': 3,
                'workflow_timeout_seconds': 3600,
                'enable_checkpoints': True,
                'checkpoint_interval': 300,
                'output_dir': config_data.get('output_dir', 'output'),
                'max_scene_concurrency': config_data.get('max_scene_concurrency', 5),
                'max_topic_concurrency': config_data.get('max_topic_concurrency', 1),
                'max_concurrent_renders': config_data.get('max_concurrent_renders', 4),
                'default_quality': 'medium',
                'use_gpu_acceleration': False,
                'preview_mode': False
            },
            'monitoring_config': {
                'enabled': True,
                'log_level': 'INFO',
                'metrics_collection_interval': 300,
                'performance_tracking': True,
                'error_tracking': True,
                'execution_tracing': True,
                'cpu_threshold': 80.0,
                'memory_threshold': 85.0,
                'execution_time_threshold': 300.0,
                'history_retention_hours': 24
            }
        }
        
        # Migrate LLM provider configurations
        if 'openai_api_key' in config_data:
            migrated['llm_providers']['openai'] = {
                'provider': 'openai',
                'api_key': config_data['openai_api_key'],
                'models': ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'],
                'default_model': 'gpt-4o',
                'enabled': True,
                'timeout': 30,
                'max_retries': 3
            }
        
        if 'anthropic_api_key' in config_data:
            migrated['llm_providers']['anthropic'] = {
                'provider': 'anthropic',
                'api_key': config_data['anthropic_api_key'],
                'models': ['claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022'],
                'default_model': 'claude-3-5-sonnet-20241022',
                'enabled': True,
                'timeout': 30,
                'max_retries': 3
            }
        
        # Migrate RAG configuration
        if config_data.get('use_rag', True):
            migrated['rag_config'] = {
                'enabled': True,
                'embedding_config': {
                    'provider': 'jina',
                    'model_name': config_data.get('embedding_model', 'jina-embeddings-v3'),
                    'dimensions': 1024,
                    'batch_size': 100,
                    'timeout': 30,
                    'max_retries': 3,
                    'device': 'cpu'
                },
                'vector_store_config': {
                    'provider': 'chroma',
                    'collection_name': 'default_collection',
                    'connection_params': {
                        'persist_directory': config_data.get('chroma_db_path', 'data/rag/chroma_db')
                    },
                    'max_results': 50,
                    'distance_metric': 'cosine',
                    'timeout': 30,
                    'max_retries': 3
                },
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'min_chunk_size': 100,
                'default_k_value': 5,
                'similarity_threshold': 0.7,
                'enable_query_expansion': True,
                'enable_semantic_search': True,
                'enable_caching': config_data.get('enable_caching', True),
                'cache_ttl': 3600,
                'max_cache_size': 1000,
                'enable_quality_monitoring': True,
                'quality_threshold': 0.7
            }
        
        # Migrate agent configurations
        migrated['agent_configs'] = {
            'planner_agent': {
                'name': 'planner_agent',
                'llm_config': {},
                'tools': [],
                'max_retries': config_data.get('max_retries', 3),
                'timeout_seconds': 300,
                'enable_human_loop': False,
                'planner_model': 'openai/gpt-4o',
                'temperature': 0.7,
                'print_cost': True,
                'verbose': False,
                'enabled': True
            },
            'code_generator_agent': {
                'name': 'code_generator_agent',
                'llm_config': {},
                'tools': [],
                'max_retries': config_data.get('max_retries', 3),
                'timeout_seconds': 300,
                'enable_human_loop': False,
                'planner_model': 'openai/gpt-4o',
                'temperature': 0.7,
                'print_cost': True,
                'verbose': False,
                'enabled': True
            },
            'renderer_agent': {
                'name': 'renderer_agent',
                'llm_config': {},
                'tools': [],
                'max_retries': config_data.get('max_retries', 3),
                'timeout_seconds': 300,
                'enable_human_loop': False,
                'helper_model': 'openai/gpt-4o-mini',
                'temperature': 0.7,
                'print_cost': True,
                'verbose': False,
                'enabled': True
            }
        }
        
        # Add Langfuse configuration if present
        if config_data.get('use_langfuse', True):
            migrated['monitoring_config']['langfuse_config'] = {
                'enabled': True,
                'secret_key': config_data.get('langfuse_secret_key'),
                'public_key': config_data.get('langfuse_public_key'),
                'host': config_data.get('langfuse_host', 'https://cloud.langfuse.com')
            }
        
        return migrated
    
    def _migrate_1_0_0_to_1_1_0(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from 1.0.0 to 1.1.0 - Enhanced monitoring."""
        migrated = config_data.copy()
        migrated['config_version'] = '1.1.0'
        
        # Enhance monitoring configuration
        monitoring_config = migrated.get('monitoring_config', {})
        
        # Add new monitoring fields
        monitoring_config.update({
            'performance_tracking': monitoring_config.get('performance_tracking', True),
            'error_tracking': monitoring_config.get('error_tracking', True),
            'execution_tracing': monitoring_config.get('execution_tracing', True),
            'cpu_threshold': monitoring_config.get('cpu_threshold', 80.0),
            'memory_threshold': monitoring_config.get('memory_threshold', 85.0),
            'execution_time_threshold': monitoring_config.get('execution_time_threshold', 300.0),
            'history_retention_hours': monitoring_config.get('history_retention_hours', 24)
        })
        
        migrated['monitoring_config'] = monitoring_config
        
        # Add human loop configuration
        migrated['human_loop_config'] = {
            'enabled': True,
            'enable_interrupts': True,
            'timeout_seconds': 300,
            'auto_approve_low_risk': False
        }
        
        # Add Docling configuration
        migrated['docling_config'] = {
            'enabled': True,
            'max_file_size_mb': 50,
            'supported_formats': ['pdf', 'docx', 'txt', 'md'],
            'timeout_seconds': 120
        }
        
        return migrated
    
    def _migrate_1_1_0_to_1_2_0(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from 1.1.0 to 1.2.0 - MCP and Context7 support."""
        migrated = config_data.copy()
        migrated['config_version'] = '1.2.0'
        
        # Add MCP servers configuration
        migrated['mcp_servers'] = {
            'context7': {
                'command': 'uvx',
                'args': ['context7-mcp-server@latest'],
                'env': {},
                'disabled': False,
                'auto_approve': []
            }
        }
        
        # Add Context7 configuration
        migrated['context7_config'] = {
            'enabled': True,
            'default_tokens': 10000,
            'timeout_seconds': 30,
            'cache_responses': True,
            'cache_ttl': 3600
        }
        
        # Add TTS configuration
        migrated.update({
            'kokoro_model_path': None,
            'kokoro_voices_path': None,
            'kokoro_default_voice': 'af',
            'kokoro_default_speed': 1.0,
            'kokoro_default_lang': 'en-us'
        })
        
        return migrated
    
    # Validation functions for different versions
    
    def _validate_1_0_0_config(self, config_data: Dict[str, Any]) -> bool:
        """Validate 1.0.0 configuration structure."""
        required_fields = [
            'config_version', 'environment', 'llm_providers', 
            'agent_configs', 'workflow_config', 'monitoring_config'
        ]
        
        for field in required_fields:
            if field not in config_data:
                logger.error(f"Missing required field for v1.0.0: {field}")
                return False
        
        # Validate agent configs
        required_agents = ['planner_agent', 'code_generator_agent', 'renderer_agent']
        for agent in required_agents:
            if agent not in config_data['agent_configs']:
                logger.error(f"Missing required agent for v1.0.0: {agent}")
                return False
        
        return True
    
    def _validate_1_1_0_config(self, config_data: Dict[str, Any]) -> bool:
        """Validate 1.1.0 configuration structure."""
        if not self._validate_1_0_0_config(config_data):
            return False
        
        # Check for 1.1.0 specific fields
        monitoring_config = config_data.get('monitoring_config', {})
        required_monitoring_fields = [
            'performance_tracking', 'error_tracking', 'execution_tracing'
        ]
        
        for field in required_monitoring_fields:
            if field not in monitoring_config:
                logger.error(f"Missing monitoring field for v1.1.0: {field}")
                return False
        
        # Check for human loop and docling configs
        if 'human_loop_config' not in config_data:
            logger.error("Missing human_loop_config for v1.1.0")
            return False
        
        if 'docling_config' not in config_data:
            logger.error("Missing docling_config for v1.1.0")
            return False
        
        return True
    
    def _validate_1_2_0_config(self, config_data: Dict[str, Any]) -> bool:
        """Validate 1.2.0 configuration structure."""
        if not self._validate_1_1_0_config(config_data):
            return False
        
        # Check for 1.2.0 specific fields
        if 'mcp_servers' not in config_data:
            logger.error("Missing mcp_servers for v1.2.0")
            return False
        
        if 'context7_config' not in config_data:
            logger.error("Missing context7_config for v1.2.0")
            return False
        
        return True
    
    def get_migration_history(self) -> List[Dict[str, Any]]:
        """Get history of configuration migrations.
        
        Returns:
            List of migration records
        """
        history_file = self.config_dir / "migration_history.json"
        
        if not history_file.exists():
            return []
        
        try:
            with open(history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read migration history: {e}")
            return []
    
    def save_migration_record(self, result: MigrationResult):
        """Save migration record to history.
        
        Args:
            result: Migration result to save
        """
        try:
            history_file = self.config_dir / "migration_history.json"
            
            # Load existing history
            history = self.get_migration_history()
            
            # Add new record
            history.append(result.to_dict())
            
            # Keep only last 50 records
            history = history[-50:]
            
            # Save updated history
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            logger.info(f"Migration record saved to {history_file}")
            
        except Exception as e:
            logger.error(f"Failed to save migration record: {e}")
    
    def rollback_migration(self, backup_path: str) -> bool:
        """Rollback to a previous configuration backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if rollback successful
        """
        try:
            backup_file = Path(backup_path)
            if not backup_file.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Load backup configuration
            with open(backup_file, 'r') as f:
                backup_config = json.load(f)
            
            # Save current config as rollback backup
            current_config_file = self.config_dir / "system_config.json"
            if current_config_file.exists():
                rollback_backup = self.config_dir / "backup" / f"rollback_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                rollback_backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(current_config_file, rollback_backup)
            
            # Restore backup
            with open(current_config_file, 'w') as f:
                json.dump(backup_config, f, indent=2)
            
            logger.info(f"Configuration rolled back from {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def cleanup_old_backups(self, keep_days: int = 30):
        """Clean up old backup files.
        
        Args:
            keep_days: Number of days to keep backups
        """
        try:
            backup_dir = self.config_dir / "backup"
            if not backup_dir.exists():
                return
            
            cutoff_time = datetime.now().timestamp() - (keep_days * 24 * 3600)
            
            for backup_file in backup_dir.glob("*.json"):
                if backup_file.stat().st_mtime < cutoff_time:
                    backup_file.unlink()
                    logger.info(f"Deleted old backup: {backup_file}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")