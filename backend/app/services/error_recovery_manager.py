"""
Error Recovery Manager for S3-Convex Integration.

This service implements comprehensive error recovery and data consistency mechanisms
for the S3-Convex integration, handling various error scenarios and ensuring data
integrity across both storage systems.

Key features:
- Compensation transaction logic for partial failures
- Data consistency reconciliation methods
- Circuit breaker pattern for external service failures
- Retry mechanisms with exponential backoff
- Error categorization and recovery strategies
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict

from agents.src.storage.s3_storage import S3Storage
from .convex_s3_sync import ConvexS3Sync, AssetSyncRequest
from backend.convex.types.schema import AssetType
from ..core.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)


class ErrorType(str, Enum):
    """Categories of errors that can occur in S3-Convex operations."""
    S3_UPLOAD_FAILURE = "s3_upload_failure"
    S3_DELETE_FAILURE = "s3_delete_failure"
    CONVEX_SYNC_FAILURE = "convex_sync_failure"
    CONVEX_QUERY_FAILURE = "convex_query_failure"
    DATA_INCONSISTENCY = "data_inconsistency"
    NETWORK_TIMEOUT = "network_timeout"
    AUTHENTICATION_ERROR = "authentication_error"
    PERMISSION_ERROR = "permission_error"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    UNKNOWN_ERROR = "unknown_error"


class RecoveryAction(str, Enum):
    """Types of recovery actions that can be taken."""
    RETRY = "retry"
    COMPENSATE = "compensate"
    ROLLBACK = "rollback"
    RECONCILE = "reconcile"
    FAIL_FAST = "fail_fast"
    CIRCUIT_BREAK = "circuit_break"
    MANUAL_INTERVENTION = "manual_intervention"


class InconsistencyType(str, Enum):
    """Types of data inconsistencies between S3 and Convex."""
    S3_EXISTS_CONVEX_MISSING = "s3_exists_convex_missing"
    CONVEX_EXISTS_S3_MISSING = "convex_exists_s3_missing"
    METADATA_MISMATCH = "metadata_mismatch"
    URL_MISMATCH = "url_mismatch"
    CHECKSUM_MISMATCH = "checksum_mismatch"


class CircuitState(str, Enum):
    """States of the circuit breaker."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class ErrorContext:
    """Context information for an error occurrence."""
    error_type: ErrorType
    error_message: str
    entity_id: str
    entity_type: str
    operation: str
    timestamp: float
    attempt_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryResult:
    """Result of a recovery operation."""
    success: bool
    action_taken: RecoveryAction
    error_context: ErrorContext
    recovery_message: str
    retry_after: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerState:
    """State of a circuit breaker for a specific service."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0
    last_success_time: float = 0
    next_attempt_time: float = 0


class ErrorRecoveryManager:
    """
    Comprehensive error recovery and data consistency manager for S3-Convex integration.
    
    This service handles various error scenarios and implements recovery strategies
    to maintain data consistency between S3 and Convex systems.
    """
    
    def __init__(
        self,
        s3_storage: Optional[S3Storage] = None,
        convex_sync: Optional[ConvexS3Sync] = None
    ):
        """
        Initialize the ErrorRecoveryManager.
        
        Args:
            s3_storage: S3Storage service instance
            convex_sync: ConvexS3Sync service instance
        """
        self.settings = get_settings()
        
        # Initialize services
        if s3_storage is None:
            bucket = self.settings.s3_bucket_name
            base_prefix = getattr(self.settings, 's3_base_prefix', '')
            self.s3_storage = S3Storage(bucket=bucket, base_prefix=base_prefix)
        else:
            self.s3_storage = s3_storage
            
        if convex_sync is None:
            self.convex_sync = ConvexS3Sync()
        else:
            self.convex_sync = convex_sync
        
        # Recovery configuration
        self.max_retry_attempts = 3
        self.base_retry_delay = 1.0  # seconds
        self.max_retry_delay = 60.0  # seconds
        self.exponential_base = 2.0
        
        # Circuit breaker configuration
        self.circuit_failure_threshold = 5
        self.circuit_timeout = 60.0  # seconds
        self.circuit_half_open_max_calls = 3
        
        # Circuit breaker states for different services
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {
            "s3_upload": CircuitBreakerState(),
            "s3_delete": CircuitBreakerState(),
            "convex_sync": CircuitBreakerState(),
            "convex_query": CircuitBreakerState()
        }
        
        # Error tracking
        self.error_history: List[ErrorContext] = []
        self.recovery_history: List[RecoveryResult] = []
        
        logger.info("ErrorRecoveryManager initialized")
    
    async def handle_s3_upload_failure(
        self,
        upload_request: Dict[str, Any],
        error: Exception
    ) -> RecoveryResult:
        """
        Handle S3 upload failures with appropriate recovery strategies.
        
        Args:
            upload_request: Original upload request parameters
            error: Exception that occurred during upload
            
        Returns:
            RecoveryResult indicating the action taken and outcome
        """
        error_type = self._classify_error(error)
        error_context = ErrorContext(
            error_type=error_type,
            error_message=str(error),
            entity_id=upload_request.get("entity_id", "unknown"),
            entity_type=upload_request.get("entity_type", "unknown"),
            operation="s3_upload",
            timestamp=time.time(),
            metadata=upload_request
        )
        
        logger.warning(f"Handling S3 upload failure: {error_context.error_message}")
        
        # Check circuit breaker
        if self._is_circuit_open("s3_upload"):
            return RecoveryResult(
                success=False,
                action_taken=RecoveryAction.CIRCUIT_BREAK,
                error_context=error_context,
                recovery_message="Circuit breaker is open for S3 uploads",
                retry_after=self.circuit_breakers["s3_upload"].next_attempt_time
            )
        
        # Determine recovery action based on error type
        if error_type in [ErrorType.NETWORK_TIMEOUT, ErrorType.RESOURCE_EXHAUSTED]:
            # Retry with exponential backoff
            if error_context.attempt_count < self.max_retry_attempts:
                retry_delay = self._calculate_retry_delay(error_context.attempt_count)
                
                try:
                    await asyncio.sleep(retry_delay)
                    
                    # Retry the upload
                    local_path = upload_request.get("local_path")
                    key = upload_request.get("key")
                    extra_args = upload_request.get("extra_args")
                    
                    if local_path and key:
                        s3_key = self.s3_storage.upload_file(local_path, key, extra_args)
                        
                        # Record success
                        self._record_circuit_success("s3_upload")
                        
                        return RecoveryResult(
                            success=True,
                            action_taken=RecoveryAction.RETRY,
                            error_context=error_context,
                            recovery_message=f"Successfully retried S3 upload after {retry_delay}s delay",
                            metadata={"s3_key": s3_key}
                        )
                    
                except Exception as retry_error:
                    logger.error(f"Retry failed: {str(retry_error)}")
                    self._record_circuit_failure("s3_upload")
                    
                    return RecoveryResult(
                        success=False,
                        action_taken=RecoveryAction.RETRY,
                        error_context=error_context,
                        recovery_message=f"Retry failed: {str(retry_error)}",
                        retry_after=time.time() + self._calculate_retry_delay(error_context.attempt_count + 1)
                    )
            else:
                # Max retries exceeded
                self._record_circuit_failure("s3_upload")
                return RecoveryResult(
                    success=False,
                    action_taken=RecoveryAction.FAIL_FAST,
                    error_context=error_context,
                    recovery_message="Max retry attempts exceeded for S3 upload"
                )
        
        elif error_type in [ErrorType.AUTHENTICATION_ERROR, ErrorType.PERMISSION_ERROR]:
            # These errors typically require manual intervention
            return RecoveryResult(
                success=False,
                action_taken=RecoveryAction.MANUAL_INTERVENTION,
                error_context=error_context,
                recovery_message="Authentication or permission error requires manual intervention"
            )
        
        else:
            # Unknown error - fail fast
            self._record_circuit_failure("s3_upload")
            return RecoveryResult(
                success=False,
                action_taken=RecoveryAction.FAIL_FAST,
                error_context=error_context,
                recovery_message=f"Unknown error type: {error_type}"
            )
    
    async def handle_convex_sync_failure(
        self,
        sync_request: AssetSyncRequest,
        error: Exception
    ) -> RecoveryResult:
        """
        Handle Convex synchronization failures with appropriate recovery strategies.
        
        Args:
            sync_request: Original sync request
            error: Exception that occurred during sync
            
        Returns:
            RecoveryResult indicating the action taken and outcome
        """
        error_type = self._classify_error(error)
        error_context = ErrorContext(
            error_type=error_type,
            error_message=str(error),
            entity_id=sync_request.entity_id,
            entity_type=sync_request.entity_type,
            operation="convex_sync",
            timestamp=time.time(),
            metadata={
                "s3_key": sync_request.s3_key,
                "asset_type": sync_request.asset_type.value,
                "operation": sync_request.operation.value
            }
        )
        
        logger.warning(f"Handling Convex sync failure: {error_context.error_message}")
        
        # Check circuit breaker
        if self._is_circuit_open("convex_sync"):
            return RecoveryResult(
                success=False,
                action_taken=RecoveryAction.CIRCUIT_BREAK,
                error_context=error_context,
                recovery_message="Circuit breaker is open for Convex sync",
                retry_after=self.circuit_breakers["convex_sync"].next_attempt_time
            )
        
        # Determine recovery action
        if error_type in [ErrorType.NETWORK_TIMEOUT, ErrorType.CONVEX_SYNC_FAILURE]:
            # Retry with exponential backoff
            if error_context.attempt_count < self.max_retry_attempts:
                retry_delay = self._calculate_retry_delay(error_context.attempt_count)
                
                try:
                    await asyncio.sleep(retry_delay)
                    
                    # Retry the sync operation
                    if sync_request.entity_type == "scene":
                        s3_assets = {sync_request.asset_type.value: sync_request.s3_url}
                        metadata = {
                            f"{sync_request.asset_type.value}_size": sync_request.size,
                            f"{sync_request.asset_type.value}_checksum": sync_request.checksum
                        }
                        
                        result = await self.convex_sync.sync_scene_assets(
                            sync_request.entity_id,
                            s3_assets,
                            metadata
                        )
                        
                    elif sync_request.entity_type == "session":
                        combined_assets = {sync_request.asset_type.value: sync_request.s3_url}
                        result = await self.convex_sync.sync_session_assets(
                            sync_request.entity_id,
                            combined_assets
                        )
                    
                    # Record success
                    self._record_circuit_success("convex_sync")
                    
                    return RecoveryResult(
                        success=True,
                        action_taken=RecoveryAction.RETRY,
                        error_context=error_context,
                        recovery_message=f"Successfully retried Convex sync after {retry_delay}s delay",
                        metadata={"sync_result": result}
                    )
                    
                except Exception as retry_error:
                    logger.error(f"Convex sync retry failed: {str(retry_error)}")
                    self._record_circuit_failure("convex_sync")
                    
                    return RecoveryResult(
                        success=False,
                        action_taken=RecoveryAction.RETRY,
                        error_context=error_context,
                        recovery_message=f"Retry failed: {str(retry_error)}",
                        retry_after=time.time() + self._calculate_retry_delay(error_context.attempt_count + 1)
                    )
            else:
                # Max retries exceeded - this is a critical error that may require compensation
                self._record_circuit_failure("convex_sync")
                
                # If S3 upload succeeded but Convex sync failed, we have a consistency issue
                compensation_result = await self._compensate_failed_sync(sync_request)
                
                return RecoveryResult(
                    success=False,
                    action_taken=RecoveryAction.COMPENSATE,
                    error_context=error_context,
                    recovery_message=f"Max retries exceeded, compensation attempted: {compensation_result}",
                    metadata={"compensation_result": compensation_result}
                )
        
        else:
            # Other error types
            self._record_circuit_failure("convex_sync")
            return RecoveryResult(
                success=False,
                action_taken=RecoveryAction.FAIL_FAST,
                error_context=error_context,
                recovery_message=f"Convex sync failed with error type: {error_type}"
            )
    
    async def reconcile_data_inconsistency(
        self,
        entity_id: str,
        inconsistency_type: InconsistencyType,
        entity_type: str = "scene"
    ) -> bool:
        """
        Reconcile data inconsistencies between S3 and Convex.
        
        Args:
            entity_id: ID of the entity with inconsistent data
            inconsistency_type: Type of inconsistency detected
            entity_type: Type of entity ("scene" or "session")
            
        Returns:
            True if reconciliation was successful, False otherwise
        """
        logger.info(f"Reconciling {inconsistency_type.value} for {entity_type} {entity_id}")
        
        try:
            if inconsistency_type == InconsistencyType.S3_EXISTS_CONVEX_MISSING:
                # S3 has assets but Convex doesn't know about them
                return await self._reconcile_s3_exists_convex_missing(entity_id, entity_type)
                
            elif inconsistency_type == InconsistencyType.CONVEX_EXISTS_S3_MISSING:
                # Convex has asset references but S3 objects are missing
                return await self._reconcile_convex_exists_s3_missing(entity_id, entity_type)
                
            elif inconsistency_type == InconsistencyType.METADATA_MISMATCH:
                # Asset metadata doesn't match between S3 and Convex
                return await self._reconcile_metadata_mismatch(entity_id, entity_type)
                
            elif inconsistency_type == InconsistencyType.URL_MISMATCH:
                # URLs in Convex don't match actual S3 URLs
                return await self._reconcile_url_mismatch(entity_id, entity_type)
                
            elif inconsistency_type == InconsistencyType.CHECKSUM_MISMATCH:
                # Checksums don't match between S3 and Convex
                return await self._reconcile_checksum_mismatch(entity_id, entity_type)
                
            else:
                logger.error(f"Unknown inconsistency type: {inconsistency_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to reconcile {inconsistency_type.value} for {entity_type} {entity_id}: {str(e)}")
            return False
    
    async def detect_inconsistencies(
        self,
        entity_id: str,
        entity_type: str = "scene"
    ) -> List[InconsistencyType]:
        """
        Detect data inconsistencies between S3 and Convex for an entity.
        
        Args:
            entity_id: ID of the entity to check
            entity_type: Type of entity ("scene" or "session")
            
        Returns:
            List of detected inconsistency types
        """
        logger.info(f"Detecting inconsistencies for {entity_type} {entity_id}")
        
        inconsistencies = []
        
        try:
            # Get Convex data
            if entity_type == "scene":
                convex_doc = await self._get_scene_document(entity_id)
            else:
                convex_doc = await self._get_session_document(entity_id)
            
            if not convex_doc:
                logger.warning(f"No Convex document found for {entity_type} {entity_id}")
                return inconsistencies
            
            # Check each asset field
            asset_fields = self._get_asset_fields_for_entity_type(entity_type)
            
            for field_name, asset_type in asset_fields.items():
                s3_asset_data = convex_doc.get(field_name)
                
                if s3_asset_data:
                    s3_key = s3_asset_data.get('s3Key')
                    s3_url = s3_asset_data.get('s3Url')
                    stored_checksum = s3_asset_data.get('checksum')
                    
                    if s3_key:
                        # Check if S3 object exists
                        try:
                            head_response = self.s3_storage.client.head_object(
                                Bucket=self.s3_storage.bucket,
                                Key=s3_key
                            )
                            
                            # Check URL consistency
                            expected_url = self.s3_storage.url_for(s3_key)
                            if s3_url != expected_url:
                                inconsistencies.append(InconsistencyType.URL_MISMATCH)
                            
                            # Check metadata consistency
                            s3_size = head_response.get('ContentLength', 0)
                            stored_size = s3_asset_data.get('size', 0)
                            if s3_size != stored_size:
                                inconsistencies.append(InconsistencyType.METADATA_MISMATCH)
                            
                            # Check checksum if available
                            if stored_checksum:
                                s3_etag = head_response.get('ETag', '').strip('"')
                                if stored_checksum != f"etag:{s3_etag}":
                                    inconsistencies.append(InconsistencyType.CHECKSUM_MISMATCH)
                            
                        except self.s3_storage.client.exceptions.NoSuchKey:
                            # Convex has reference but S3 object is missing
                            inconsistencies.append(InconsistencyType.CONVEX_EXISTS_S3_MISSING)
                        except Exception as e:
                            logger.error(f"Error checking S3 object {s3_key}: {str(e)}")
            
            # Check for S3 objects that aren't referenced in Convex
            # This would require listing S3 objects under the entity prefix
            # and comparing with Convex references
            
            logger.info(f"Detected {len(inconsistencies)} inconsistencies for {entity_type} {entity_id}")
            return list(set(inconsistencies))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Failed to detect inconsistencies for {entity_type} {entity_id}: {str(e)}")
            return []
    
    def get_circuit_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the current status of all circuit breakers.
        
        Returns:
            Dictionary containing circuit breaker status for each service
        """
        status = {}
        current_time = time.time()
        
        for service_name, breaker in self.circuit_breakers.items():
            status[service_name] = {
                "state": breaker.state.value,
                "failure_count": breaker.failure_count,
                "last_failure_time": breaker.last_failure_time,
                "last_success_time": breaker.last_success_time,
                "next_attempt_time": breaker.next_attempt_time,
                "is_available": not self._is_circuit_open(service_name),
                "time_until_retry": max(0, breaker.next_attempt_time - current_time) if breaker.next_attempt_time > current_time else 0
            }
        
        return status
    
    def reset_circuit_breaker(self, service_name: str) -> bool:
        """
        Manually reset a circuit breaker to closed state.
        
        Args:
            service_name: Name of the service circuit breaker to reset
            
        Returns:
            True if reset was successful, False if service not found
        """
        if service_name in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreakerState()
            logger.info(f"Reset circuit breaker for service: {service_name}")
            return True
        else:
            logger.warning(f"Circuit breaker not found for service: {service_name}")
            return False
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about errors and recovery operations.
        
        Returns:
            Dictionary containing error and recovery statistics
        """
        current_time = time.time()
        one_hour_ago = current_time - 3600
        one_day_ago = current_time - 86400
        
        # Filter recent errors
        recent_errors_1h = [e for e in self.error_history if e.timestamp > one_hour_ago]
        recent_errors_24h = [e for e in self.error_history if e.timestamp > one_day_ago]
        
        # Count errors by type
        error_counts_1h = defaultdict(int)
        error_counts_24h = defaultdict(int)
        
        for error in recent_errors_1h:
            error_counts_1h[error.error_type.value] += 1
            
        for error in recent_errors_24h:
            error_counts_24h[error.error_type.value] += 1
        
        # Recovery statistics
        recent_recoveries_1h = [r for r in self.recovery_history if r.error_context.timestamp > one_hour_ago]
        recent_recoveries_24h = [r for r in self.recovery_history if r.error_context.timestamp > one_day_ago]
        
        successful_recoveries_1h = len([r for r in recent_recoveries_1h if r.success])
        successful_recoveries_24h = len([r for r in recent_recoveries_24h if r.success])
        
        return {
            "error_counts": {
                "last_1h": dict(error_counts_1h),
                "last_24h": dict(error_counts_24h),
                "total": len(self.error_history)
            },
            "recovery_stats": {
                "last_1h": {
                    "total_attempts": len(recent_recoveries_1h),
                    "successful": successful_recoveries_1h,
                    "success_rate": successful_recoveries_1h / len(recent_recoveries_1h) if recent_recoveries_1h else 0
                },
                "last_24h": {
                    "total_attempts": len(recent_recoveries_24h),
                    "successful": successful_recoveries_24h,
                    "success_rate": successful_recoveries_24h / len(recent_recoveries_24h) if recent_recoveries_24h else 0
                },
                "total_attempts": len(self.recovery_history)
            },
            "circuit_breaker_status": self.get_circuit_breaker_status()
        }
    
    # Private helper methods
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify an exception into an ErrorType."""
        error_str = str(error).lower()
        error_type_name = type(error).__name__.lower()
        
        # Network-related errors
        if "timeout" in error_str or "timeout" in error_type_name:
            return ErrorType.NETWORK_TIMEOUT
        
        # Authentication/authorization errors
        if any(keyword in error_str for keyword in ["auth", "credential", "access denied", "forbidden"]):
            return ErrorType.AUTHENTICATION_ERROR
        
        if any(keyword in error_str for keyword in ["permission", "unauthorized", "access denied"]):
            return ErrorType.PERMISSION_ERROR
        
        # Resource exhaustion
        if any(keyword in error_str for keyword in ["quota", "limit", "throttle", "rate"]):
            return ErrorType.RESOURCE_EXHAUSTED
        
        # S3-specific errors
        if any(keyword in error_str for keyword in ["s3", "bucket", "key"]):
            if "upload" in error_str or "put" in error_str:
                return ErrorType.S3_UPLOAD_FAILURE
            elif "delete" in error_str:
                return ErrorType.S3_DELETE_FAILURE
        
        # Convex-specific errors
        if any(keyword in error_str for keyword in ["convex", "sync", "mutation", "query"]):
            if "query" in error_str:
                return ErrorType.CONVEX_QUERY_FAILURE
            else:
                return ErrorType.CONVEX_SYNC_FAILURE
        
        # Data consistency errors
        if any(keyword in error_str for keyword in ["consistency", "mismatch", "inconsistent"]):
            return ErrorType.DATA_INCONSISTENCY
        
        return ErrorType.UNKNOWN_ERROR
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay using exponential backoff."""
        delay = self.base_retry_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_retry_delay)
    
    def _is_circuit_open(self, service_name: str) -> bool:
        """Check if a circuit breaker is open."""
        if service_name not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[service_name]
        current_time = time.time()
        
        if breaker.state == CircuitState.OPEN:
            if current_time >= breaker.next_attempt_time:
                # Transition to half-open
                breaker.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker for {service_name} transitioned to HALF_OPEN")
                return False
            return True
        
        return False
    
    def _record_circuit_failure(self, service_name: str):
        """Record a failure for circuit breaker tracking."""
        if service_name not in self.circuit_breakers:
            return
        
        breaker = self.circuit_breakers[service_name]
        breaker.failure_count += 1
        breaker.last_failure_time = time.time()
        
        if breaker.failure_count >= self.circuit_failure_threshold:
            breaker.state = CircuitState.OPEN
            breaker.next_attempt_time = time.time() + self.circuit_timeout
            logger.warning(f"Circuit breaker for {service_name} opened after {breaker.failure_count} failures")
    
    def _record_circuit_success(self, service_name: str):
        """Record a success for circuit breaker tracking."""
        if service_name not in self.circuit_breakers:
            return
        
        breaker = self.circuit_breakers[service_name]
        breaker.last_success_time = time.time()
        
        if breaker.state == CircuitState.HALF_OPEN:
            # Transition back to closed
            breaker.state = CircuitState.CLOSED
            breaker.failure_count = 0
            logger.info(f"Circuit breaker for {service_name} closed after successful operation")
        elif breaker.state == CircuitState.CLOSED:
            # Reset failure count on success
            breaker.failure_count = 0
    
    async def _compensate_failed_sync(self, sync_request: AssetSyncRequest) -> str:
        """
        Implement compensation logic for failed sync operations.
        
        When Convex sync fails but S3 upload succeeded, we need to either:
        1. Clean up the S3 object (if sync is critical)
        2. Queue the sync for retry later
        3. Mark the asset as "sync_pending"
        """
        try:
            logger.info(f"Compensating for failed sync of {sync_request.s3_key}")
            
            # Option 1: Clean up S3 object to maintain consistency
            # This is the safest approach but results in data loss
            try:
                self.s3_storage.client.delete_object(
                    Bucket=self.s3_storage.bucket,
                    Key=sync_request.s3_key
                )
                logger.info(f"Cleaned up S3 object {sync_request.s3_key} after sync failure")
                return "s3_cleanup_successful"
                
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up S3 object {sync_request.s3_key}: {str(cleanup_error)}")
                
                # Option 2: Mark for manual intervention
                # In a production system, you might queue this for later processing
                # or store it in a "failed_syncs" table for manual review
                logger.warning(f"S3 object {sync_request.s3_key} is orphaned and requires manual cleanup")
                return "manual_intervention_required"
                
        except Exception as e:
            logger.error(f"Compensation failed for {sync_request.s3_key}: {str(e)}")
            return "compensation_failed"
    
    async def _reconcile_s3_exists_convex_missing(self, entity_id: str, entity_type: str) -> bool:
        """Reconcile case where S3 has assets but Convex doesn't know about them."""
        try:
            # List S3 objects for the entity
            prefix = f"{entity_type}s/{entity_id}/"
            manifest = self.s3_storage.get_asset_manifest(prefix)
            
            if not manifest["assets"]:
                return True  # Nothing to reconcile
            
            # Create sync requests for missing assets
            sync_requests = []
            for asset_info in manifest["assets"]:
                # Try to determine asset type from key
                asset_type = self._guess_asset_type_from_key(asset_info["key"])
                if asset_type:
                    sync_request = AssetSyncRequest(
                        entity_id=entity_id,
                        entity_type=entity_type,
                        s3_key=asset_info["key"],
                        s3_url=asset_info["url"],
                        asset_type=asset_type,
                        content_type=asset_info["content_type"],
                        size=asset_info["size"],
                        checksum=asset_info["checksum"]
                    )
                    sync_requests.append(sync_request)
            
            # Batch sync the missing assets
            if sync_requests:
                batch_result = await self.convex_sync.batch_sync_assets(sync_requests)
                success_rate = len(batch_result.successful) / len(sync_requests)
                logger.info(f"Reconciled {len(batch_result.successful)}/{len(sync_requests)} missing assets")
                return success_rate > 0.8  # Consider successful if >80% synced
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to reconcile S3 exists Convex missing for {entity_type} {entity_id}: {str(e)}")
            return False
    
    async def _reconcile_convex_exists_s3_missing(self, entity_id: str, entity_type: str) -> bool:
        """Reconcile case where Convex has asset references but S3 objects are missing."""
        try:
            # Get Convex document
            if entity_type == "scene":
                convex_doc = await self._get_scene_document(entity_id)
            else:
                convex_doc = await self._get_session_document(entity_id)
            
            if not convex_doc:
                return True
            
            # Check each asset field and clear references to missing S3 objects
            asset_fields = self._get_asset_fields_for_entity_type(entity_type)
            updates = {}
            
            for field_name, asset_type in asset_fields.items():
                s3_asset_data = convex_doc.get(field_name)
                
                if s3_asset_data:
                    s3_key = s3_asset_data.get('s3Key')
                    
                    if s3_key:
                        try:
                            # Check if S3 object exists
                            self.s3_storage.client.head_object(
                                Bucket=self.s3_storage.bucket,
                                Key=s3_key
                            )
                        except self.s3_storage.client.exceptions.NoSuchKey:
                            # S3 object is missing, clear the reference
                            updates[field_name] = None
                            logger.info(f"Cleared reference to missing S3 object: {s3_key}")
            
            # Update Convex document if needed
            if updates:
                if entity_type == "scene":
                    await self.convex_sync._update_scene_document(entity_id, updates)
                else:
                    await self.convex_sync._update_session_document(entity_id, updates)
                
                logger.info(f"Updated {entity_type} {entity_id} to remove {len(updates)} missing asset references")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to reconcile Convex exists S3 missing for {entity_type} {entity_id}: {str(e)}")
            return False
    
    async def _reconcile_metadata_mismatch(self, entity_id: str, entity_type: str) -> bool:
        """Reconcile metadata mismatches between S3 and Convex."""
        try:
            # Get current Convex document
            if entity_type == "scene":
                convex_doc = await self._get_scene_document(entity_id)
            else:
                convex_doc = await self._get_session_document(entity_id)
            
            if not convex_doc:
                return True
            
            # Check and update metadata for each asset
            asset_fields = self._get_asset_fields_for_entity_type(entity_type)
            updates = {}
            
            for field_name, asset_type in asset_fields.items():
                s3_asset_data = convex_doc.get(field_name)
                
                if s3_asset_data and s3_asset_data.get('s3Key'):
                    s3_key = s3_asset_data['s3Key']
                    
                    try:
                        # Get current S3 metadata
                        head_response = self.s3_storage.client.head_object(
                            Bucket=self.s3_storage.bucket,
                            Key=s3_key
                        )
                        
                        # Update metadata if different
                        s3_size = head_response.get('ContentLength', 0)
                        s3_content_type = head_response.get('ContentType', 'application/octet-stream')
                        s3_etag = head_response.get('ETag', '').strip('"')
                        
                        updated_asset = s3_asset_data.copy()
                        updated_asset['size'] = s3_size
                        updated_asset['contentType'] = s3_content_type
                        updated_asset['checksum'] = f"etag:{s3_etag}"
                        
                        if updated_asset != s3_asset_data:
                            updates[field_name] = updated_asset
                            logger.info(f"Updated metadata for asset {s3_key}")
                    
                    except Exception as e:
                        logger.error(f"Failed to get S3 metadata for {s3_key}: {str(e)}")
            
            # Apply updates if any
            if updates:
                if entity_type == "scene":
                    await self.convex_sync._update_scene_document(entity_id, updates)
                else:
                    await self.convex_sync._update_session_document(entity_id, updates)
                
                logger.info(f"Updated metadata for {len(updates)} assets in {entity_type} {entity_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to reconcile metadata mismatch for {entity_type} {entity_id}: {str(e)}")
            return False
    
    async def _reconcile_url_mismatch(self, entity_id: str, entity_type: str) -> bool:
        """Reconcile URL mismatches between S3 and Convex."""
        try:
            # Get current Convex document
            if entity_type == "scene":
                convex_doc = await self._get_scene_document(entity_id)
            else:
                convex_doc = await self._get_session_document(entity_id)
            
            if not convex_doc:
                return True
            
            # Check and update URLs for each asset
            asset_fields = self._get_asset_fields_for_entity_type(entity_type)
            updates = {}
            
            for field_name, asset_type in asset_fields.items():
                s3_asset_data = convex_doc.get(field_name)
                
                if s3_asset_data and s3_asset_data.get('s3Key'):
                    s3_key = s3_asset_data['s3Key']
                    stored_url = s3_asset_data.get('s3Url')
                    
                    # Generate correct URL
                    correct_url = self.s3_storage.url_for(s3_key)
                    
                    if stored_url != correct_url:
                        updated_asset = s3_asset_data.copy()
                        updated_asset['s3Url'] = correct_url
                        updates[field_name] = updated_asset
                        logger.info(f"Updated URL for asset {s3_key}")
            
            # Apply updates if any
            if updates:
                if entity_type == "scene":
                    await self.convex_sync._update_scene_document(entity_id, updates)
                else:
                    await self.convex_sync._update_session_document(entity_id, updates)
                
                logger.info(f"Updated URLs for {len(updates)} assets in {entity_type} {entity_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to reconcile URL mismatch for {entity_type} {entity_id}: {str(e)}")
            return False
    
    async def _reconcile_checksum_mismatch(self, entity_id: str, entity_type: str) -> bool:
        """Reconcile checksum mismatches between S3 and Convex."""
        # This is similar to metadata reconciliation but focuses specifically on checksums
        return await self._reconcile_metadata_mismatch(entity_id, entity_type)
    
    async def _get_scene_document(self, scene_id: str) -> Optional[Dict[str, Any]]:
        """Get scene document from Convex."""
        try:
            return self.convex_sync.convex_client.query(
                "getDocument",
                {"collection": "scenes", "id": scene_id}
            )
        except Exception as e:
            logger.error(f"Failed to get scene document {scene_id}: {str(e)}")
            return None
    
    async def _get_session_document(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session document from Convex."""
        try:
            return self.convex_sync.convex_client.query(
                "getDocument",
                {"collection": "videoSessions", "id": session_id}
            )
        except Exception as e:
            logger.error(f"Failed to get session document {session_id}: {str(e)}")
            return None
    
    def _get_asset_fields_for_entity_type(self, entity_type: str) -> Dict[str, AssetType]:
        """Get asset field mappings for an entity type."""
        if entity_type == "scene":
            return {
                'videoAsset': AssetType.VIDEO_CHUNK,
                'sourceCodeAsset': AssetType.SOURCE_CODE,
                'thumbnailAsset': AssetType.THUMBNAIL,
                'subtitleAsset': AssetType.SUBTITLE
            }
        elif entity_type == "session":
            return {
                'combinedVideoAsset': AssetType.COMBINED_VIDEO,
                'combinedSubtitleAsset': AssetType.SUBTITLE,
                'manifestAsset': AssetType.MANIFEST
            }
        else:
            return {}
    
    def _guess_asset_type_from_key(self, s3_key: str) -> Optional[AssetType]:
        """Guess asset type from S3 key path."""
        key_lower = s3_key.lower()
        
        if 'video' in key_lower:
            if 'combined' in key_lower:
                return AssetType.COMBINED_VIDEO
            else:
                return AssetType.VIDEO_CHUNK
        elif 'source' in key_lower or 'code' in key_lower:
            return AssetType.SOURCE_CODE
        elif 'thumbnail' in key_lower:
            return AssetType.THUMBNAIL
        elif 'subtitle' in key_lower:
            return AssetType.SUBTITLE
        elif 'manifest' in key_lower:
            return AssetType.MANIFEST
        
        return None