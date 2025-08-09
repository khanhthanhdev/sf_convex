"""
Redis client configuration and utilities
"""
import redis
import json
from typing import Any, Optional, Union
from .settings import settings

class RedisClient:
    """Redis client wrapper with utilities"""
    
    def __init__(self):
        self._client = None
        self._connection_pool = None
        
    @property
    def client(self) -> redis.Redis:
        """Get Redis client instance"""
        if self._client is None:
            self._connection_pool = redis.ConnectionPool.from_url(
                settings.REDIS_URL,
                max_connections=20,
                retry_on_timeout=True,
                decode_responses=True
            )
            self._client = redis.Redis(connection_pool=self._connection_pool)
        return self._client
    
    async def ping(self) -> bool:
        """Check if Redis is available"""
        try:
            return self.client.ping()
        except (redis.ConnectionError, redis.TimeoutError):
            return False
    
    def get(self, key: str) -> Optional[str]:
        """Get value from Redis"""
        try:
            return self.client.get(key)
        except Exception:
            return None
    
    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """Set value in Redis with optional expiration"""
        try:
            return self.client.set(key, value, ex=ex)
        except Exception:
            return False
    
    def get_json(self, key: str) -> Optional[dict]:
        """Get JSON value from Redis"""
        try:
            value = self.client.get(key)
            if value is None:
                return None
            return json.loads(value)
        except Exception:
            return None
    
    def set_json(self, key: str, value: dict, ex: Optional[int] = None) -> bool:
        """Set JSON value in Redis with optional expiration"""
        try:
            json_str = json.dumps(value, separators=(',', ':'))
            return self.client.set(key, json_str, ex=ex)
        except Exception:
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis"""
        try:
            return bool(self.client.delete(key))
        except Exception:
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        try:
            return bool(self.client.exists(key))
        except Exception:
            return False
    
    def get_info(self) -> dict:
        """Get Redis server info"""
        try:
            return self.client.info()
        except Exception:
            return {}
    
    def close(self):
        """Close Redis connection"""
        if self._client:
            self._client.close()
        if self._connection_pool:
            self._connection_pool.disconnect()

# Global Redis client instance
redis_client = RedisClient()
