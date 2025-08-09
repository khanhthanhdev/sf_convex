from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    ENV: str = "dev"
    API_PREFIX: str = "/api/v1"
    
    # Redis/Celery Configuration
    REDIS_URL: str = "redis://localhost:6379"
    CELERY_BROKER_URL: Optional[str] = None  # default to REDIS_URL if None
    CELERY_RESULT_BACKEND: Optional[str] = None
    
    # Convex Configuration
    CONVEX_WEBHOOK_SECRET: str = "dev-secret-change-in-production"
    CONVEX_ACTION_BASE_URL: str = "http://localhost:3000"
    
    # AWS Configuration
    S3_BUCKET: Optional[str] = None
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True
        
    def get_celery_broker_url(self) -> str:
        """Get Celery broker URL, default to Redis URL if not set"""
        return self.CELERY_BROKER_URL or self.REDIS_URL
        
    def get_celery_result_backend(self) -> str:
        """Get Celery result backend, default to Redis URL if not set"""
        return self.CELERY_RESULT_BACKEND or self.REDIS_URL

settings = Settings()
