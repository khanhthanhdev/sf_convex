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
    CONVEX_DEPLOYMENT_URL: str = "your-convex-deployment-url"  # Replace with actual deployment URL
    
    # AWS/S3 Configuration
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_SESSION_TOKEN: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    
    # Primary outputs bucket and optional base prefix (e.g., "projects/")
    S3_BUCKET: Optional[str] = None
    S3_BASE_PREFIX: str = ""
    
    # Optional public base URL (e.g., CloudFront distribution). If set, URLs will use this base
    S3_PUBLIC_BASE_URL: Optional[str] = None
    
    # Optional KMS key for server-side encryption
    S3_KMS_KEY_ID: Optional[str] = None
    
    # Transfer tuning and behavior
    S3_UPLOAD_ON_WRITE: bool = False
    S3_MAX_CONCURRENCY: int = 8
    S3_MULTIPART_THRESHOLD_MB: int = 64
    
    # Optional custom endpoint (e.g., MinIO) and path-style toggle
    S3_ENDPOINT_URL: Optional[str] = None
    S3_FORCE_PATH_STYLE: bool = False
    
    # Default presigned URL expiration (seconds)
    S3_PRESIGN_EXPIRATION: int = 3600
    
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
